import collections
import typing

import pandas as pd
import pulp

floats = typing.Sequence[float]
import numpy as np
import pydantic

from energypylinear.defaults import defaults


class EVIntervalData(pydantic.BaseModel):
    charge_events: typing.Union[list[list[int]], np.ndarray]
    idx: typing.Any = []
    charge_event_mwh: typing.Union[list[int], np.ndarray]

    class Config:
        arbitrary_types_allowed = True

    @pydantic.validator("idx", always=True, pre=True)
    def setup_idx(cls, value, values):
        return np.arange(values["charge_events"].shape[0])

    @pydantic.root_validator()
    def validate_all_things(cls, values):

        #  only_positive or zero charge_event_mwh
        assert all(
            values["charge_event_mwh"] >= 0
        ), "charge_event_mwh has negative values"

        assert all(
            np.array(values["charge_events"]).sum(axis=0) > 0
        ), "sum across axis=0"

        assert (
            values["idx"].shape[0] == values["charge_events"].shape[0]
        ), "charge_event_mwh not equal to length of electricitiy prices."
        assert (
            values["charge_events"].shape[1] == values["charge_event_mwh"].shape[0]
        ), "charge_events not equal to charge_event_mwh"
        return values


class IntervalData(pydantic.BaseModel):
    electricity_prices: floats
    gas_prices: typing.Union[float, floats, None] = None
    electricity_carbon_intensities: typing.Union[float, floats, None] = None

    high_temperature_load_mwh: typing.Union[float, floats, None] = None
    low_temperature_load_mwh: typing.Union[float, floats, None] = None
    idx: typing.Any = []

    evs: typing.Union[EVIntervalData, None] = None

    @pydantic.validator("evs")
    def validate_evs(cls, evs, values, field):
        assert all(evs.idx == values["idx"])
        return evs

    @pydantic.validator(
        "gas_prices",
        "electricity_carbon_intensities",
        "high_temperature_load_mwh",
        "low_temperature_load_mwh",
        pre=True,
        always=True,
    )
    def validate_float(cls, value, values, field):
        if isinstance(value, (float, int)):
            return [value] * len(values["electricity_prices"])

        elif value is None:
            return [getattr(defaults, field.name)] * len(values["electricity_prices"])

        else:
            assert len(value) == len(values["electricity_prices"])
            return value

    @pydantic.validator("idx", always=True)
    def setup_idx(cls, value, values):
        return list(range(len(values["electricity_prices"])))


def validate_results(results: pd.DataFrame) -> None:
    """
    validations TODO
    - don't import / export in same interval
    - column names (types?)
    - validate against interval data - lengths
    """

    cols = [
        "import_power_mwh",
        "export_power_mwh",
    ]
    for c in cols:
        assert c in results.columns


def extract_results(interval_data: IntervalData, vars: dict) -> pd.DataFrame:
    results = collections.defaultdict(list)
    for i in interval_data.idx:
        site = vars["sites"][i]

        results["import_power_mwh"].append(site.import_power_mwh.value())
        results["export_power_mwh"].append(site.export_power_mwh.value())

        spill = vars["spills"][i]
        for attr in [
            "electric_generation_mwh",
            "high_temperature_generation_mwh",
            "electric_load_mwh",
            "low_temperature_load_mwh",
        ]:
            name = f"{spill.cfg.name}"
            results[f"{name}-{attr}"].append(getattr(spill, attr).value())

        if len(vars["batteries"]) > 0:
            for battery in vars["batteries"][i]:
                name = f"{battery.cfg.name}"
                for attr in [
                    "charge_mwh",
                    "discharge_mwh",
                    "losses_mwh",
                    "initial_charge_mwh",
                    "final_charge_mwh",
                    # "efficiency_pct",  TODO this is a float
                ]:
                    results[f"{name}-{attr}"].append(getattr(battery, attr).value())

        if len(vars["generators"]) > 0:
            for generator in vars["generators"][i]:
                name = f"{generator.cfg.name}"
                for attr in [
                    "electric_generation_mwh",
                    "gas_consumption_mwh",
                    "high_temperature_generation_mwh",
                ]:
                    results[f"{name}-{attr}"].append(getattr(generator, attr).value())

        if len(vars["boilers"]):
            boilers = vars["boilers"][i]
            for boiler in boilers:
                name = f"{boiler.cfg.name}"
                for attr in ["high_temperature_generation_mwh", "gas_consumption_mwh"]:
                    results[f"{name}-{attr}"].append(getattr(boiler, attr).value())

        if len(vars["evs-array"]):
            evs = vars["evs-array"][i]
            """
            for each timestep (i)
                for each charger
                    select all charge events
            """
            for charger_idx, charger_cfg in enumerate(evs.charger_cfgs[0, 0, :]):
                for attr in ["charge_mwh", "charge_binary"]:
                    results[f"{charger_cfg.name}-{attr}"].append(
                        sum([x.value() for x in getattr(evs, attr)[0, :, charger_idx]])
                    )
        if len(vars["spill-evs-array"]):
            evs = vars["spill-evs-array"][i]
            for charger_idx, charger_cfg in enumerate(evs.charger_cfgs[0, 0, :]):
                for attr in ["charge_mwh", "charge_binary"]:
                    results[f"{charger_cfg.name}-{attr}"].append(
                        sum([x.value() for x in getattr(evs, attr)[0, :, charger_idx]])
                    )

    #  add totals
    #  can I do this without pandas??
    results = pd.DataFrame(results)
    for col in [
        "electric_generation_mwh",
        "gas_consumption_mwh",
        "high_temperature_generation_mwh",
    ]:
        cols = [c for c in results.columns if (col in c)]
        results[f"total-{col}"] = results[cols].sum(axis=1)

    #  add balances + check them - TODO
    validate_results(results)

    #  add warnings on the spill generator + boiler use
    return results
