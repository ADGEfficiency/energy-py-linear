"""Extract results from a solved linear program."""
import collections

import numpy as np
import pandas as pd
import pydantic
from rich import print

import energypylinear as epl
from energypylinear.flags import Flags
from energypylinear.interval_data import IntervalData
from energypylinear.optimizer import Optimizer

optimizer = Optimizer()


class SimulationResult(pydantic.BaseModel):
    """The output of a simulation."""

    simulation: pd.DataFrame
    interval_data: IntervalData
    feasible: bool
    spill: bool

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed: bool = True


def extract_results(
    interval_data: IntervalData, vars: dict, feasible: bool, flags: Flags = Flags()
) -> SimulationResult:
    """Creates a simulation result from interval data and a solved linear program.

    Args:
        interval_data: input interval data to the simulation.
        vars: linear program variables.
        feasible: whether this linear program solution is feasible.
    """
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
                    "charge_binary",
                    "discharge_mwh",
                    "discharge_binary",
                    "losses_mwh",
                    "initial_charge_mwh",
                    "final_charge_mwh",
                    # "efficiency_pct",  TODO this is a float
                ]:
                    results[f"{name}-{attr}"].append(
                        optimizer.value(getattr(battery, attr))
                    )

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

            for charger_idx, charger_cfg in enumerate(evs.charger_cfgs[0, 0, :]):
                for attr in ["charge_mwh", "charge_binary"]:
                    results[f"{charger_cfg.name}-{attr}"].append(
                        sum([x.value() for x in getattr(evs, attr)[0, :, charger_idx]])
                    )

            #  want to grab all the charge_mwh for each charge event
            for charge_event_idx, _ in enumerate(evs.charger_cfgs[0, :, 0]):
                for attr in ["charge_mwh"]:
                    results[f"charge-event-{charge_event_idx}-{attr}"].append(
                        sum(
                            [
                                x.value()
                                for x in getattr(evs, attr)[0, charge_event_idx, :]
                            ]
                        )
                    )

            evs = vars["spill-evs-array"][i]
            for charger_idx, charger_cfg in enumerate(evs.charger_cfgs[0, 0, :]):
                for attr in ["charge_mwh", "charge_binary"]:
                    results[f"{charger_cfg.name}-{attr}"].append(
                        sum([x.value() for x in getattr(evs, attr)[0, :, charger_idx]])
                    )
            for charge_event_idx, _ in enumerate(evs.charger_cfgs[0, :, 0]):
                for attr in ["charge_mwh"]:
                    results[f"spill-charge-event-{charge_event_idx}-{attr}"].append(
                        sum(
                            [
                                x.value()
                                for x in getattr(evs, attr)[0, charge_event_idx, :]
                            ]
                        )
                    )

    simulation = pd.DataFrame(results)

    #  add totals for charge events across both the spill and normal chargers
    if len(vars["evs-array"]):
        assert isinstance(interval_data.evs, epl.interval_data.EVIntervalData)
        assert interval_data.evs is not None
        for charge_event_idx, _ in enumerate(interval_data.evs.charge_event_mwh):
            simulation[
                f"charge-event-{charge_event_idx}-total-charge_mwh"
            ] = simulation[
                [
                    f"charge-event-{charge_event_idx}-charge_mwh",
                    f"spill-charge-event-{charge_event_idx}-charge_mwh",
                ]
            ].sum(
                axis=1
            )

    #  add totals
    #  can I do this without pandas??
    for col in [
        "electric_generation_mwh",
        "gas_consumption_mwh",
        "high_temperature_generation_mwh",
    ]:
        cols = [c for c in simulation.columns if (col in c)]
        simulation[col] = simulation[cols].sum(axis=1)

    #  add balances + check them - TODO
    validate_results(interval_data, simulation)

    #  add warnings on the use of any spill asset
    spill_columns = [c for c in simulation.columns if "spill" in c]
    #  filter out binary columns - TODO separate loop while dev
    spill_columns = [c for c in spill_columns if "charge_binary" not in c]
    spill_results = simulation[spill_columns]
    assert isinstance(spill_results, pd.DataFrame)
    spill_occured = spill_results.sum().sum() > 0.0

    spills = spill_results.sum(axis=0).to_dict()
    spills = {k: v for k, v in spills.items() if v > 0}
    if spill_occured and flags.fail_on_spill_asset_use:
        spill_message = f"""
        Spill Occurred!
        {len(spills)} of {spill_results.shape[1]} spill columns
        {spills}
        """
        raise ValueError(spill_message)
    elif spill_occured:
        spill_message = f"""
        [red]Spill Occurred![/]
        {len(spills)} of {spill_results.shape[1]} spill columns
        {spills}
        """
        print(spill_message)

    #  include some interval data in simulation results
    assert isinstance(interval_data.electricity_prices, np.ndarray)
    assert isinstance(interval_data.electricity_carbon_intensities, np.ndarray)
    simulation["electricity_prices"] = interval_data.electricity_prices
    simulation[
        "electricity_carbon_intensities"
    ] = interval_data.electricity_carbon_intensities

    return SimulationResult(
        simulation=simulation,
        interval_data=interval_data,
        feasible=feasible,
        spill=spill_occured,
    )


def validate_results(interval_data: IntervalData, simulation: pd.DataFrame) -> None:
    """Check that our simulation results make sense.

    Args:
        interval_data: input interval data to the simulation.
        simulation: simulation results.
    """

    cols = [
        "import_power_mwh",
        "export_power_mwh",
    ]
    for c in cols:
        assert c in simulation.columns

    if interval_data.evs:
        for charge_event_idx, charge_event_mwh in enumerate(
            interval_data.evs.charge_event_mwh
        ):
            np.testing.assert_almost_equal(
                simulation[f"charge-event-{charge_event_idx}-total-charge_mwh"].sum(),
                charge_event_mwh,
                decimal=5,
            )
        """
        want to check
        - only one charger -> one charge event each interval
        """
        cols = [
            c
            for c in simulation.columns
            if c.startswith("charger-")
            and c.endswith("-charge_binary")
            and "spill" not in c
        ]
        subset = simulation[cols]
        assert (subset <= 1).all().all()
