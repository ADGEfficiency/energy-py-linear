"""Extract results from a solved linear program to pd.DataFrame's."""
import collections

import numpy as np
import pandas as pd
import pandera as pa
import pydantic
from rich import print

import energypylinear as epl
from energypylinear.defaults import defaults
from energypylinear.flags import Flags
from energypylinear.interval_data import IntervalData
from energypylinear.optimizer import Optimizer
from energypylinear.results.schema import quantities, simulation_schema

optimizer = Optimizer()


class SimulationResult(pydantic.BaseModel):
    """The output of a simulation.

    Attributes:
        simulation: pd.DataFrame
            Simulation outputs, validated with `simulation_schema`.

        interval_data: IntervalData
            Input interval data to the simulation. EV data structures can be
            multi-dimensional, so we store as a collection of arrays,
            rather than a single DataFrame like `simulation`.
    """

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

        results["site-import_power_mwh"].append(site.import_power_mwh.value())
        results["site-export_power_mwh"].append(site.export_power_mwh.value())

        spills = epl.utils.filter_assets(vars, "spill", i=i)
        if len(spills) > 0:
            for spill in spills:
                for attr in quantities:
                    results[f"{spill.cfg.name}-{attr}"].append(
                        optimizer.value(getattr(spill, attr))
                    )

        #  needs to change with the removal of the batteries flag
        batteries = epl.utils.filter_assets(vars, "battery", i=i)
        if len(batteries) > 0:
            for battery in batteries:
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

        generators = epl.utils.filter_assets(vars, "generator", i=i)
        if generators:
            for generator in generators:
                name = f"{generator.cfg.name}"
                for attr in [
                    "electric_generation_mwh",
                    "gas_consumption_mwh",
                    "high_temperature_generation_mwh",
                    "low_temperature_generation_mwh",
                ]:
                    results[f"{name}-{attr}"].append(getattr(generator, attr).value())

        boilers = epl.utils.filter_assets(vars, "boiler", i=i)
        if boilers:
            for boiler in boilers:
                name = f"{boiler.cfg.name}"
                for attr in ["high_temperature_generation_mwh", "gas_consumption_mwh"]:
                    results[f"{name}-{attr}"].append(getattr(boiler, attr).value())

        #  add results from the valve
        valves = epl.utils.filter_assets(vars, "valve", i=i)
        if len(valves) > 0:
            for valve in valves:
                for attr in [
                    "high_temperature_load_mwh",
                    "low_temperature_generation_mwh",
                ]:
                    results[f"{valve.cfg.name}-{attr}"].append(
                        optimizer.value(getattr(valve, attr))
                    )

        #  hmmm - works a bit different
        ev_arrays = vars.get("evs-array")
        ev_cols = [
            "electric_charge_mwh",
            "electric_charge_binary",
            "electric_discharge_mwh",
            "electric_discharge_binary",
        ]
        if ev_arrays:
            evs = ev_arrays[i]
            for charger_idx, charger_cfg in enumerate(evs.charger_cfgs[0, 0, :]):
                for attr in ev_cols:
                    results[f"{charger_cfg.name}-{attr}"].append(
                        sum([x.value() for x in getattr(evs, attr)[0, :, charger_idx]])
                    )

            #  want to grab all the charge_mwh for each charge event
            for charge_event_idx, _ in enumerate(evs.charger_cfgs[0, :, 0]):
                for attr in [
                    "electric_charge_mwh",
                    "electric_discharge_mwh",
                ]:
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
                for attr in ev_cols:
                    results[f"{charger_cfg.name}-{attr}"].append(
                        sum([x.value() for x in getattr(evs, attr)[0, :, charger_idx]])
                    )

            for charge_event_idx, _ in enumerate(evs.charger_cfgs[0, :, 0]):
                for attr in ["electric_charge_mwh", "electric_discharge_mwh"]:
                    results[f"spill-charge-event-{charge_event_idx}-{attr}"].append(
                        sum(
                            [
                                x.value()
                                for x in getattr(evs, attr)[0, charge_event_idx, :]
                            ]
                        )
                    )

    simulation = pd.DataFrame(results)

    # #  add totals for charge events across both the spill and normal chargers
    # ev_arrays = vars.get("evs-array")
    # if ev_arrays:
    #     assert isinstance(interval_data.evs, epl.interval_data.EVIntervalData)
    #     assert interval_data.evs is not None
    #     for charge_event_idx, _ in enumerate(
    #         interval_data.evs.charge_event_mwh
    #     ):
    #         simulation[
    #             f"charge-event-{charge_event_idx}-total-charge_mwh"
    #         ] = simulation[
    #             [
    #                 f"charge-event-{charge_event_idx}-charge_mwh",
    #                 f"spill-charge-event-{charge_event_idx}-charge_mwh",
    #             ]
    #         ].sum(
    #             axis=1
    #         )

    #  include some interval data in simulation results
    assert isinstance(interval_data.electricity_prices, np.ndarray)
    assert isinstance(interval_data.electricity_carbon_intensities, np.ndarray)
    simulation["electricity_prices"] = interval_data.electricity_prices
    simulation[
        "electricity_carbon_intensities"
    ] = interval_data.electricity_carbon_intensities

    #  add totals
    total_mapper = {}
    for col in quantities:

        """
        #  across all non-spill chargers
        charge-event-0-charge_mwh

        # across all chargers, including spill
        charge-event-0-total-charge_mwh

        # across spill charger for each event
        # don't have the equivilant for each other charger
        spill-charge-event-0-charge_mwh
        """
        cols = [
            c
            for c in simulation.columns
            if ("-" + col in c) and ("charge-event" not in c)
        ]
        simulation[f"total-{col}"] = simulation[cols].sum(axis=1)
        total_mapper[col] = cols

    total_mapper["spills"] = [
        c for c in simulation.columns if ("spill" in c) and ("binary") not in c
    ]
    simulation["total-spills_mwh"] = simulation[total_mapper["spills"]].sum(axis=1)

    total_mapper["losses"] = [c for c in simulation.columns if "-losses_mwh" in c]
    simulation["total-losses_mwh"] = simulation[total_mapper["losses"]].sum(axis=1)

    simulation["site-electricity_balance_mwh"] = (
        simulation["site-import_power_mwh"] - simulation["site-export_power_mwh"]
    )

    print("Total Mapper")
    print(total_mapper)

    simulation_schema.validate(simulation)
    validate_results(interval_data, simulation)
    spill_occured = warn_spills(simulation, flags)

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
    #  TODO
    check_electricity_balance(simulation)

    #  hmmmmmmmmmmmmmmmmmmm TODO move into above
    simulation[
        "load-high_temperature_load_mwh"
    ] = interval_data.high_temperature_load_mwh
    simulation["load-low_temperature_load_mwh"] = interval_data.low_temperature_load_mwh

    check_high_temperature_heat_balance(simulation)
    check_low_temperature_heat_balance(simulation)

    if interval_data.evs:

        #  TODO replace with a check on SOC

        # for charge_event_idx, charge_event_mwh in enumerate(
        #     interval_data.evs.charge_event_mwh
        # ):
        #     np.testing.assert_almost_equal(
        #         simulation[f"charge-event-{charge_event_idx}-total-charge_mwh"].sum(),
        #         charge_event_mwh,
        #         decimal=defaults.decimal_tolerance,
        #     )
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
