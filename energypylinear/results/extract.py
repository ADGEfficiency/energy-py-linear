"""Extract results from a solved linear program to pd.DataFrame's."""
import collections

import numpy as np
import pandas as pd
import pydantic

import energypylinear as epl
from energypylinear.flags import Flags
from energypylinear.interval_data import IntervalData
from energypylinear.logger import logger
from energypylinear.optimizer import Optimizer
from energypylinear.results.checks import validate_results
from energypylinear.results.schema import (quantities, simulation_schema,
                                           spill_quantities)
from energypylinear.results.warnings import warn_spills

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

        feasible: bool
        spill: bool
    """

    simulation: pd.DataFrame
    interval_data: IntervalData
    feasible: bool
    spill: bool

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed: bool = True


def extract_evs_results(
    ivars: "epl.interval_data.IntervalVars", results: dict, i: int, verbose: bool = True
) -> None:
    """Extract simulation result data for the EV asset."""
    pkg = []
    for asset_name in ivars.asset.keys():
        try:
            vas = ivars.filter_evs_array(is_spill=False, i=i, asset_name=asset_name)
            pkg.append(vas)
            if verbose:
                logger.info(
                    "filtering evs-array",
                    func="extract_evs_results",
                    i=i,
                    is_spill=False,
                    asset_name=asset_name,
                )
            else:
                logger.debug(
                    "filtering evs-array",
                    func="extract_evs_results",
                    i=i,
                    is_spill=False,
                    asset_name=asset_name,
                )

        except IndexError:
            pass

    #  extract all of the ev assets
    for evs in pkg:
        asset_name = evs.cfg.name
        assert not evs.is_spill
        ev_cols = [
            "electric_charge_mwh",
            "electric_charge_binary",
            "electric_discharge_mwh",
            "electric_discharge_binary",
        ]

        #  chargers are summed acress each charg event
        for charger_idx, charger_cfg in enumerate(evs.cfg.charger_cfgs):
            for attr in ev_cols:
                name = f"{asset_name}-{charger_cfg.name}-{attr}"
                if verbose:
                    logger.info(
                        "extracting evs-array",
                        func="extract_evs_results",
                        i=i,
                        is_spill=False,
                        asset_name=asset_name,
                        column=name,
                    )
                else:
                    logger.debug(
                        "extracting evs-array",
                        func="extract_evs_results",
                        i=i,
                        is_spill=False,
                        asset_name=asset_name,
                        column=name,
                    )
                results[name].append(
                    sum(
                        [
                            optimizer.value(x)
                            for x in getattr(evs, attr)[0, :, charger_idx]
                        ]
                    )
                )

        #  charge events (non-spill) are summed across each charger
        #  one charge event, multiple chargers
        for charge_event_idx, charge_event_cfg in enumerate(evs.cfg.charge_event_cfgs):
            for attr in [
                "electric_charge_mwh",
                "electric_discharge_mwh",
                "electric_loss_mwh",
            ]:
                name = f"{asset_name}-{charge_event_cfg.name}-{attr}"
                if verbose:
                    logger.info(
                        "extracting evs-array",
                        func="extract_evs_results",
                        i=i,
                        is_spill=False,
                        asset_name=asset_name,
                        column=name,
                    )
                else:
                    logger.debug(
                        "extracting evs-array",
                        func="extract_evs_results",
                        i=i,
                        is_spill=False,
                        asset_name=asset_name,
                        column=name,
                    )
                results[name].append(
                    sum(
                        [
                            optimizer.value(x)
                            for x in getattr(evs, attr)[0, charge_event_idx, :]
                        ]
                    )
                )

        #  socs are for a charge event - one soc per charge event
        for attr in [
            "initial_soc_mwh",
            "final_soc_mwh",
        ]:
            socs = getattr(evs, attr)[0]
            assert isinstance(socs, np.ndarray)
            assert isinstance(evs.cfg.charge_event_cfgs, np.ndarray)

            assert socs.shape == evs.cfg.charge_event_cfgs.shape

            for charge_event_idx, soc in enumerate(socs):
                name = f"{asset_name}-charge-event-{charge_event_idx}-{attr}"
                if verbose:
                    logger.info(
                        "extracting evs-array",
                        func="extract_evs_results",
                        i=i,
                        is_spill=evs.is_spill,
                        asset_name=asset_name,
                        column=name,
                    )
                else:
                    logger.debug(
                        "extracting evs-array",
                        func="extract_evs_results",
                        i=i,
                        is_spill=evs.is_spill,
                        asset_name=asset_name,
                        column=name,
                    )
                results[name].append(soc.value())

    pkg = []
    for asset_name in ivars.asset.keys():
        try:
            vas = ivars.filter_evs_array(is_spill=True, i=i, asset_name=asset_name)
            pkg.append(vas)
        except IndexError:
            pass

    for spill_evs in pkg:
        asset_name = spill_evs.cfg.name
        assert spill_evs.is_spill
        #  spill charger charge & discharge
        for charger_idx, spill_cfg in enumerate(spill_evs.cfg.spill_charger_cfgs):
            for attr in ev_cols:
                name = f"{spill_evs.cfg.name}-{spill_cfg.name}-{attr}"
                if verbose:
                    logger.info(
                        "extracting evs-array",
                        func="extract_evs_results",
                        i=i,
                        is_spill=spill_evs.is_spill,
                        asset_name=asset_name,
                        column=name,
                    )
                else:
                    logger.debug(
                        "extracting evs-array",
                        func="extract_evs_results",
                        i=i,
                        is_spill=spill_evs.is_spill,
                        asset_name=asset_name,
                        column=name,
                    )

                results[name].append(
                    sum(
                        [
                            optimizer.value(x)
                            for x in getattr(spill_evs, attr)[0, :, charger_idx]
                        ]
                    )
                )


def extract_results(
    interval_data: IntervalData,
    ivars: "epl.interval_data.IntervalVars",
    feasible: bool,
    flags: Flags = Flags(),
    verbose: bool = True,
) -> SimulationResult:
    """Creates a simulation result from interval data and a solved linear program.

    This is the highest level interface of this module - it does a number of things:
        - extract data from the linear programming variables,
        - extracts data for many different assets (site, battery etc),
        - adds total columns,
        - validates the results.

    This function returns an object that has the input interval data object,
    and the output simulation results as a pd.DataFrame.

    Args:
        interval_data: input interval data to the simulation.
        vars: linear program variables.
        feasible: whether this linear program solution is feasible.
        verbose: whether to print warnings and messages.
    """
    results = collections.defaultdict(list)
    for i in interval_data.idx:
        #  cheating here with the site asset name - TODO
        site = ivars.filter_site(i, "site")

        results["site-import_power_mwh"].append(site.import_power_mwh.value())
        results["site-export_power_mwh"].append(site.export_power_mwh.value())

        spills = ivars.filter_objective_variables(
            epl.assets.spill.SpillOneInterval, i=i
        )[0]
        if len(spills) > 0:
            for spill in spills:
                assert isinstance(spill, epl.assets.spill.SpillOneInterval)
                for attr in spill_quantities:
                    results[f"{spill.cfg.name}-{attr}"].append(
                        optimizer.value(getattr(spill, attr))
                    )

        if batteries := ivars.filter_objective_variables(
            epl.assets.battery.BatteryOneInterval, i=i
        )[0]:
            for battery in batteries:
                assert isinstance(battery, epl.assets.battery.BatteryOneInterval)
                name = f"{battery.cfg.name}"
                for attr in [
                    "electric_charge_mwh",
                    "electric_charge_binary",
                    "electric_discharge_mwh",
                    "electric_discharge_binary",
                    "electric_loss_mwh",
                    "initial_charge_mwh",
                    "final_charge_mwh",
                    # "efficiency_pct",  TODO this is a float
                ]:
                    results[f"{name}-{attr}"].append(
                        optimizer.value(getattr(battery, attr))
                    )

        if generators := ivars.filter_objective_variables(
            epl.assets.chp.GeneratorOneInterval, i=i
        )[0]:
            for generator in generators:
                assert isinstance(generator, epl.assets.chp.GeneratorOneInterval)
                name = f"{generator.cfg.name}"
                for attr in [
                    "electric_generation_mwh",
                    "gas_consumption_mwh",
                    "high_temperature_generation_mwh",
                    "low_temperature_generation_mwh",
                ]:
                    results[f"{name}-{attr}"].append(getattr(generator, attr).value())

        if boilers := ivars.filter_objective_variables(
            epl.assets.boiler.BoilerOneInterval, i=i
        )[0]:
            for boiler in boilers:
                assert isinstance(boiler, epl.assets.boiler.BoilerOneInterval)
                name = f"{boiler.cfg.name}"
                for attr in ["high_temperature_generation_mwh", "gas_consumption_mwh"]:
                    results[f"{name}-{attr}"].append(getattr(boiler, attr).value())

        #  add results from the valve
        # valves = epl.utils.filter_assets(vars, "valve", i=i)
        if valves := ivars.filter_objective_variables(
            epl.assets.valve.ValveOneInterval, i=i
        )[0]:
            for valve in valves:
                assert isinstance(valve, epl.assets.valve.ValveOneInterval)
                for attr in [
                    "high_temperature_load_mwh",
                    "low_temperature_generation_mwh",
                ]:
                    results[f"{valve.cfg.name}-{attr}"].append(
                        optimizer.value(getattr(valve, attr))
                    )

        if ivars.filter_objective_variables(epl.assets.evs.EVOneInterval, i=i):
            extract_evs_results(ivars, results, i, verbose=verbose)

    def check_array_lengths(results: dict[str, list]) -> None:
        lens = []
        dbg = []
        for k, v in results.items():
            lens.append(len(v))
            dbg.append((k, len(v)))

        assert len(set(lens)) == 1, f"{len(set(lens))} {dbg}"

    check_array_lengths(results)
    simulation = pd.DataFrame(results)

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
        cols = [
            c
            for c in simulation.columns
            if ("-" + col in c) and ("charge-event" not in c)
        ]
        simulation[f"total-{col}"] = simulation[cols].sum(axis=1)
        total_mapper[col] = cols

    total_mapper["spills"] = [
        c
        for c in simulation.columns
        #  not charger- is because charger + charge event are duped
        if ("spill" in c) and ("binary") not in c and ("charger-" in c)
    ]
    simulation["total-spills_mwh"] = simulation[total_mapper["spills"]].sum(axis=1)

    total_mapper["losses"] = [c for c in simulation.columns if "electric_loss_mwh" in c]
    simulation["total-electric_loss_mwh"] = simulation[total_mapper["losses"]].sum(
        axis=1
    )

    simulation["site-electricity_balance_mwh"] = (
        simulation["site-import_power_mwh"] - simulation["site-export_power_mwh"]
    )

    if verbose:
        logger.info("total_mapper", mapper=total_mapper)
    else:
        logger.debug("total_mapper", mapper=total_mapper)

    simulation_schema.validate(simulation)
    validate_results(interval_data, simulation, verbose=verbose)
    spill_occured = warn_spills(simulation, flags)

    return SimulationResult(
        simulation=simulation,
        interval_data=interval_data,
        feasible=feasible,
        spill=spill_occured,
    )
