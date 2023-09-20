"""Extract results from a solved linear program to pd.DataFrame's."""
import collections
import typing

import numpy as np
import pandas as pd
import pydantic

import energypylinear as epl
from energypylinear.flags import Flags
from energypylinear.interval_data import IntervalData
from energypylinear.logger import logger
from energypylinear.optimizer import Optimizer
from energypylinear.results.checks import check_results
from energypylinear.results.schema import (
    quantities,
    simulation_schema,
    spill_quantities,
)
from energypylinear.results.warnings import warn_spills

optimizer = Optimizer()



def check_array_lengths(results: dict[str, list]) -> None:
    """Check that all lists in the results dictionary have the same length.

    Args:
        results (dict[str, list]):
            Dictionary containing lists whose lengths need to be checked.

    Raises:
        AssertionError: If lists in the dictionary have different lengths.
    """
    #  TODO move to utils
    lens = []
    dbg = []
    for k, v in results.items():
        lens.append(len(v))
        dbg.append((k, len(v)))
    assert len(set(lens)) == 1, f"{len(set(lens))} {dbg}"


class SimulationResult(pydantic.BaseModel):
    """The output of a simulation.

    Attributes:
        site: the site used to construct the optimization problem
        assets: assets used in the optimization
        results: simulation results
        feasible: whether the linear program was feasible
        spill: whether the spill asset was used to make the program feasible
    """

    site: "epl.assets.site.Site"
    assets: list
    results: pd.DataFrame
    feasible: bool
    spill: bool

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed: bool = True


def extract_site_results(
    site: "epl.Site", ivars: "epl.interval_data.IntervalVars", results: dict, i: int
):
    site_one_interval = ivars.filter_site(i, site.cfg.name)
    results["site-import_power_mwh"].append(site_one_interval.import_power_mwh.value())
    results["site-export_power_mwh"].append(site_one_interval.export_power_mwh.value())

    assert isinstance(site.cfg.interval_data.electricity_prices, np.ndarray)
    assert isinstance(site.cfg.interval_data.electricity_carbon_intensities, np.ndarray)

    for attr in [
        "electricity_prices",
        "electricity_carbon_intensities",
        "high_temperature_load_mwh",
        "low_temperature_load_mwh",
        "low_temperature_generation_mwh"
    ]:
        results[f"{site.cfg.name}-{attr}"].append(
            getattr(site.cfg.interval_data, attr)[i]
        )


def extract_spill_results(
    ivars: "epl.interval_data.IntervalVars", results: dict, i: int
):
    if spills := ivars.filter_objective_variables(
        epl.assets.spill.SpillOneInterval, i=i
    )[0]:
        for spill in spills:
            assert isinstance(spill, epl.assets.spill.SpillOneInterval)
            for attr in spill_quantities:
                results[f"{spill.cfg.name}-{attr}"].append(
                    optimizer.value(getattr(spill, attr))
                )


def extract_generator_results(
    ivars: "epl.interval_data.IntervalVars", results: dict, i: int
):
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


def extract_boiler_results(
    ivars: "epl.interval_data.IntervalVars", results: dict, i: int
):
    if boilers := ivars.filter_objective_variables(
        epl.assets.boiler.BoilerOneInterval, i=i
    )[0]:
        for boiler in boilers:
            assert isinstance(boiler, epl.assets.boiler.BoilerOneInterval)
            name = f"{boiler.cfg.name}"
            for attr in ["high_temperature_generation_mwh", "gas_consumption_mwh"]:
                results[f"{name}-{attr}"].append(getattr(boiler, attr).value())


def extract_valve_results(
    ivars: "epl.interval_data.IntervalVars", results: dict, i: int
):
    #  add results from the valve
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


def extract_battery_results(
    ivars: "epl.interval_data.IntervalVars", results: dict, i: int
):
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


def extract_evs_results(
    ivars: "epl.interval_data.IntervalVars", results: dict, i: int, verbose: bool = True
) -> None:
    """Extract simulation result data for the EV asset."""
    if ivars.filter_objective_variables(epl.assets.evs.EVOneInterval, i=i):
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
            for charge_event_idx, charge_event_cfg in enumerate(
                evs.cfg.charge_event_cfgs
            ):
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


def extract_heat_pump_results(
    ivars: "epl.interval_data.IntervalVars", results: dict, i: int, verbose: bool = True
) -> None:
    """Extract simulation result data for the Heat Pump asset."""
    if heat_pumps := ivars.filter_objective_variables(
        epl.assets.heat_pump.HeatPumpOneInterval, i=i
    )[0]:
        for heat_pump in heat_pumps:
            for attr in [
                "electric_load_mwh",
                "low_temperature_load_mwh",
                "high_temperature_generation_mwh",
            ]:
                name = f"{heat_pump.cfg.name}-{attr}"
                results[name].append(optimizer.value(getattr(heat_pump, attr)))


def extract_renewable_generator_results(
    ivars: "epl.interval_data.IntervalVars", results: dict, i: int, verbose: bool = True
):
    if renewables := ivars.filter_objective_variables(
        epl.assets.renewable_generator.RenewableGeneratorOneInterval, i=i
    )[0]:
        #  TODO this could be made generic
        from energypylinear.assets.renewable_generator import (
            RenewableGeneratorIntervalData,
        )

        fields = list(RenewableGeneratorIntervalData.__fields__.keys())
        fields.remove("idx")
        for renewable in renewables:
            for attr in fields:
                name = f"{renewable.cfg.name}-{attr}"
                results[name].append(optimizer.value(getattr(renewable, attr)))


def add_totals(
    results: pd.DataFrame,
) -> dict:
    total_mapper = {}
    for col in quantities:
        cols = [
            c
            for c in results.columns
            if ("-" + col in c) and ("charge-event" not in c)
        ]
        results[f"total-{col}"] = results[cols].sum(axis=1)
        total_mapper[col] = cols

    total_mapper["spills"] = [
        c
        for c in results.columns
        #  not charger- is because charger + charge event are duped
        if ("spill" in c) and ("binary") not in c and ("charger-" in c)
    ]
    results["total-spills_mwh"] = results[total_mapper["spills"]].sum(axis=1)

    total_mapper["losses"] = [c for c in results.columns if "electric_loss_mwh" in c]
    results["total-electric_loss_mwh"] = results[total_mapper["losses"]].sum(
        axis=1
    )
    results["site-electricity_balance_mwh"] = (
        results["site-import_power_mwh"] - results["site-export_power_mwh"]
    )
    return total_mapper


def extract_results(
    site: "epl.assets.site.Site",
    assets: list,
    ivars: "epl.interval_data.IntervalVars",
    feasible: bool,
    flags: Flags = Flags(),
    verbose: bool = True,
) -> SimulationResult:
    """Extracts simulation results from the site, assets and linear program data.

    Also adds total columns & performs checks like energy balances.

    This function returns the output simulation results as a single pd.DataFrame.
    """

    #  validate that we have all the data we need
    assert len(site.cfg.interval_data.idx) == len(ivars.objective_variables)
    for asset in assets:
        assert len(asset.cfg.interval_data.idx) == len(ivars.objective_variables)

    #  extract linear program results from the assets
    lp_results = collections.defaultdict(list)
    for i in site.cfg.interval_data.idx:
        extract_site_results(site, ivars, lp_results, i)
        extract_spill_results(ivars, lp_results, i)
        extract_battery_results(ivars, lp_results, i)
        extract_generator_results(ivars, lp_results, i)
        extract_boiler_results(ivars, lp_results, i)
        extract_valve_results(ivars, lp_results, i)
        extract_evs_results(ivars, lp_results, i, verbose=verbose)
        extract_heat_pump_results(ivars, lp_results, i, verbose=verbose)
        extract_renewable_generator_results(ivars, lp_results, i, verbose=verbose)

    check_array_lengths(lp_results)
    results = pd.DataFrame(lp_results)

    #  add total columns
    total_mapper = add_totals(results)

    if verbose:
        logger.info("total_mapper", mapper=total_mapper)
    else:
        logger.debug("total_mapper", mapper=total_mapper)


    simulation_schema.validate(results)
    check_results(total_mapper, assets, results, verbose=verbose)
    spill_occured = warn_spills(results, flags, verbose=verbose)

    return SimulationResult(
        site=site,
        assets=assets,
        results=results,
        feasible=feasible,
        spill=spill_occured,
    )
