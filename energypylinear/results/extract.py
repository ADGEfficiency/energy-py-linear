"""Extract results from a solved linear program to a pd.DataFrame."""
import collections
import json

import numpy as np
import pandas as pd
import pydantic

import energypylinear as epl
from energypylinear.assets.evs import create_evs_array
from energypylinear.defaults import defaults
from energypylinear.flags import Flags
from energypylinear.logger import logger, set_logging_level
from energypylinear.optimizer import Optimizer
from energypylinear.results.checks import check_results
from energypylinear.results.schema import quantities, simulation_schema
from energypylinear.results.warnings import warn_spills
from energypylinear.utils import check_array_lengths

optimizer = Optimizer()


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
    status: "epl.optimizer.OptimizationStatus"
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<energypylinear.SimulationResult feasible:{self.feasible}, rows:{self.results.shape[0]}, cols:{self.results.shape[1]}>"

    def __str__(self) -> str:
        """A string representation of self."""
        return repr(self)


def extract_site_results(
    site: "epl.Site", ivars: "epl.IntervalVars", results: dict, i: int
) -> None:
    """Extract simulation result data for epl.Site."""
    sites = ivars.filter_objective_variables(
        instance_type=epl.assets.site.SiteOneInterval, asset_name=site.cfg.name, i=i
    )
    assert len(sites) == 1
    site_one_interval = sites[0]
    assert isinstance(site_one_interval, epl.assets.site.SiteOneInterval)
    results["site-import_power_mwh"].append(site_one_interval.import_power_mwh.value())
    results["site-export_power_mwh"].append(site_one_interval.export_power_mwh.value())

    assert isinstance(site.cfg.interval_data.electricity_prices, np.ndarray)
    assert isinstance(site.cfg.interval_data.electricity_carbon_intensities, np.ndarray)

    for attr in [
        "electricity_prices",
        "electricity_carbon_intensities",
        "high_temperature_load_mwh",
        "low_temperature_load_mwh",
        "low_temperature_generation_mwh",
        "gas_prices",
        "electric_load_mwh",
    ]:
        results[f"{site.cfg.name}-{attr}"].append(
            getattr(site.cfg.interval_data, attr)[i]
        )


def extract_spill_results(ivars: "epl.IntervalVars", results: dict, i: int) -> None:
    """Extract simulation result data for epl.Spill."""
    if spills := ivars.filter_objective_variables(
        instance_type=epl.assets.spill.SpillOneInterval, i=i
    ):
        for spill in spills:
            assert isinstance(spill, epl.assets.spill.SpillOneInterval)

            spill_quantities = [
                "electric_generation_mwh",
                "electric_load_mwh",
                "high_temperature_generation_mwh",
                "low_temperature_generation_mwh",
                "high_temperature_load_mwh",
                "low_temperature_load_mwh",
                "gas_consumption_mwh",
            ]
            for attr in spill_quantities:
                results[f"{spill.cfg.name}-{attr}"].append(
                    optimizer.value(getattr(spill, attr))
                )


def extract_chp_results(ivars: "epl.IntervalVars", results: dict, i: int) -> None:
    """Extract simulation result data for epl.CHP."""
    if generators := ivars.filter_objective_variables(
        instance_type=epl.assets.chp.CHPOneInterval, i=i
    ):
        for generator in generators:
            assert isinstance(generator, epl.assets.chp.CHPOneInterval)
            for attr in [
                "electric_generation_mwh",
                "gas_consumption_mwh",
                "high_temperature_generation_mwh",
                "low_temperature_generation_mwh",
            ]:
                results[f"{generator.cfg.name}-{attr}"].append(
                    getattr(generator, attr).value()
                )


def extract_boiler_results(ivars: "epl.IntervalVars", results: dict, i: int) -> None:
    """Extract simulation result data for epl.Boiler."""
    if boilers := ivars.filter_objective_variables(
        instance_type=epl.assets.boiler.BoilerOneInterval, i=i
    ):
        for boiler in boilers:
            assert isinstance(boiler, epl.assets.boiler.BoilerOneInterval)
            for attr in ["high_temperature_generation_mwh", "gas_consumption_mwh"]:
                results[f"{boiler.cfg.name}-{attr}"].append(
                    getattr(boiler, attr).value()
                )


def extract_valve_results(ivars: "epl.IntervalVars", results: dict, i: int) -> None:
    """Extract simulation result data for epl.Valve."""
    if valves := ivars.filter_objective_variables(
        instance_type=epl.assets.valve.ValveOneInterval, i=i
    ):
        for valve in valves:
            assert isinstance(valve, epl.assets.valve.ValveOneInterval)
            for attr in [
                "high_temperature_load_mwh",
                "low_temperature_generation_mwh",
            ]:
                results[f"{valve.cfg.name}-{attr}"].append(
                    optimizer.value(getattr(valve, attr))
                )


def extract_battery_results(ivars: "epl.IntervalVars", results: dict, i: int) -> None:
    """Extract simulation result data for epl.Battery."""
    if batteries := ivars.filter_objective_variables(
        instance_type=epl.assets.battery.BatteryOneInterval, i=i
    ):
        for battery in batteries:
            assert isinstance(battery, epl.assets.battery.BatteryOneInterval)
            name = f"{battery.cfg.name}"
            for attr in [
                "electric_charge_mwh",
                "electric_charge_binary",
                "electric_discharge_mwh",
                "electric_discharge_binary",
                "electric_loss_mwh",
                "electric_initial_charge_mwh",
                "electric_final_charge_mwh",
                # "efficiency_pct",  TODO this is a float
            ]:
                results[f"{name}-{attr}"].append(
                    optimizer.value(getattr(battery, attr), clip_from_zero=True)
                )


def extract_evs_results(
    ivars: "epl.IntervalVars",
    results: dict,
    i: int,
    asset_names: set,
    verbose: int | bool = defaults.log_level,
) -> None:
    """Extract simulation result data for epl.EVs."""
    if ivars.filter_objective_variables(
        instance_type=epl.assets.evs.EVOneInterval, i=i
    ):
        pkg = []
        for asset_name in [n for n in asset_names if "evs" in n]:
            evs_array = create_evs_array(ivars, i, asset_name, is_spill=False)
            pkg.append(evs_array)

        #  extract all of the ev assets
        for evs in pkg:
            asset_name = evs.cfg.name
            assert not evs.is_spill
            ev_cols = [
                "electric_charge_mwh",
                "electric_charge_binary",
                "electric_discharge_mwh",
                "electric_discharge_binary",
                "electric_loss_mwh",
            ]

            #  chargers are summed across each charge event
            for charger_idx, charger_cfg in enumerate(evs.cfg.charger_cfgs):
                for attr in ev_cols:
                    name = f"{asset_name}-{charger_cfg.name}-{attr}"
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
                    results[name].append(soc.value())

        pkg = []
        for asset_name in [n for n in asset_names if "evs" in n]:
            evs_array = create_evs_array(ivars, i, asset_name, is_spill=True)
            pkg.append(evs_array)

        for spill_evs in pkg:
            asset_name = spill_evs.cfg.name
            assert spill_evs.is_spill
            #  spill charger charge & discharge
            for charger_idx, spill_cfg in enumerate(spill_evs.cfg.spill_charger_cfgs):
                for attr in ev_cols:
                    name = f"{spill_evs.cfg.name}-{spill_cfg.name}-{attr}"

                    results[name].append(
                        sum(
                            [
                                optimizer.value(x)
                                for x in getattr(spill_evs, attr)[0, :, charger_idx]
                            ]
                        )
                    )


def extract_heat_pump_results(
    ivars: "epl.IntervalVars",
    results: dict,
    i: int,
    verbose: int | bool = defaults.log_level,
) -> None:
    """Extract simulation result data for epl.HeatPump."""
    if heat_pumps := ivars.filter_objective_variables(
        instance_type=epl.assets.heat_pump.HeatPumpOneInterval, i=i
    ):
        for heat_pump in heat_pumps:
            for attr in [
                "electric_load_mwh",
                "low_temperature_load_mwh",
                "high_temperature_generation_mwh",
            ]:
                name = f"{heat_pump.cfg.name}-{attr}"
                results[name].append(optimizer.value(getattr(heat_pump, attr)))


def extract_renewable_generator_results(
    ivars: "epl.IntervalVars",
    results: dict,
    i: int,
    verbose: int | bool = defaults.log_level,
) -> None:
    """Extract simulation result data for epl.RenewableGenerator."""
    if renewables := ivars.filter_objective_variables(
        instance_type=epl.assets.renewable_generator.RenewableGeneratorOneInterval, i=i
    ):
        fields = ["electric_generation_mwh"]
        for renewable in renewables:
            for attr in fields:
                name = f"{renewable.cfg.name}-{attr}"
                results[name].append(optimizer.value(getattr(renewable, attr)))


def add_totals(
    results: pd.DataFrame,
) -> dict:
    """Creates total columns in the simulation.results pd.DataFrame"""
    total_mapper = {}
    for col in quantities:
        cols = [
            c for c in results.columns if ("-" + col in c) and ("charge-event" not in c)
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

    total_mapper["losses"] = [
        c
        for c in results.columns
        if "electric_loss_mwh" in c and "evs-charger" not in c
    ]
    results["total-electric_loss_mwh"] = results[total_mapper["losses"]].sum(axis=1)
    results["site-electricity_balance_mwh"] = (
        results["site-import_power_mwh"] - results["site-export_power_mwh"]
    )
    return total_mapper


def extract_results(
    site: "epl.assets.site.Site",
    assets: list,
    ivars: "epl.IntervalVars",
    feasible: bool,
    status: "epl.optimizer.OptimizationStatus",
    flags: Flags = Flags(),
    verbose: int | bool = defaults.log_level,
) -> SimulationResult:
    """Extracts simulation results from the site, assets and linear program data.

    Also adds total columns & performs checks like energy balances.

    This function returns the output simulation results as a single pd.DataFrame.
    """
    set_logging_level(logger, level=verbose)

    """
    TODO

    if we have asset data that is longer than the anchor input data (prices or carbon intensities)
    this gets automatically clipped

    not sure what to do here yet...

    ```
    electricity_carbon_intensities = [-1.0, 0.1, 1.0]
    electric_generation_mwh=[50, 50, 50, 50],
    asset = epl.RenewableGenerator(
        electric_generation_mwh=electric_generation_mwh,
        name="wind",
        electric_generation_lower_bound_pct=0.0,
        electricity_carbon_intensities=electricity_carbon_intensities,
    )
    ```

    #  validate that we have all the data we need
    # assert len(site.cfg.interval_data.idx) == len(ivars.objective_variables)
    # for asset in assets:
    #     if hasattr(asset.cfg, "interval_data"):
    #         assert len(asset.cfg.interval_data.idx) == len(ivars.objective_variables)
    """

    # constraints = site.optimizer.constraints()
    # name = "site_electricity_balance,i:0"
    # constraint = constraints[name]
    # for var, coeff in constraint.items():
    #     print(f"{var.name}: Value={var.value()}, Coefficient={coeff}")

    #  extract linear program results from the assets
    lp_results: dict[str, list] = collections.defaultdict(list)
    for i in site.cfg.interval_data.idx:
        extract_site_results(site, ivars, lp_results, i)
        extract_spill_results(ivars, lp_results, i)
        extract_battery_results(ivars, lp_results, i)
        extract_chp_results(ivars, lp_results, i)
        extract_boiler_results(ivars, lp_results, i)
        extract_valve_results(ivars, lp_results, i)
        extract_evs_results(
            ivars,
            lp_results,
            i,
            asset_names={asset.cfg.name for asset in assets},
            verbose=verbose,
        )
        extract_heat_pump_results(ivars, lp_results, i, verbose=verbose)
        extract_renewable_generator_results(ivars, lp_results, i, verbose=verbose)

    check_array_lengths(lp_results)
    results = pd.DataFrame(lp_results)

    #  add total columns to the results df
    total_mapper = add_totals(results)
    pretty = json.dumps(total_mapper, indent=2)
    logger.debug(f"results.extract: total_mapper={pretty}")

    if feasible:
        simulation_schema.validate(results)
        check_results(
            results,
            total_mapper=total_mapper,
            verbose=verbose,
            check_valve=any([isinstance(a, epl.Valve) for a in assets]),
            check_evs=any([isinstance(a, epl.EVs) for a in assets]),
        )
    spill_occured = warn_spills(results, flags, verbose=verbose)

    return SimulationResult(
        status=status,
        site=site,
        assets=assets,
        results=results,
        feasible=feasible,
        spill=spill_occured,
    )
