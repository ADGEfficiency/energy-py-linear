"""Extract results from a solved linear program to pd.DataFrame's."""

import json

import pandas as pd

from energypylinear.defaults import defaults
from energypylinear.logger import logger, set_logging_level
from energypylinear.optimizer import Optimizer

optimizer = Optimizer()


def aggregate_check(debug: pd.DataFrame, non_sum_aggs: dict | None = None) -> str:
    aggs = {k: "sum" for k in debug.keys()}

    if non_sum_aggs:
        for k, v in non_sum_aggs.items():
            aggs[k] = v

    aggregated = debug.agg(aggs)
    aggregated = aggregated.to_dict()
    return json.dumps({f"{aggs[k]}-{k}": v for k, v in aggregated.items()}, indent=2)


def check_electricity_balance(
    simulation: pd.DataFrame,
    verbose: int = defaults.log_level,
) -> pd.DataFrame:
    """Checks the electricity balance."""
    inp = (
        simulation["site-import_power_mwh"]
        + simulation["total-electric_generation_mwh"]
    )
    out = simulation["site-export_power_mwh"] + simulation["total-electric_load_mwh"]
    accumulation = (
        simulation["total-electric_discharge_mwh"]
        - simulation["total-electric_charge_mwh"]
    )

    balance = abs(inp + accumulation - out) < 1e-4

    soc = simulation[[c for c in simulation.columns if "final_soc" in c]].sum(axis=1)
    debug = pd.DataFrame(
        {
            "input": inp,
            "accumulation": accumulation,
            "output": out,
            "balance": balance,
            "import": simulation["site-import_power_mwh"],
            "generation": simulation["total-electric_generation_mwh"],
            "export": simulation["site-export_power_mwh"],
            "load": simulation["total-electric_load_mwh"],
            "charge": simulation["total-electric_charge_mwh"],
            "discharge": simulation["total-electric_discharge_mwh"],
            "loss": simulation["total-electric_loss_mwh"],
            "spills": simulation["total-spills_mwh"],
            "soc": soc,
        }
    )

    aggregated = aggregate_check(debug, {"balance": "all"})
    logger.debug(f"checks.check_electricity_balance, aggs={aggregated}")
    assert balance.all(), aggregated
    return debug


def check_high_temperature_heat_balance(
    simulation: pd.DataFrame,
    total_mapper: dict | None = None,
    verbose: int = defaults.log_level,
) -> pd.DataFrame:
    """Checks the high temperature heat balance."""
    inp = simulation["total-high_temperature_generation_mwh"]
    out = simulation["total-high_temperature_load_mwh"]
    balance = abs(inp - out) < 1e-4

    debug = pd.DataFrame(
        {
            "in": inp,
            "out": out,
            "balance": balance,
        }
    )
    if total_mapper:
        for key in ["high_temperature_generation_mwh", "high_temperature_load_mwh"]:
            for col in total_mapper[key]:
                debug[col] = simulation[col]

    aggregated = aggregate_check(debug, {"balance": "all"})
    logger.debug(f"checks.check_high_temperature_heat_balance, aggs={aggregated}")
    assert balance.all(), aggregated
    return debug


def check_low_temperature_heat_balance(
    simulation: pd.DataFrame,
    total_mapper: dict | None = None,
    verbose: int = defaults.log_level,
) -> pd.DataFrame:
    """Checks the high temperature heat balance."""
    inp = simulation["total-low_temperature_generation_mwh"]
    out = simulation["total-low_temperature_load_mwh"]
    balance = abs(inp - out) < 1e-4

    debug = pd.DataFrame(
        {
            "in": inp,
            "out": out,
            "balance": balance,
        }
    )
    if total_mapper:
        for key in ["low_temperature_generation_mwh", "low_temperature_load_mwh"]:
            for col in total_mapper[key]:
                debug[col] = simulation[col]

    aggregated = aggregate_check(debug, {"balance": "all"})
    logger.debug(f"checks.check_low_temperature_heat_balance, aggs={aggregated}")
    assert balance.all(), aggregated
    return debug


def check_results(
    results: pd.DataFrame,
    total_mapper: dict | None = None,
    verbose: int = defaults.log_level,
    check_valve: bool = False,
    check_evs: bool = False,
) -> dict:
    """Check that our simulation results make sense.

    Args:
        interval_data: input interval data to the simulation.
        simulation: simulation results.
    """
    set_logging_level(logger, verbose)
    electricity_balance = check_electricity_balance(results, verbose)
    ht_balance = check_high_temperature_heat_balance(
        results,
        total_mapper,
        verbose,
    )
    lt_balance = check_low_temperature_heat_balance(
        results,
        total_mapper,
        verbose,
    )

    #  TODO could be refactored into `check_valve_heat_balance`
    if check_valve:
        assert all(
            results["valve-low_temperature_generation_mwh"]
            == results["valve-high_temperature_load_mwh"]
        )

    if check_valve:
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
            for c in results.columns
            if c.startswith("charger-")
            and c.endswith("-charge_binary")
            and "spill" not in c
        ]
        subset = results[cols]
        assert (subset <= 1).all().all()

    return {
        "electricity-balance": electricity_balance,
        "high-temperature-heat-balance": ht_balance,
        "low-temperature-heat-balance": lt_balance,
    }
