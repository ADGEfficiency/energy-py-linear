"""Extract results from a solved linear program to pd.DataFrame's."""

import pandas as pd

import energypylinear as epl
from energypylinear.interval_data import IntervalData
from energypylinear.logger import logger
from energypylinear.optimizer import Optimizer

optimizer = Optimizer()


def check_electricity_balance(
    simulation: pd.DataFrame, verbose: bool = True
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

    #  TODO - bunch of debt here
    #  have this idea of wanting to show the balance without the spill effect
    raw_balance = abs(inp + accumulation - out) < 1e-4
    balance = abs(inp + accumulation - out) < 1e-4

    soc = simulation[[c for c in simulation.columns if "final_soc" in c]].sum(axis=1)
    debug = pd.DataFrame(
        {
            "input": inp,
            "accumulation": accumulation,
            "output": out,
            "raw_balance": raw_balance,
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
    if verbose:
        logger.info("check_electricity_balance", debug=debug.to_dict(orient="list"))
    import json

    assert balance.all(), json.dumps(debug.to_dict(), indent=4)
    return debug


def check_high_temperature_heat_balance(
    total_mapper: dict, simulation: pd.DataFrame, verbose: bool = True
) -> None:
    """Checks the high temperature heat balance."""
    inp = simulation["total-high_temperature_generation_mwh"]
    out = simulation["total-high_temperature_load_mwh"]
    balance = abs(inp - out) < 1e-4

    dbg = pd.DataFrame(
        {
            "in": inp,
            "out": out,
            "balance": balance,
        }
    )
    for key in ["high_temperature_generation_mwh", "high_temperature_load_mwh"]:
        for col in total_mapper[key]:
            dbg[col] = simulation[col]

    if verbose:
        logger.info(
            "check_high_temperature_heat_balance", debug=dbg.to_dict(orient="list")
        )
    assert balance.all()


def check_low_temperature_heat_balance(
    total_mapper: dict, simulation: pd.DataFrame, verbose: bool = True
) -> None:
    """Checks the high temperature heat balance."""
    inp = simulation["total-low_temperature_generation_mwh"]
    out = simulation["total-low_temperature_load_mwh"]
    balance = abs(inp - out) < 1e-4

    dbg = pd.DataFrame(
        {
            "in": inp,
            "out": out,
            "balance": balance,
        }
    )
    for key in ["low_temperature_generation_mwh", "low_temperature_load_mwh"]:
        for col in total_mapper[key]:
            dbg[col] = simulation[col]

    if verbose:
        logger.info(
            "check_low_temperature_heat_balance", debug=dbg.to_dict(orient="list")
        )
    assert balance.all()


def check_results(
    total_mapper, assets: list, results: pd.DataFrame, verbose: bool = True
) -> None:
    """Check that our simulation results make sense.

    Args:
        interval_data: input interval data to the simulation.
        simulation: simulation results.
    """
    check_electricity_balance(results, verbose)
    check_high_temperature_heat_balance(total_mapper, results, verbose)
    check_low_temperature_heat_balance(total_mapper, results, verbose)

    #  TODO could be refactored into `check_valve_heat_balance`
    if any([isinstance(a, epl.assets.valve.Valve) for a in assets]):
        assert all(
            results["valve-low_temperature_generation_mwh"]
            == results["valve-high_temperature_load_mwh"]
        )

    if any([isinstance(a, epl.assets.evs.EVs) for a in assets]):

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
