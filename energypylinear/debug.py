"""one off debug stuff"""
import pandas as pd
from rich import print

from energypylinear.results.checks import (
    check_high_temperature_heat_balance,
    check_low_temperature_heat_balance,
)


def _debug_column(simulation: pd.DataFrame, col: str) -> None:
    """Print a subset of one column for debugging"""
    cols = [c for c in simulation.columns if col in c]
    subset = simulation[cols]
    subset.columns = [c.replace(col, "") for c in subset.columns]
    print(col)
    print(subset)


def debug_simulation(simulation: pd.DataFrame) -> None:
    """be great to debug the balances here as well"""
    print("[red]DEBUG[/red]")
    debug = [
        "site-import_power_mwh",
        "site-export_power_mwh",
        "site-electricity_carbon_intensities",
    ]
    print(simulation[debug])

    _debug_column(simulation, "electric_charge_mwh")
    _debug_column(simulation, "electric_discharge_mwh")
    _debug_column(simulation, "initial_soc_mwh")
    _debug_column(simulation, "final_soc_mwh")
    _debug_column(simulation, "electric_loss_mwh")


def debug_balances(simulation: pd.DataFrame) -> None:
    """runs balance checks"""
    check_high_temperature_heat_balance(simulation, verbose=True)
    check_low_temperature_heat_balance(simulation, verbose=True)


def debug_asset(
    simulation: pd.DataFrame, name: str, verbose: bool = True
) -> pd.DataFrame:
    """Extracts result columns for a singel asset."""
    cols = [c for c in simulation.columns if name in c]
    if verbose:
        print(simulation[cols])
    return simulation[cols]
