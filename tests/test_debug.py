"""Tests the debugging tools."""

import energypylinear as epl
from energypylinear.debug import debug_asset, debug_balances, debug_simulation


def test_debug() -> None:
    """Tests that the debugging tools work correctly."""
    asset = epl.Battery(
        electricity_prices=[100, 100],
    )
    simulation = asset.optimize(
        verbose=True,
    )

    debug_simulation(simulation.results)
    debug_balances(simulation.results)
    debug_asset(simulation.results, asset.cfg.name, True)
