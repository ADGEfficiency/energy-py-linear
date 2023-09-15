"""Tests the debugging tools."""

import energypylinear as epl
from energypylinear.debug import debug_asset, debug_balances, debug_simulation


def test_debug() -> None:
    """Tests that the debugging tools work correctly."""
    asset = epl.Battery()
    results = asset.optimize(
        electricity_prices=[100, 100],
        verbose=True,
    )
    simulation = results.simulation

    debug_simulation(simulation)
    debug_balances(simulation)
    debug_asset(simulation, asset.cfg.name, True)
