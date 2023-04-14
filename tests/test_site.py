"""Test the site API."""
import numpy as np

import energypylinear as epl
from energypylinear.defaults import defaults


def test_site() -> None:
    """Tests the epl.Site() API"""
    site = epl.Site(
        assets=[
            epl.Battery(),
            epl.Battery(name="fast-battery"),
            epl.Generator(
                electric_power_max_mw=100,
                electric_efficiency_pct=0.3,
                high_temperature_efficiency_pct=0.5,
            ),
        ]
    )

    results = site.optimize(electricity_prices=[100, 1000, -20, 40, 50])
    print(results.simulation)
    print(results.simulation.columns)

    simulation = results.simulation

    """
    this is a bit of a regression test

    first interval we both charge and generate max electricity
    second interval we discharge and generate
    """
    np.testing.assert_array_almost_equal(
        simulation["site-import_power_mwh"],
        [0, 0, 4.0, 0.44444, 0.0],
        decimal=defaults.decimal_tolerance,
    )
    np.testing.assert_array_almost_equal(
        simulation["site-export_power_mwh"],
        [96.0, 103.6, 0, 0, 4.0],
        decimal=defaults.decimal_tolerance,
    )

    """
    how to check simulation
    - plot of import/export power versus price
    - plot of asset charge / discharge versus price
    """
