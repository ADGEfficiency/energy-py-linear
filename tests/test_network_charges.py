"""Test that we can simulate network charges.

Should belong in the custom constraints tests really.
"""

import numpy as np

import energypylinear as epl


def test_network_charges() -> None:
    """Test that we can simulate a network charge using extra interval data."""
    # first test nothing happens with no network charge
    site = epl.Site(
        assets=[
            epl.CHP(
                electric_power_max_mw=200,
                electric_efficiency_pct=1.0,
                name="chp",
            ),
        ],
        electricity_prices=np.zeros(5),
        network_charge=[0, 0, 0, 0, 0],
        electric_load_mwh=100,
    )

    objective = [
        {
            "asset_type": "site",
            "variable": "import_power_mwh",
            "interval_data": "electricity_prices",
        },
        {
            "asset_type": "site",
            "variable": "export_power_mwh",
            "interval_data": "electricity_prices",
            "coefficient": -1,
        },
        {
            "asset_type": "*",
            "variable": "gas_consumption_mwh",
            "interval_data": "gas_prices",
        },
        {
            "asset_type": "site",
            "variable": "import_power_mwh",
            "interval_data": "network_charge",
            "coefficient": 1000,
        },
    ]
    sim = site.optimize(objective, verbose=0)
    assert all(sim.results["site-import_power_mwh"] == np.full(5, 100))
    assert all(sim.results["site-export_power_mwh"] == np.zeros(5))
    assert all(sim.results["chp-electric_generation_mwh"] == np.zeros(5))

    # now change the network charge
    # expect that we fire the generator
    site = epl.Site(
        assets=[
            epl.CHP(
                electric_power_max_mw=200,
                electric_efficiency_pct=1.0,
                name="chp",
            ),
        ],
        electricity_prices=np.zeros(5),
        network_charge=[0, 300, 0, 0, 0],
        electric_load_mwh=100,
    )
    sim = site.optimize(objective, verbose=0)
    assert all(sim.results["site-import_power_mwh"] == [100, 0, 100, 100, 100])
    assert all(sim.results["site-export_power_mwh"] == np.zeros(5))
    assert all(sim.results["chp-electric_generation_mwh"] == [0, 100, 0, 0, 0])
