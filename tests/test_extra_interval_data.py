import pytest

import energypylinear as epl


def test_get_extra_interval_data() -> None:
    """Test that we can pass in and use extra interval data."""

    # TODO - should it be custom or extra...???

    site = epl.Site(
        assets=[
            epl.Battery(
                power_mw=2, capacity_mwh=4, efficiency_pct=0.9, name="small-battery"
            ),
            epl.CHP(
                electric_power_max_mw=50,
                electric_efficiency_pct=0.4,
                high_temperature_efficiency_pct=0.4,
                name="gas-engine-chp",
            ),
            epl.Boiler(high_temperature_generation_max_mw=100),
            epl.Spill(),
            epl.Valve(),
        ],
        electricity_prices=[100, 1000, -20, 40, 45],
        network_charge=[0, 300, 300, 0, 0],
        interval_data=10,
        not_interval_data="hello",
    )
    assert hasattr(site.cfg.interval_data, "network_charge")
    assert not hasattr(site.cfg.interval_data, "not_interval_data")

    # TODO - should raise error with the not_interval_data="hello" - an extra kwarg we cannot process

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
            # here we use the extra / custom interval data
            "interval_data": "network_charge",
        },
    ]
    sim = site.optimize(objective)
    assert "site-network_charge" in sim.results.columns

    # below we check that the extra interval data is repeated
    site = epl.Site(
        assets=[
            epl.Battery(
                power_mw=2, capacity_mwh=4, efficiency_pct=0.9, name="small-battery"
            ),
            epl.CHP(
                electric_power_max_mw=50,
                electric_efficiency_pct=0.4,
                high_temperature_efficiency_pct=0.4,
                name="gas-engine-chp",
            ),
            epl.Boiler(high_temperature_generation_max_mw=100),
            epl.Spill(),
            epl.Valve(),
        ],
        electricity_prices=[100, 1000, -20, 40, 45],
        # network charge is too short, should fail - but only if we aren't trying to repeat
        # instead could test current behaviour, which is to always repeat...
        network_charge=[1, 300, 300, 0],
    )
    sim = site.optimize(objective)
    assert sim.results["site-network_charge"].tolist() == [1, 300, 300, 0, 1]

    # TODO - check we fail if we try to use extra interval data that isn't passed into the site init
