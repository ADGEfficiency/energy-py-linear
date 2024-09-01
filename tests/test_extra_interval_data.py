"""Test that we can use custom interval data."""
import numpy as np
import pytest

import energypylinear as epl
from energypylinear.data_generation import generate_random_ev_input_data
from energypylinear.defaults import defaults
from tests.test_custom_objectives import asset_names, get_assets


def test_get_custom_interval_data() -> None:
    """Test that we can pass in and use custom interval data with a Site."""

    # TODO - should it be custom or custom...???

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

    # TODO - should raise error with the not_interval_data="hello" - an custom kwarg we cannot process

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
            # here we use the custom / custom interval data
            "interval_data": "network_charge",
        },
    ]
    sim = site.optimize(objective)
    assert "site-network_charge" in sim.results.columns

    # below we check that the custom interval data is repeated
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

    # TODO - check we fail if we try to use custom interval data that isn't passed into the site init


@pytest.mark.parametrize("asset_name", asset_names)
def test_get_custom_interval_data_assets(asset_name: str) -> None:
    """Test that we can pass in and use custom interval data with all the assets."""
    ds = generate_random_ev_input_data(48, n_chargers=3, charge_length=3, seed=None)

    ds["network_charge"] = np.zeros_like(ds["electricity_prices"])

    # TODO - should just return a dict
    assets = get_assets(ds, asset_name)
    assert len(assets) == 1
    asset = assets[0]
    assert hasattr(asset.site.cfg.interval_data, "network_charge")

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
            "variable": "export_power_mwh",
            "interval_data": "network_charge",
            "coefficient": -1000,
        },
    ]

    objective.extend(
        [
            {
                "asset_type": "spill",
                "variable": variable,
                "coefficient": defaults.spill_objective_penalty,
            }
            for variable in [
                "electric_generation_mwh",
                "high_temperature_generation_mwh",
                "electric_load_mwh",
                "electric_charge_mwh",
                "electric_discharge_mwh",
            ]
        ]
    )
    objective.extend(
        [
            {
                "asset_type": "spill_evs",
                "variable": variable,
                "coefficient": defaults.spill_objective_penalty,
            }
            for variable in [
                "electric_generation_mwh",
                "high_temperature_generation_mwh",
                "electric_load_mwh",
                "electric_charge_mwh",
                "electric_discharge_mwh",
            ]
        ],
    )
    sim = asset.optimize(objective)
    assert "site-network_charge" in sim.results.columns

    # TODO - check the export power - should be at site limit...
    # but its a bit trikcy with the different assets...
