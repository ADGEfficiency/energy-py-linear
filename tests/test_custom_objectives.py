"""Tests the implementation of custom objective functions."""


import numpy as np
import pytest

import energypylinear as epl
from energypylinear.data_generation import generate_random_ev_input_data
from energypylinear.defaults import defaults


@pytest.mark.parametrize("asset", ["battery", "evs", "chp"])
@pytest.mark.parametrize("objective", ["price", "carbon"])
def test_hardcoded(asset: str, objective: str) -> None:
    """Tests that the hardcoded objective function definitions are the same as the custom version."""
    ds = generate_random_ev_input_data(48, n_chargers=3, charge_length=3, seed=None)

    assets: list
    if asset == "battery":
        assets = [epl.Battery(power_mw=2, capacity_mwh=4, efficiency_pct=0.9)]
    elif asset == "evs":
        assets = [epl.EVs(**ds, charger_turndown=0.0, charge_event_efficiency=1.0)]
    elif asset == "chp":
        assets = [
            epl.CHP(
                electric_power_max_mw=100,
                electric_power_min_mw=50,
                electric_efficiency_pct=0.2,
                high_temperature_efficiency_pct=0.2,
                low_temperature_efficiency_pct=0.2,
            )
        ]
    else:
        assets = []
    assets.extend([epl.Boiler(high_temperature_generation_max_mw=200), epl.Valve()])

    site = epl.Site(
        assets=assets,
        electricity_prices=ds["electricity_prices"],
        electricity_carbon_intensities=ds["electricity_carbon_intensities"],
        gas_prices=30,
        electric_load_mwh=30,
        high_temperature_load_mwh=100,
        low_temperature_load_mwh=100,
        low_temperature_generation_mwh=0,
    )
    hardcoded = site.optimize(verbose=True, objective=objective)

    if objective == "price":
        terms = [
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
        ]
    elif objective == "carbon":
        terms = [
            {
                "asset_type": "site",
                "variable": "import_power_mwh",
                "interval_data": "electricity_carbon_intensities",
            },
            {
                "asset_type": "site",
                "variable": "export_power_mwh",
                "interval_data": "electricity_carbon_intensities",
                "coefficient": -1,
            },
            {
                "asset_type": "*",
                "variable": "gas_consumption_mwh",
                "coefficient": defaults.gas_carbon_intensity,
            },
        ]

    custom = site.optimize(
        verbose=True,
        objective={"terms": terms},
    )

    for col in [
        "site-import_power_mwh",
        "site-export_power_mwh",
        "total-gas_consumption_mwh",
    ]:
        np.testing.assert_array_almost_equal(
            hardcoded.results[col].values, custom.results[col].values
        )


@pytest.mark.parametrize("n", range(5))
@pytest.mark.parametrize("objective", ["price", "carbon"])
def test_hardcoded_with_spills(n: int, objective: str) -> None:
    """Tests that the hardcoded objective function definitions are the same as the custom version."""
    ds = generate_random_ev_input_data(
        24, n_chargers=1, n_charge_events=100, charge_length=3, seed=None
    )

    assets: list = [
        epl.Battery(power_mw=2, capacity_mwh=4, efficiency_pct=0.9),
        epl.EVs(**ds, charger_turndown=0.0, charge_event_efficiency=1.0),
        epl.CHP(
            electric_power_max_mw=100,
            electric_power_min_mw=50,
            electric_efficiency_pct=0.2,
            high_temperature_efficiency_pct=0.2,
            low_temperature_efficiency_pct=0.2,
        ),
        epl.Spill(),
    ]

    site = epl.Site(
        assets=assets,
        electricity_prices=ds["electricity_prices"],
        gas_prices=30,
        electric_load_mwh=30,
        high_temperature_load_mwh=100,
        low_temperature_load_mwh=100,
        low_temperature_generation_mwh=0,
    )
    hardcoded = site.optimize(verbose=True, objective=objective)

    if objective == "price":
        terms = [
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
        ]
    elif objective == "carbon":
        terms = [
            {
                "asset_type": "site",
                "variable": "import_power_mwh",
                "interval_data": "electricity_carbon_intensities",
            },
            {
                "asset_type": "site",
                "variable": "export_power_mwh",
                "interval_data": "electricity_carbon_intensities",
                "coefficient": -1,
            },
            {
                "asset_type": "*",
                "variable": "gas_consumption_mwh",
                "coefficient": defaults.gas_carbon_intensity,
            },
        ]

    custom = site.optimize(
        verbose=True,
        objective={
            "terms": [
                *terms,
                *[
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
                ],
                *[
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
            ]
        },
    )

    for col in [
        "site-import_power_mwh",
        "site-export_power_mwh",
        "total-gas_consumption_mwh",
        "total-high_temperature_generation_mwh",
        "total-low_temperature_generation_mwh",
    ]:
        np.testing.assert_array_almost_equal(
            hardcoded.results[col].values, custom.results[col].values
        )


def test_single_asset() -> None:
    """Test that we can apply a custom objective function to a specific asset.

    This test sets up two CHP assets - one with a gas price that is half the other.
    """
    assets = [
        epl.CHP(
            electric_power_max_mw=100, electric_efficiency_pct=0.5, name="chp-eins"
        ),
        epl.CHP(
            electric_power_max_mw=100, electric_efficiency_pct=0.5, name="chp-zwei"
        ),
        epl.Boiler(),
        epl.Valve(),
    ]
    site = epl.Site(
        assets=assets,
        electricity_prices=[250, 250, 250, 250, 250],
        gas_prices=20,
        electric_load_mwh=[0, 50, 75, 100, 300],
        export_limit_mw=50,
    )
    simulation = site.optimize(
        objective={
            "terms": [
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
                    "asset_name": "chp-eins",
                    "variable": "gas_consumption_mwh",
                    "interval_data": "gas_prices",
                },
                {
                    "asset_name": "chp-zwei",
                    "variable": "gas_consumption_mwh",
                    "interval_data": "gas_prices",
                    "coefficient": 0.5,
                },
            ]
        },
    )
    print(
        simulation.results[
            [
                "site-import_power_mwh",
                "site-export_power_mwh",
                "chp-eins-electric_generation_mwh",
                "chp-zwei-electric_generation_mwh",
            ]
        ]
    )
    np.testing.assert_array_almost_equal(
        simulation.results["site-import_power_mwh"], [0.0, 0.0, 0.0, 0.0, 100]
    )
    np.testing.assert_array_almost_equal(
        simulation.results["site-export_power_mwh"], [50.0, 50, 50, 50, 0]
    )
    np.testing.assert_array_almost_equal(
        simulation.results["chp-eins-electric_generation_mwh"], [0.0, 0.0, 25, 50, 100]
    )
    np.testing.assert_array_almost_equal(
        simulation.results["chp-zwei-electric_generation_mwh"],
        [50.0, 100, 100, 100, 100],
    )
