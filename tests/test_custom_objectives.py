"""Tests the implementation of custom objective functions."""


import random

import numpy as np
import pytest

import energypylinear as epl
from energypylinear.data_generation import generate_random_ev_input_data


@pytest.mark.parametrize("asset", ["battery", "evs", "chp"])
def test_hardcoded_price(asset: str) -> None:
    """Tests that the custom objective function definition of the `price` objective is the same as the hardcoded version."""
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
        gas_prices=30,
        electric_load_mwh=30,
        high_temperature_load_mwh=100,
        low_temperature_load_mwh=100,
        low_temperature_generation_mwh=0,
    )
    hardcoded = site.optimize(verbose=True, objective="price")
    custom = site.optimize(
        verbose=True,
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
                    "asset_type": "*",
                    "variable": "gas_consumption_mwh",
                    "interval_data": "gas_prices",
                },
            ]
        },
    )

    for col in [
        "site-import_power_mwh",
        "site-export_power_mwh",
        "total-gas_consumption_mwh",
    ]:
        np.testing.assert_array_almost_equal(
            hardcoded.results[col].values, custom.results[col].values
        )


@pytest.mark.parametrize("seed", [random.randint(0, 1000) for _ in range(5)])
def test_hardcoded_price_with_spills(seed: int) -> None:
    """Tests that the custom objective function definition of the `price` objective is the same as the hardcoded version."""
    ds = generate_random_ev_input_data(
        24, n_chargers=1, n_charge_events=100, charge_length=3, seed=seed
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
    hardcoded = site.optimize(verbose=True, objective="price")
    from energypylinear.defaults import defaults

    custom = site.optimize(
        verbose=True,
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
                    "asset_type": "*",
                    "variable": "gas_consumption_mwh",
                    "interval_data": "gas_prices",
                },
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
