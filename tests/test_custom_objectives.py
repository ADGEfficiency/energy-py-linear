"""Tests the implementation of custom objective functions."""


import energypylinear as epl
from energypylinear.data_generation import generate_random_ev_input_data


def test_custom_objective_price() -> None:
    """Tests that the custom objective function definition of the `price` objective is the same as the hardcoded version."""
    seed = 42
    ds = generate_random_ev_input_data(48, n_chargers=3, charge_length=3, seed=seed)
    assets = [
        epl.Battery(
            power_mw=2,
            capacity_mwh=4,
            efficiency_pct=0.9,
        )
    ]

    site = epl.Site(
        assets=assets,
        electricity_prices=ds["electricity_prices"],
        gas_prices=30,
        electric_load_mwh=30,
        high_temperature_load_mwh=0,
        low_temperature_load_mwh=0,
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
            ]
        },
    )
