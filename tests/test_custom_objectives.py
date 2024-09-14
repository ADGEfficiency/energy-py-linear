"""Test that we can use custom objective functions."""
import numpy as np
import pytest

import energypylinear as epl
from energypylinear.data_generation import generate_random_ev_input_data
from energypylinear.defaults import defaults
from energypylinear.objectives import OneTerm
from tests.common import asset_names, get_assets


def get_objective_terms() -> dict[str, list]:
    """Helper function to get the price and carbon objectives."""
    price = [
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
    carbon = [
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

    for obj in [price, carbon]:
        obj.extend(
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
        obj.extend(
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
    return {"price": price, "carbon": carbon}


@pytest.mark.parametrize("asset", asset_names)
@pytest.mark.parametrize("objective", ["price", "carbon"])
def test_hardcoded_asset_api(asset: str, objective: str) -> None:
    """Tests that the hardcoded objective function definitions are the same as the custom version when using asset.optimize()."""
    ds = generate_random_ev_input_data(48, n_chargers=3, charge_length=3, seed=None)
    assets: list = get_assets(ds, asset)
    terms = get_objective_terms()[objective]
    custom = assets[0].optimize(objective={"terms": terms})
    hardcoded = assets[0].optimize(objective=objective)

    for col in [
        "site-import_power_mwh",
        "site-export_power_mwh",
        "total-gas_consumption_mwh",
    ]:
        np.testing.assert_array_almost_equal(
            hardcoded.results[col], custom.results[col]
        )

    assert epl.get_accounts(custom.results) == epl.get_accounts(hardcoded.results)


@pytest.mark.parametrize("asset", asset_names)
@pytest.mark.parametrize("objective", ["price", "carbon"])
def test_hardcoded_site_api(asset: str, objective: str) -> None:
    """Tests that the hardcoded objective function definitions are the same as the custom version when using site.optimize()."""
    ds = generate_random_ev_input_data(48, n_chargers=3, charge_length=3, seed=None)
    assets: list = get_assets(ds, asset)
    assets.extend([epl.Boiler(high_temperature_generation_max_mw=200), epl.Valve()])
    terms = get_objective_terms()[objective]
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
    hardcoded = site.optimize(verbose=3, objective=objective)
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
            hardcoded.results[col], custom.results[col]
        )
    assert epl.get_accounts(custom.results) == epl.get_accounts(hardcoded.results)


@pytest.mark.parametrize("n", range(5))
@pytest.mark.parametrize("objective", ["price", "carbon"])
def test_hardcoded_many_assets(n: int, objective: str) -> None:
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
    terms = get_objective_terms()[objective]
    custom = site.optimize(
        verbose=True,
        objective={"terms": terms},
    )

    for col in [
        "site-import_power_mwh",
        "site-export_power_mwh",
        "total-gas_consumption_mwh",
        "total-high_temperature_generation_mwh",
        "total-low_temperature_generation_mwh",
    ]:
        np.testing.assert_array_almost_equal(
            hardcoded.results[col], custom.results[col]
        )
    assert epl.get_accounts(custom.results) == epl.get_accounts(hardcoded.results)


def test_two_chp_different_gas_prices() -> None:
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
    assert epl.get_accounts(simulation.results).custom.cost == 0
    assert epl.get_accounts(simulation.results).custom.emissions == 0


def test_renewable_certificate() -> None:
    """Test that we can apply a custom objective function to a specific asset.

    This test sets two renewable generators, where one renewable generator output receives an additional income per MWh.
    """
    assets = [
        epl.RenewableGenerator(
            electric_generation_mwh=50,
            name="wind",
            electric_generation_lower_bound_pct=0.0,
        ),
        epl.RenewableGenerator(
            electric_generation_mwh=50,
            name="solar",
            electric_generation_lower_bound_pct=0.0,
        ),
    ]
    site = epl.Site(
        assets=assets,
        electricity_prices=[250, 250, 250, 250, 250],
        export_limit_mw=25,
        electric_load_mwh=[0, 50, 75, 100, 300],
    )
    simulation = site.optimize(
        objective=epl.CustomObjectiveFunction(
            terms=[
                epl.Term(
                    asset_type="site",
                    variable="import_power_mwh",
                    interval_data="electricity_prices",
                ),
                epl.Term(
                    asset_type="site",
                    variable="export_power_mwh",
                    interval_data="electricity_prices",
                    coefficient=-1,
                ),
                epl.Term(
                    asset_name="solar",
                    variable="electric_generation_mwh",
                    coefficient=-25,
                ),
            ]
        )
    )
    np.testing.assert_array_almost_equal(
        simulation.results["solar-electric_generation_mwh"],
        [25, 50, 50, 50, 50],
    )
    np.testing.assert_array_almost_equal(
        simulation.results["wind-electric_generation_mwh"], [0, 25, 50, 50, 50]
    )

    custom_terms: list[OneTerm] = [
        epl.Term(
            asset_name="solar", variable="electric_generation_mwh", coefficient=-25
        )
    ]
    assert epl.get_accounts(
        simulation.results, custom_terms=custom_terms
    ).custom.cost == -25 * np.sum([25, 50, 50, 50, 50])


def test_heat_dump_cost() -> None:
    """Test that we can apply a custom objective function to a specific asset.

    This test sets up a heat pump system where dumping heat has a cost.
    """
    # first setup a site with a penalty to dumping heat, but a high electricity price
    # heat pump will operate even though the electricity price is high
    assets = [epl.HeatPump(electric_power_mw=10, cop=4), epl.Spill()]
    site = epl.Site(
        assets=assets,
        electricity_prices=[200],
        export_limit_mw=25,
        high_temperature_load_mwh=100,
        low_temperature_generation_mwh=100,
    )
    simulation = site.optimize(
        objective=epl.CustomObjectiveFunction(
            terms=[
                epl.Term(
                    asset_type="site",
                    variable="import_power_mwh",
                    interval_data="electricity_prices",
                ),
                epl.Term(
                    asset_type="site",
                    variable="export_power_mwh",
                    interval_data="electricity_prices",
                    coefficient=-1,
                ),
                epl.Term(
                    asset_name="spill",
                    variable="low_temperature_load_mwh",
                    coefficient=100,
                ),
                *[
                    epl.Term(
                        asset_type="spill",
                        variable=variable,
                        coefficient=defaults.spill_objective_penalty,
                    )
                    for variable in [
                        "electric_generation_mwh",
                        "high_temperature_generation_mwh",
                        "electric_load_mwh",
                        "electric_charge_mwh",
                        "electric_discharge_mwh",
                    ]
                ],
            ]
        )
    )
    np.testing.assert_array_almost_equal(
        simulation.results["spill-low_temperature_load_mwh"], [70]
    )
    np.testing.assert_array_almost_equal(
        simulation.results["heat-pump-electric_load_mwh"], [10]
    )
    # TODO - not testing accounts here

    # now setup a site with no penalty to dumping heat, but a high electricity price
    # heat pump will not operate
    assets = [epl.HeatPump(electric_power_mw=10, cop=4), epl.Spill()]
    site = epl.Site(
        assets=assets,
        electricity_prices=[200],
        export_limit_mw=25,
        high_temperature_load_mwh=100,
        low_temperature_generation_mwh=100,
    )
    simulation = site.optimize(
        objective=epl.CustomObjectiveFunction(
            terms=[
                epl.Term(
                    asset_type="site",
                    variable="import_power_mwh",
                    interval_data="electricity_prices",
                ),
                epl.Term(
                    asset_type="site",
                    variable="export_power_mwh",
                    interval_data="electricity_prices",
                    coefficient=-1,
                ),
            ]
        )
    )
    np.testing.assert_array_almost_equal(
        simulation.results["spill-low_temperature_load_mwh"], [100]
    )
    np.testing.assert_array_almost_equal(
        simulation.results["heat-pump-electric_load_mwh"], [0]
    )
    # TODO - not testing accounts here


def test_synthetic_ppa() -> None:
    """Test a synthetic PPA where we swap renewable output at spot for a fixed price."""
    ds = generate_random_ev_input_data(
        48, n_chargers=3, charge_length=3, seed=None, prices_mu=0, prices_std=50
    )
    wind_mwh = np.random.uniform(0, 100, len(ds["electricity_prices"]))
    assets: list[epl.Asset] = [
        epl.RenewableGenerator(
            electric_generation_mwh=wind_mwh,
            name="wind",
            electric_generation_lower_bound_pct=0.0,
        ),
        epl.Battery(power_mw=20, capacity_mwh=20),
    ]

    terms = get_objective_terms()["price"]
    site = epl.Site(
        assets=assets,
        electricity_prices=ds["electricity_prices"],
        electric_load_mwh=30,
    )
    no_ppa = site.optimize(
        verbose=True,
        objective={"terms": terms},
    )
    ppa_terms = get_objective_terms()["price"]
    ppa_terms.extend(
        [
            {
                "asset_name": "wind",
                "variable": "electric_generation_mwh",
                "interval_data": "electricity_prices",
                "coefficient": 1,
            },
            {
                "asset_name": "wind",
                "variable": "electric_generation_mwh",
                "coefficient": -70,
            },
        ]
    )
    ppa = site.optimize(
        verbose=True,
        objective={"terms": ppa_terms},
    )
    assert (
        no_ppa.results["wind-electric_generation_mwh"].sum()
        < ppa.results["wind-electric_generation_mwh"].sum()
    )
    np.testing.assert_array_almost_equal(
        ppa.results["wind-electric_generation_mwh"], wind_mwh
    )
    np.testing.assert_allclose(
        epl.get_accounts(ppa.results, custom_terms=ppa_terms[-2:]).custom.cost,
        sum(wind_mwh * (np.array(ds["electricity_prices"]) - 70)),
    )


def test_custom_errors() -> None:
    """Test that we raise the correct value error when we try use a hardcoded objective that doesn't exist."""

    site = epl.Site(
        assets=[epl.Battery()],
        electricity_prices=[250, 250, 250, 250, 250],
        gas_prices=20,
        electric_load_mwh=[0, 50, 75, 100, 300],
        export_limit_mw=50,
    )
    with pytest.raises(ValueError):
        site.optimize(objective="not-a-valid-objective")


def test_custom_terms_account() -> None:
    """Tests to round of test coverage of accounts."""
    ds = generate_random_ev_input_data(48, n_chargers=3, charge_length=3, seed=None)

    assets = [
        epl.Battery(
            power_mw=2, capacity_mwh=4, efficiency_pct=0.9, name="small-battery"
        ),
        epl.Battery(
            power_mw=20, capacity_mwh=40, efficiency_pct=0.9, name="large-battery"
        ),
        epl.RenewableGenerator(
            electric_generation_mwh=np.random.uniform(
                0, 100, size=len(ds["electricity_prices"])
            )
        ),
    ]
    terms = get_objective_terms()["price"]
    terms.extend(
        [
            # targets the find all assets where asset type == "*"
            {
                "asset_type": "*",
                "variable": "electric_generation_mwh",
                "coefficient": 1,
            },
            # targets the type_id inference in `find_asset_type_with_variables`
            {
                "asset_type": "renewable-generator",
                "variable": "electric_generation_mwh",
                "coefficient": 1,
            },
            # targets the `add_many_variables` with an asset by type
            {
                "function": "max_many_variables",
                "variables": {
                    "asset_type": "renewable-generator",
                    "variable": "electric_generation_mwh",
                },
                "constant": 10,
                "coefficient": 10,
                "M": 1000,
            },
            # targets the `add_many_variables` with an asset by name
            {
                "function": "max_many_variables",
                "variables": {
                    "asset_name": "large-battery",
                    "variable": "electric_charge_mwh",
                },
                "constant": 10,
                "coefficient": 10,
                "M": 1000,
            },
        ]
    )

    site = epl.Site(
        assets=assets,
        electricity_prices=ds["electricity_prices"],
        electricity_carbon_intensities=ds["electricity_carbon_intensities"],
    )
    simulation = site.optimize(
        verbose=True,
        objective={"terms": terms},
    )

    epl.get_accounts(simulation.results, custom_terms=terms[-3:])
