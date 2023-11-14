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
        simulation.results["solar-electric_generation_mwh"].values,
        [25, 50, 50, 50, 50],
    )
    np.testing.assert_array_almost_equal(
        simulation.results["wind-electric_generation_mwh"].values, [0, 25, 50, 50, 50]
    )


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
    cols = ["spill-low_temperature_load_mwh"]
    np.testing.assert_array_almost_equal(
        simulation.results["spill-low_temperature_load_mwh"], [100]
    )
    np.testing.assert_array_almost_equal(
        simulation.results["heat-pump-electric_load_mwh"], [0]
    )


def test_behind_meter_battery_no_arbitrage() -> None:
    """
    Want to test:
        - battery behind the meter,
        - it can use behind the meter generation to charge,
        - can discharge to offset on site demand,
        - cannot import power to charge.

    Behind the meter offset = site load - site import

    Think I need to penalize?

    battery charge - (site load - site import)
    """


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
