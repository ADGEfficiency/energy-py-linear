"""Test CHP asset."""
import numpy as np

import energypylinear as epl
from energypylinear.defaults import defaults


def test_chp_gas_turbine_price() -> None:
    """Test gas turbine optimization for price."""
    asset = epl.CHP(
        electric_power_max_mw=100,
        electric_power_min_mw=50,
        electric_efficiency_pct=0.3,
        high_temperature_efficiency_pct=0.5,
        electricity_prices=[1000, -100, 1000],
        gas_prices=20,
        high_temperature_load_mwh=[20, 20, 1000],
        freq_mins=60,
        name="chp",
    )
    simulation = asset.optimize()
    """
    - high electricity price, low heat demand
    - expect chp to run full load and dump heat to low temperature
    """
    row = simulation.results.iloc[0, :]
    assert row["chp-electric_generation_mwh"] == 100

    np.testing.assert_almost_equal(
        row["spill-low_temperature_load_mwh"],
        (100 / 0.3) * 0.5 - 20,
        decimal=defaults.decimal_tolerance,
    )
    """
    - low electricity price, low heat demand
    - expect all heat demand met from boiler
    """
    row = simulation.results.iloc[1, :]
    assert row["chp-electric_generation_mwh"] == 0
    assert row["boiler-high_temperature_generation_mwh"] == 20

    """
    - high electricity price, high heat demand
    - expect chp to run full load and boiler to pick up slack
    """
    row = simulation.results.iloc[2, :]
    assert row["chp-electric_generation_mwh"] == 100
    np.testing.assert_almost_equal(
        row["boiler-high_temperature_generation_mwh"],
        1000 - (100 / 0.3) * 0.5,
        decimal=defaults.decimal_tolerance,
    )

    #  TODO - should be done elsewhere - just for coverage
    asset.site.optimizer.constraints()


def test_chp_gas_turbine_carbon() -> None:
    """Test gas turbine optimization for carbon."""
    asset = epl.CHP(
        electric_power_max_mw=100,
        electric_power_min_mw=50,
        electric_efficiency_pct=0.3,
        high_temperature_efficiency_pct=0.5,
        electricity_prices=[1000, -100, 1000],
        electricity_carbon_intensities=[1.5, 0.1, 1.5],
        gas_prices=20,
        high_temperature_load_mwh=[20, 20, 1000],
        freq_mins=60,
    )
    simulation = asset.optimize(
        objective="carbon",
    )
    """
    - high carbon intensity, low heat demand
    - expect chp to run full load and dump heat to low temperature
    """
    row = simulation.results.iloc[0, :]
    assert row["chp-electric_generation_mwh"] == 100
    np.testing.assert_almost_equal(
        row["spill-low_temperature_load_mwh"],
        (100 / 0.3) * 0.5 - 20,
        decimal=defaults.decimal_tolerance,
    )
    """
    - low carbon intensity, low heat demand
    - expect all heat demand met from boiler
    """
    row = simulation.results.iloc[1, :]
    assert row["chp-electric_generation_mwh"] == 0
    assert row["boiler-high_temperature_generation_mwh"] == 20

    """
    - high carbon intensity, high heat demand
    - expect chp to run full load and boiler to pick up slack
    """
    row = simulation.results.iloc[2, :]
    assert row["chp-electric_generation_mwh"] == 100
    np.testing.assert_almost_equal(
        row["boiler-high_temperature_generation_mwh"],
        1000 - (100 / 0.3) * 0.5,
        decimal=defaults.decimal_tolerance,
    )


def test_chp_gas_engine_price() -> None:
    """Test gas engine optimization for price."""
    asset = epl.CHP(
        electric_power_max_mw=100,
        electric_power_min_mw=10,
        electric_efficiency_pct=0.4,
        high_temperature_efficiency_pct=0.2,
        low_temperature_efficiency_pct=0.2,
        electricity_prices=[
            1000.0,
        ],
        gas_prices=20.0,
        high_temperature_load_mwh=[
            20.0,
        ],
        low_temperature_load_mwh=[
            20.0,
        ],
        freq_mins=60,
    )
    simulation = asset.optimize()
    """
    - high electricity price, low heat demand
    - expect chp to run full load and dump heat
    """
    row = simulation.results.iloc[0, :]
    assert row["chp-electric_generation_mwh"] == 100
    np.testing.assert_almost_equal(
        row["spill-low_temperature_load_mwh"],
        (100 / 0.4) * 0.4 - 40,
        decimal=defaults.decimal_tolerance,
    )


def test_chp_gas_engine_carbon() -> None:
    """Test gas engine optimization for carbon."""
    name = "chp"
    asset = epl.CHP(
        electric_power_max_mw=100,
        electric_power_min_mw=10,
        electric_efficiency_pct=0.4,
        high_temperature_efficiency_pct=0.2,
        low_temperature_efficiency_pct=0.2,
        electricity_prices=[0, 0],
        electricity_carbon_intensities=[1.5, 0.1],
        gas_prices=20.0,
        high_temperature_load_mwh=[20.0, 20],
        low_temperature_load_mwh=[20.0, 20],
        freq_mins=60,
    )
    simulation = asset.optimize(
        objective="carbon",
    )
    """
    - high carbon intensity, low heat demand
    - expect chp to run full load and dump heat
    """
    row = simulation.results.iloc[0, :]
    assert row["chp-electric_generation_mwh"] == 100
    np.testing.assert_almost_equal(
        row["spill-low_temperature_load_mwh"],
        (100 / 0.4) * 0.4 - 40,
        decimal=defaults.decimal_tolerance,
    )
    """
    - low carbon intensity, low heat demand
    - expect chp to not run at all
    """
    row = simulation.results.iloc[1, :]
    assert row["chp-electric_generation_mwh"] == 0
    np.testing.assert_almost_equal(
        row["spill-low_temperature_load_mwh"],
        0,
        decimal=defaults.decimal_tolerance,
    )
