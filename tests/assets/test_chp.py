import numpy as np

import energypylinear as epl
from energypylinear.defaults import defaults


def test_chp_gas_turbine_price() -> None:
    asset = epl.chp.Generator(
        electric_power_max_mw=100,
        electric_power_min_mw=50,
        electric_efficiency_pct=0.3,
        high_temperature_efficiency_pct=0.5,
    )
    results = asset.optimize(
        electricity_prices=[1000, -100, 1000],
        gas_prices=20,
        high_temperature_load_mwh=[20, 20, 1000],
        freq_mins=60,
    )
    results = results.simulation
    """
    - high electricity price, low heat demand
    - expect generator to run full load and dump heat to low temperature
    """
    row = results.iloc[0, :]
    assert row["generator-electric_generation_mwh"] == 100
    np.testing.assert_almost_equal(
        row["spill-default-low_temperature_load_mwh"],
        (100 / 0.3) * 0.5 - 20,
        decimal=defaults.decimal_tolerance,
    )
    """
    - low electricity price, low heat demand
    - expect all heat demand met from boiler
    """
    row = results.iloc[1, :]
    assert row["generator-electric_generation_mwh"] == 0
    assert row["boiler-high_temperature_generation_mwh"] == 20

    """
    - high electricity price, high heat demand
    - expect generator to run full load and boiler to pick up slack
    """
    row = results.iloc[2, :]
    assert row["generator-electric_generation_mwh"] == 100
    np.testing.assert_almost_equal(
        row["boiler-high_temperature_generation_mwh"],
        1000 - (100 / 0.3) * 0.5,
        decimal=defaults.decimal_tolerance,
    )


def test_chp_gas_turbine_carbon() -> None:
    asset = epl.chp.Generator(
        electric_power_max_mw=100,
        electric_power_min_mw=50,
        electric_efficiency_pct=0.3,
        high_temperature_efficiency_pct=0.5,
    )
    results = asset.optimize(
        electricity_prices=[1000, -100, 1000],
        electricity_carbon_intensities=[1.5, 0.1, 1.5],
        gas_prices=20,
        high_temperature_load_mwh=[20, 20, 1000],
        freq_mins=60,
        objective="carbon",
    )
    results = results.simulation
    """
    - high carbon intensity, low heat demand
    - expect generator to run full load and dump heat to low temperature
    """
    row = results.iloc[0, :]
    assert row["generator-electric_generation_mwh"] == 100
    np.testing.assert_almost_equal(
        row["spill-default-low_temperature_load_mwh"],
        (100 / 0.3) * 0.5 - 20,
        decimal=defaults.decimal_tolerance,
    )
    """
    - low carbon intensity, low heat demand
    - expect all heat demand met from boiler
    """
    row = results.iloc[1, :]
    assert row["generator-electric_generation_mwh"] == 0
    assert row["boiler-high_temperature_generation_mwh"] == 20

    """
    - high carbon intensity, high heat demand
    - expect generator to run full load and boiler to pick up slack
    """
    row = results.iloc[2, :]
    assert row["generator-electric_generation_mwh"] == 100
    np.testing.assert_almost_equal(
        row["boiler-high_temperature_generation_mwh"],
        1000 - (100 / 0.3) * 0.5,
        decimal=defaults.decimal_tolerance,
    )


def test_chp_gas_engine_price() -> None:
    asset = epl.chp.Generator(
        electric_power_max_mw=100,
        electric_power_min_mw=10,
        electric_efficiency_pct=0.4,
        high_temperature_efficiency_pct=0.2,
        low_temperature_efficiency_pct=0.2,
    )
    results = asset.optimize(
        electricity_prices=[
            1000,
        ],
        gas_prices=20,
        high_temperature_load_mwh=[
            20,
        ],
        low_temperature_load_mwh=[
            20,
        ],
        freq_mins=60,
    )
    results = results.simulation
    """
    - high electricity price, low heat demand
    - expect generator to run full load and dump heat
    """
    row = results.iloc[0, :]
    assert row["generator-electric_generation_mwh"] == 100
    np.testing.assert_almost_equal(
        row["spill-default-low_temperature_load_mwh"],
        (100 / 0.4) * 0.4 - 40,
        decimal=defaults.decimal_tolerance,
    )


def test_chp_gas_engine_carbon() -> None:
    asset = epl.chp.Generator(
        electric_power_max_mw=100,
        electric_power_min_mw=10,
        electric_efficiency_pct=0.4,
        high_temperature_efficiency_pct=0.2,
        low_temperature_efficiency_pct=0.2,
    )
    results = asset.optimize(
        electricity_prices=[0, 0],
        electricity_carbon_intensities=[1.5, 0.1],
        gas_prices=20,
        high_temperature_load_mwh=[20, 20],
        low_temperature_load_mwh=[20, 20],
        freq_mins=60,
        objective="carbon",
    )
    results = results.simulation
    """
    - high carbon intensity, low heat demand
    - expect generator to run full load and dump heat
    """
    row = results.iloc[0, :]
    assert row["generator-electric_generation_mwh"] == 100
    np.testing.assert_almost_equal(
        row["spill-default-low_temperature_load_mwh"],
        (100 / 0.4) * 0.4 - 40,
        decimal=defaults.decimal_tolerance,
    )
    """
    - low carbon intensity, low heat demand
    - expect generator to not run at all
    """
    row = results.iloc[1, :]
    assert row["generator-electric_generation_mwh"] == 0
    np.testing.assert_almost_equal(
        row["spill-default-low_temperature_load_mwh"],
        0,
        decimal=defaults.decimal_tolerance,
    )
