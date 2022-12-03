"""
More tests
- test efficiencies
- test min / maxes of generator output
- import / export limits
- test the spill warnings
- test low temperature load - will require the valve

TODO

test errors
- test negative power_mw, efficiency > 1.0 etc
- test we throw error when initial_charge_mwh or final_charge_mwh larger than capacity
- test final_charge_mwh = None
- test freq mins
"""


import numpy as np
import pytest

import energypylinear as epl
from energypylinear.battery import Battery
from energypylinear.chp import Generator
from energypylinear.defaults import defaults


def test_asset_api_gas_turbine() -> None:
    asset = Generator(
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
    """
    - high electricity price, low heat demand
    - expect generator to run full load and dump heat to low temperature
    """
    row = results.iloc[0, :]
    assert row["generator-alpha-electric_generation_mwh"] == 100
    np.testing.assert_almost_equal(
        row["spill-alpha-low_temperature_load_mwh"],
        (100 / 0.3) * 0.5 - 20,
        decimal=defaults.decimal_tolerance,
    )

    """
    - low electricity price, low heat demand
    - expect all heat demand met from boiler
    """
    row = results.iloc[1, :]
    assert row["generator-alpha-electric_generation_mwh"] == 0
    assert row["boiler-alpha-high_temperature_generation_mwh"] == 20

    """
    - high electricity price, high heat demand
    - expect generator to run full load and boiler to pick up slack
    """
    row = results.iloc[2, :]
    assert row["generator-alpha-electric_generation_mwh"] == 100
    np.testing.assert_almost_equal(
        row["boiler-alpha-high_temperature_generation_mwh"],
        1000 - (100 / 0.3) * 0.5,
        decimal=defaults.decimal_tolerance,
    )


def test_asset_api_gas_engine() -> None:
    #  add a gas engine
    asset = Generator(
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
    # breakpoint()  # fmt: skip
    """
    - high electricity price, low heat demand
    - expect generator to run full load and dump heat
    """
    row = results.iloc[0, :]
    assert row["generator-alpha-electric_generation_mwh"] == 100
    np.testing.assert_almost_equal(
        row["spill-alpha-low_temperature_load_mwh"],
        (100 / 0.4) * 0.4 - 40,
        decimal=defaults.decimal_tolerance,
    )


def test_asset_api_battery() -> None:
    power_mw = 2
    capacity_mwh = 4
    efficiency = 0.95
    initial_charge_mwh = 0.5
    final_charge_mwh = 1.5

    freq_mins = 30
    asset = Battery(power_mw=power_mw, capacity_mwh=capacity_mwh, efficiency=efficiency)
    results = asset.optimize(
        electricity_prices=[10, 50, 10, 50, 10],
        freq_mins=freq_mins,
        initial_charge_mwh=initial_charge_mwh,
        final_charge_mwh=final_charge_mwh,
    )

    freq = epl.freq.Freq(freq_mins)

    #  check we don't exceed the battery rating
    assert all(results["battery-alpha-charge_mwh"] <= freq.mw_to_mwh(power_mw))
    assert all(results["battery-alpha-discharge_mwh"] <= freq.mw_to_mwh(power_mw))

    #  check charge & discharge are always positive
    assert all(results["battery-alpha-charge_mwh"] >= 0)
    assert all(results["battery-alpha-discharge_mwh"] >= 0)

    #  check we don't exceed battery capacity
    name = "battery-alpha"
    for var in ["initial_charge_mwh", "final_charge_mwh"]:
        assert all(results[f"{name}-{var}"] <= capacity_mwh)
        assert all(results[f"{name}-{var}"] >= 0)

    #  check we set initial and final charge correctly
    assert results[f"{name}-initial_charge_mwh"].iloc[0] == initial_charge_mwh
    assert results[f"{name}-final_charge_mwh"].iloc[-1] == final_charge_mwh

    #  check losses are a percentage of our charge
    mask = results[f"{name}-charge_mwh"] > 0
    subset = results[mask]
    np.testing.assert_almost_equal(
        subset[f"{name}-losses_mwh"].values,
        (1 - efficiency) * subset[f"{name}-charge_mwh"].values,
    )

    #  check losses are always zero when we discharge
    mask = results[f"{name}-discharge_mwh"] > 0
    subset = results[mask]
    assert all(subset[f"{name}-losses_mwh"] == 0)


@pytest.mark.parametrize(
    "electricity_prices, initial_charge_mwh, expected_dispatch",
    [
        ([10, 10, 10], 0, [0, 0, 0]),
        ([20, 10, 10], 6, [-4, -2, 0]),
        ([10, 50, 10, 5000, 10], 0, [4, -4, 4, -4, 0]),
    ],
)
def test_asset_api_battery_optimization(
    electricity_prices, initial_charge_mwh, expected_dispatch
) -> None:
    power_mw = 4
    capacity_mwh = 6
    efficiency = 1.0
    freq_mins = 60
    asset = Battery(power_mw=power_mw, capacity_mwh=capacity_mwh, efficiency=efficiency)
    results = asset.optimize(
        electricity_prices=electricity_prices,
        freq_mins=freq_mins,
        initial_charge_mwh=initial_charge_mwh,
        final_charge_mwh=0,
    )
    charge = results["battery-alpha-charge_mwh"].values
    discharge = results["battery-alpha-discharge_mwh"].values
    dispatch = charge - discharge
    np.testing.assert_almost_equal(dispatch, expected_dispatch)


def test_asset_api_carbon_optimization():
    """
    how to test this?

    run a battery with different carbon intensities in different intervals
    check dispatch changes appropriately
    """
    pass
