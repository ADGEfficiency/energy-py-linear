"""
More tests
- test efficiencies
- test min / maxes of generator output
- import / export limits
- test the spill warnings
- test low temperature load - will require the valve
"""
import numpy as np

import energypylinear as epl
from energypylinear.battery import Battery
from energypylinear.chp import Generator


def test_asset_api_gas_turbine():
    asset = Generator(
        electric_power_max_mw=100,
        electric_power_min_mw=50,
        electric_efficiency_pct=0.4,
        high_temperature_efficiency_pct=0.6,
    )
    results = asset.optimize(
        electricity_prices=[1000, -100, 1000],
        gas_prices=20,
        high_temperature_load_mwh=[20, 20, 1000],
        freq_mins=60,
    )
    """
    - high electricity price, low heat demand
    - expect generator to run full load and dump heat
    """
    row = results.iloc[0, :]
    assert row["generator-alpha-electric_generation_mwh"] == 100
    assert row["spill-alpha-high_temperature_load_mwh"] == (100 / 0.4) * 0.6 - 20

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
    assert (
        row["boiler-alpha-high_temperature_generation_mwh"] == 1000 - (100 / 0.4) * 0.6
    )


"""
TODO

test errors
- test negative power_mw, efficiency > 1.0 etc
- test we throw error when initial_charge_mwh or final_charge_mwh larger than capacity
- test final_charge_mwh = None
"""


def test_asset_api_battery():
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

    # /Users/adam/energy-py-linear-v1/tests/test_battery.py

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
