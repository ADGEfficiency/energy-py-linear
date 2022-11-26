from energypylinear.battery import Battery
from energypylinear.chp import Generator

# def test_asset_api_battery():
#     asset = Battery(power_mw=2, capacity_mwh=4, efficiency=0.95)
#     results = asset.optimize(prices=[10, 50, 10, 50, 10], freq_mins=30)


"""
More tests
- test efficiencies
- test min / maxes?
- import / export limits
"""


def test_asset_api_gas_turbine():
    asset = Generator(
        electric_power_max_mw=100,
        electric_power_min_mw=50,
        electric_efficiency_pct=0.4,
        high_temperature_efficiency_pct=0.6,
    )
    """
    test
    - high elecricity prices with load below, in range and higher than chp
    - low electricity price
    - test the spill warnings
    - test load temperature load
    """
    results = asset.optimize(
        electricity_prices=[1000, -100, 1000],
        gas_prices=20,
        high_temperature_load_mwh=[20, 20, 1000],
        freq_mins=60,
    )
    """
    First row:
    - high electricity price, low heat demand
    - expect generator to run full load and dump heat
    """
    row = results.iloc[0, :]
    assert row["generator-electric_generation_mwh"] == 100
    assert row["spill-high_temperature_load_mwh"] == (100 / 0.4) * 0.6 - 20

    """
    Second row:
    - low electricity price, low heat demand
    - expect all heat demand met from boiler
    """
    row = results.iloc[1, :]
    assert row["generator-electric_generation_mwh"] == 0
    assert row["boiler-high_temperature_generation_mwh"] == 20

    """
    Third row:
    - high electricity price, high heat demand
    - expect generator to run full load and boiler to pick up slack
    """
    row = results.iloc[2, :]
    assert row["generator-electric_generation_mwh"] == 100
    assert row["boiler-high_temperature_generation_mwh"] == 1000 - (100 / 0.4) * 0.6
