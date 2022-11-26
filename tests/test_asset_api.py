from energypylinear.battery import Battery
from energypylinear.chp import Generator

# def test_asset_api_battery():
#     asset = Battery(power_mw=2, capacity_mwh=4, efficiency=0.95)
#     results = asset.optimize(prices=[10, 50, 10, 50, 10], freq_mins=30)


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
        electricity_prices=[1000, -100, 10, 50, 10],
        gas_prices=20,
        high_temperature_load_mwh=[20, 75, 100, 20, 100],
        freq_mins=60,
    )

    print(results)
