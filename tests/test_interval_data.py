import typing

import energypylinear as epl


def test_interval_data():
    electricity_prices = [20, 40, -100, 50]
    gas_prices = [20, 40, -100, 50]
    carbon_intensities = [0.1, 0.4, 0.3, 0.9]

    id = epl.data.IntervalData(
        electricity_prices=electricity_prices,
        gas_prices=gas_prices,
        electricity_carbon_intensities=carbon_intensities,
    )
    assert id.electricity_prices == electricity_prices
    assert id.gas_prices == gas_prices
    assert id.electricity_carbon_intensities == carbon_intensities

    id = epl.data.IntervalData(
        electricity_prices=electricity_prices,
        gas_prices=40,
        electricity_carbon_intensities=0.05,
    )
    assert id.electricity_prices == electricity_prices
    assert id.gas_prices == [40, 40, 40, 40]
    assert id.electricity_carbon_intensities == [0.05, 0.05, 0.05, 0.05]
    assert id.high_temperature_load_mwh == [0, 0, 0, 0]
