import pytest
import unittest

import energypylinear


@pytest.mark.parametrize(
    'prices, initial_charge, expected_dispatch',
    [
        ([10, 10, 10], 0, [0, 0, 0]),
        ([20, 10, 10], 1, [-1, 0, 0]),
        ([10, 50, 10, 50, 10], 0, [4, -4, 4, -4, 0])
    ]
)
def test_battery_optimization(prices, initial_charge, expected_dispatch):
    power = 4
    capacity = 6

    model = energypylinear.Battery(
        power=power, capacity=capacity, efficiency=1.0
    )

    info = model.optimize(
        prices=prices, initial_charge=initial_charge, timestep='1hr'
    )

    dispatch = [res['Net [MW]'] for res in info]
    unittest.TestCase().assertCountEqual(dispatch, expected_dispatch)


def test_battery_optimization_against_forecast():
    model = energypylinear.Battery(
        power=4, capacity=6, efficiency=1.0
    )

    info = model.optimize(
        prices=[10, 10, 10], forecasts=[10, 10, 10]
    )

    result = [res['Net [MW]'] for res in info]
    unittest.TestCase().assertCountEqual(result, [0, 0, 0])
