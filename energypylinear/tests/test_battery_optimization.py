import pytest
import unittest

import energypylinear


@pytest.mark.parametrize(
    'prices, initial_charge, expected_dispatch',
    [
        ([10, 10, 10], 0, [0, 0, 0, None]),
        ([20, 10, 10], 1, [-1, 0, 0, None]),
        ([10, 50, 10, 50, 10], 0, [4, -4, 4, -4, 0, None])
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
