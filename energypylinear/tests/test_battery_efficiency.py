import unittest
import pytest

import energypylinear

"""
losses = net * (1 - effy)
gross = loss + net

gross = net * (1 - effy) + net

net = gross / 1.5
"""


@pytest.mark.parametrize(
    'prices, initial_charge, efficiency, expected_dispatch',
    [
        ([10, 10, 10], 0, 0.5, [0.0, 0.0, None]),
        ([20, 10, 10], 1, 0.5, [-0.33333334000000003, 0.0, None]),
        ([10, 50, 10, 50, 10], 0, 0.5, [4.0, -1.3333334, 4.0, -1.3333334, None])
    ]
)
def test_batt_efficiency(prices, initial_charge, efficiency, expected_dispatch):
    power = 4
    capacity = 4

    model = energypylinear.Battery(
        power=power, capacity=capacity, timestep='1hr', efficiency=efficiency
    )

    info = model.optimize(
        prices=prices, initial_charge=initial_charge
    )

    dispatch = [res['Net [MW]'] for res in info]

    unittest.TestCase().assertCountEqual(dispatch, expected_dispatch)
