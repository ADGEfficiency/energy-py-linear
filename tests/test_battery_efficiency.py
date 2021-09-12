import numpy as np
import pytest

import energypylinear

"""
losses = net * (1 - effy)
gross = loss + net

gross = net * (1 - effy) + net

net = gross / 1.5
"""


@pytest.mark.parametrize(
    "prices, initial_charge, efficiency, expected_dispatch",
    [
        ([10, 10, 10], 0, 0.5, [0.0, 0.0, 0.0]),
        ([20, 10, 10], 1, 0.5, [-0.5, 0.0, 0.0]),
        ([10, 50, 10, 50, 10], 0, 0.5, [1.0, -0.5, 1.0, -0.5, 0.0]),
    ],
)
def test_batt_efficiency(prices, initial_charge, efficiency, expected_dispatch):
    power = 1.0
    capacity = 1.0
    freq = "60T"

    mdl = energypylinear.Battery(power=power, capacity=capacity, efficiency=efficiency)
    info = mdl.optimize(prices=prices, initial_charge=initial_charge, freq=freq)
    dispatch = [res["Net [MW]"] for res in info]
    np.testing.assert_array_equal(dispatch, expected_dispatch)
