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
    'prices, initial_charge, efficiency, expected_dispatch',
    [
        ([10, 10, 10], 0, 0.5, [0, 0, np.nan]),
        ([20, 10, 10], 1, 0.5, [1/1.5-1, 0, np.nan]),
        ([10, 50, 10, 50, 10], 0, 0.5, [4, 4/1.5-4, 4, 4/1.5-4, np.nan])
    ]
)
def test_battery_efficiency(prices, initial_charge, efficiency, expected_dispatch):
    power = 4
    capacity = 4

    model = energypylinear.Battery(
        power=power, capacity=capacity, timestep='1hr', efficiency=efficiency
    )

    info = model.optimize(
        prices=prices, initial_charge=initial_charge
    )

    dispatch = info.loc[:, 'Net [MW]'].values

    np.testing.assert_array_almost_equal(dispatch, expected_dispatch)

