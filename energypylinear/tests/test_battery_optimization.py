import numpy as np
import pytest

import energypylinear

@pytest.mark.parametrize(
    'prices, initial_charge, expected_dispatch',
    [
        ([10, 10, 10], 0, [0, 0, np.nan]),
        ([20, 10, 10], 1, [-1, 0, np.nan]),
        ([10, 50, 10, 50, 10], 0, [4, -4, 4, -4, np.nan])
    ]
)
def test_battery_optimization(prices, initial_charge, expected_dispatch):
    power = 4
    capacity = 4

    model = energypylinear.Battery(
        power=power, capacity=capacity, timestep='1hr'
    )

    info = model.optimize(
        prices=prices, initial_charge=initial_charge
    )

    dispatch = info.loc[:, 'Power [MW]'].values

    np.testing.assert_equal(dispatch, expected_dispatch)

