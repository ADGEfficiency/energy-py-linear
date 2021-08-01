import numpy as np
import pytest

import energypylinear

from test_battery import map_values


@pytest.mark.parametrize(
    'prices, initial_charge, efficiency, expected_dispatch',
    [
        #  do nothing
        ([10, 10, 10], 0, 0.5, [0.0, 0.0, 0.0]),
        #  discharge in first interval
        ([20, 10, 10], 1, 0.5, [-0.5, 0.0, 0.0]),
        #  charge, discharge, charge, discharge, nothing
        ([10, 50, 10, 50, 10], 0, 0.5, [1.0, -0.5, 1.0, -0.5, 0.0])
    ]
)
def test_batt_efficiency(prices, initial_charge, efficiency, expected_dispatch):
    power = 1.0
    capacity = 1.0
    timestep = '1hr'
    model = energypylinear.Battery(power=power, capacity=capacity, efficiency=efficiency)
    info = model.optimize(prices=prices, initial_charge=initial_charge, timestep=timestep)
    dispatch = [res['Net [MW]'] for res in info]
    np.testing.assert_almost_equal(dispatch, expected_dispatch)


@pytest.mark.parametrize(
    'prices',
    (
        np.random.uniform(-1000, 1000, 256).tolist(),
    )
)
def test_battery_losses(prices):
    efficiency = 0.9
    model = energypylinear.Battery(
        power=2.0, capacity=4.0, efficiency=efficiency
    )

    info = model.optimize(
        prices=prices, initial_charge=1.0, timestep='30min'
    )

    for rec in info:
        gross = rec['Gross [MW]']
        net = rec['Net [MW]']
        loss = rec['Losses [MW]']
        export = rec['Export [MW]']
        imp = rec['Import [MW]']

        #  discharging
        if gross < 0:
            print('gross, net, loss, export, import')
            print(gross, net, loss, export, imp)
            np.testing.assert_almost_equal(loss, (1-efficiency) * gross * -1)
