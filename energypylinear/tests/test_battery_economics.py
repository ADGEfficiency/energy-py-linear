import numpy as np
import pytest

import energypylinear


@pytest.mark.parametrize(
    'power, capacity, initial_charge, timestep',
    [
        (2, 4, 0, '1hr'),
        (4, 2, 2, '1hr'),
        (2, 4, 0, '30min'),
        (4, 2, 2, '30min'),
        (2, 4, 0, '5min'),
        (4, 2, 2, '5min')
    ]
)
def test_cost_calculation(power, capacity, initial_charge, timestep):
    prices = [10, 20, 30, -90, 50, 2000, -1]

    model = energypylinear.Battery(
        power=power, capacity=capacity, timestep=timestep, efficiency=1.0
    )

    info = model.optimize(
        prices=prices, initial_charge=initial_charge
    )

    dispatch = info.loc[:, 'Net [MW]'].values
    timestep = model.timestep
    step = model.step
    check_actual_costs = sum(dispatch[:-1] * prices[:-1]) / step 

    actual_costs = info.loc[:, 'Actual [$/{}]'.format(timestep)].values

    np.testing.assert_almost_equal(check_actual_costs, sum(actual_costs[:-1]))

