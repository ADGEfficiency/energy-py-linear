import pytest

import energypylinear


@pytest.mark.parametrize(
    'prices, initial_charge, carbon',
    [
        ([20, 10, 10], 1, [10, 20, 20]),
        ([10, 50, 10, 50, 10], 0, [50, 10, 50, 10, 50])
    ]
)
def test_carbon_optimization(prices, initial_charge, carbon):
    power = 4
    capacity = 6

    model = energypylinear.Battery(
        power=power, capacity=capacity, efficiency=1.0
    )

    p_info = model.optimize(
        prices=prices, initial_charge=initial_charge, timestep='1hr'
    )

    p_dispatch = [res['Actual [$/1hr]'] for res in p_info]

    # now we optimize for carbon and see if our net cost was higher
    c_info = model.optimize(
        prices=prices,
        carbon=carbon,
        initial_charge=initial_charge,
        timestep='1hr',
        objective='carbon'
    )
    c_dispatch = [res['Actual [$/1hr]'] for res in c_info]

    assert sum(c_dispatch) >= sum(p_dispatch)

