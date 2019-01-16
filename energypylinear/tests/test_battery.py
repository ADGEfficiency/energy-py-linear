import numpy as np
import pytest

import energypylinear


@pytest.mark.parametrize(
    'power, capacity, initial_charge',
    [(2, 4, 0), (4, 2, 2)]
)
def test_power_capacity_initial_charge(power, capacity, initial_charge):
    prices = [10, 20, 30]

    model = energypylinear.Battery(
        power=power, capacity=capacity
    )

    info = model.optimize(
        prices=prices, initial_charge=initial_charge
    )

    #  check we don't charge or discharge more than battery rating
    dispatch = info.loc[:, 'Power [MW]'].values
    assert(max(dispatch) <= power)
    assert(min(dispatch) >= -power)

    #  check we don't exceed battery capacity
    charges = info.loc[:, 'Charge [MWh]'].values
    assert(max(charges) <= capacity)
    assert(min(charges) >= 0)

    #  check we set initial charge correctly
    assert(charges[0] == initial_charge)
