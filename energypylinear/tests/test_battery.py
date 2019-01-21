import numpy as np
import pytest

import energypylinear


@pytest.mark.parametrize(
    'power, capacity, initial_charge',
    [(2, 4, 0), (4, 2, 2)]
)
def test_power_capacity_initial_charge(power, capacity, initial_charge):
    prices = [10, 20, 30, 40, 10, 50]

    model = energypylinear.Battery(
        power=power, capacity=capacity
    )

    info = model.optimize(
        prices=prices, initial_charge=initial_charge
    )

    #  check we don't charge or discharge more than battery rating
    dispatch = info.loc[:, 'Gross [MW]'].values
    assert(max(dispatch) <= power)
    assert(min(dispatch) >= -power)

    #  check we don't exceed battery capacity
    charges = info.loc[:, 'Charge [MWh]'].values
    assert(max(charges) <= capacity)
    assert(min(charges) >= 0)

    #  check we set initial charge correctly
    assert(charges[0] == initial_charge)

    #  check gross is always bigger than net
    gross = np.abs(info.loc[:, 'Gross [MW]'].values)
    net = np.abs(info.loc[:, 'Net [MW]'].values)

    np.testing.assert_array_compare(
        np.greater_equal, gross, net
    )

    #  check losses are smaller than export
    gross = info.loc[:, 'Gross [MW]'].values
    net = info.loc[:, 'Net [MW]'].values
    export = info.loc[:, 'Export [MW]'].values
    losses = info.loc[:, 'Losses [MW]'].values

    np.testing.assert_array_compare(
        np.greater_equal, export, losses
    )
