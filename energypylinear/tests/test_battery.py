import pytest

import energypylinear


def map_values(results, keyName):
    return [res[keyName] for res in results if res[keyName] is not None]


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
    dispatch = map_values(info, 'Gross [MW]')
    assert(max(dispatch) <= power)
    assert(min(dispatch) >= -power)

    #  check we don't exceed battery capacity
    charges = map_values(info, 'Charge [MWh]')
    assert(max(charges) <= capacity)
    assert(min(charges) >= 0)

    #  check we set initial charge correctly
    assert(charges[0] == initial_charge)

    # check gross is greater or eq to net
    gross = [abs(x) for x in map_values(info, 'Gross [MW]')]
    net = [abs(x) for x in map_values(info, 'Net [MW]')]
    losses = map_values(info, 'Losses [MW]')

    assert all([g >= n for g, n in zip(gross, net)])

    # check losses are smaller or eq to export
    export = map_values(info, 'Export [MW]')
    losses = map_values(info, 'Losses [MW]')

    assert all([l <= e for l, e in zip(losses, export)])
