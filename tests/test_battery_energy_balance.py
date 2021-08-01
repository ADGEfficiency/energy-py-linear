import numpy as np

import energypylinear


def test_battery_energy_balance():

    prices = np.random.uniform(-100, 100, 96)
    power = 1.0
    capacity = 1.0
    timestep = '1hr'
    efficiency = 0.9
    initial_charge = 1.0

    model = energypylinear.Battery(
        power=power, capacity=capacity, efficiency=efficiency
    )

    info = model.optimize(
        prices=prices, initial_charge=initial_charge, timestep=timestep
    )

    energy_in = sum([res['Import [MW]'] for res in info])
    energy_out = sum([res['Export [MW]'] for res in info])

    initial_charge = info[0]['Initial charge [MWh]']
    final_charge = info[-1]['Final charge [MWh]']

    balance = energy_in - energy_out + (initial_charge - final_charge)
    np.testing.assert_almost_equal(balance, 0)
