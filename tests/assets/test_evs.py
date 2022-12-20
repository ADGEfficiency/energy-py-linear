import numpy as np

import energypylinear as epl
from energypylinear.freq import Freq

"""
TODO
- test smaller charger mws
- test overlap between chargers
- test use of the spill chargers
    - make spill charger optional or choose to throw an error when spill charger used?
"""


def test_evs():
    evs = epl.evs.EVs(
        charger_mws=[100, 100],
        # charger_efficiency=1.0  TODO
    )
    charge_event_mwh = [50, 100, 30, 40]
    results = evs.optimize(
        electricity_prices=[-100, 50, 30, 50, 40],
        charge_events=[
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0],
        ],
        charge_event_mwh=charge_event_mwh,
    )
    #  test total import power equal to total charge event mwh
    #  requires efficiency to be 100%
    np.testing.assert_equal(results["import_power_mwh"].sum(), sum(charge_event_mwh))
    #  no exporting at all
    np.testing.assert_equal(results["export_power_mwh"].sum(), 0)

    #  test dispatch exactly as we expect
    np.testing.assert_array_equal(results["import_power_mwh"], [50, 40, 100, 0, 30])
