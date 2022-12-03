import numpy as np

import energypylinear as epl
from energypylinear.freq import Freq


def test_evs():
    evs = epl.evs.EVs(
        charger_mws=[2, 4],
    )
    evs.optimize(
        electricity_prices=[50, 50, 50, 50, 50],
        charge_events=[[1, 1, 1, 1, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 1]],
        charge_event_mwh=[50, 100, 25],
    )

    if False:
        import collections

        vars = collections.defaultdict(list)
        for i in idx:
            vars["evs"].append(
                ev_one_interval(optimizer, chargers, charge_events, i, freq)
            )
            vars["spill-evs"].append(
                ev_one_interval(
                    optimizer,
                    spill_chargers,
                    charge_events,
                    i,
                    freq,
                )
            )

            constrain_within_interval(
                optimizer,
                vars["evs"][i],
                charge_events,
                freq,
                chargers,
                i,
            )
            constrain_within_interval(
                optimizer,
                vars["spill-evs"][i],
                charge_events,
                freq,
                spill_chargers,
                i,
                add_single_charger_or_event_constraints=False,
            )

        constrain_after_intervals(optimizer, vars, charge_event_mwh)

        #  check the stack worked correctly
        #  TODO move after interval data refactor
        stacked_charge_mwh = stack_ev(vars, "charge_mwh")
        assert stacked_charge_mwh.shape[0] == len(idx)
        assert stacked_charge_mwh.shape[1] == charge_events.shape[1]
        assert (
            stacked_charge_mwh.shape[2] == chargers.shape[0] + spill_chargers.shape[0]
        )
        out = stacked_charge_mwh
