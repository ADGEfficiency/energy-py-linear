"""Interval variable testing."""
from rich import print

import energypylinear as epl
from energypylinear.interval_data import IntervalVars


def test_interval_vars() -> None:
    """
    for interval
        make some lp vars
        add them to vars

    get the evs objects for spill and non-spill
    """

    ds = epl.data_generation.generate_random_ev_input_data(
        48, n_chargers=3, charge_length=3, n_charge_events=12, seed=42
    )
    asset = epl.EVs(**ds)

    optimizer = epl.Optimizer()

    evs = asset.one_interval(optimizer, i=0, freq=epl.Freq(60))

    ivars = IntervalVars()
    site = asset.site.one_interval(
        optimizer, asset.site.cfg, i=0, freq=epl.freq.Freq(60)
    )
    ivars.append([site])
    ivars.append(evs)
    ivars.filter_objective_variables(epl.assets.evs.EVOneInterval, 0, asset.cfg.name)
    ivars[0]
    ivars[-1]
