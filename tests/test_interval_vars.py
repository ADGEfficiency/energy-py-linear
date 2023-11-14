"""Interval variable testing."""

import energypylinear as epl
from energypylinear.interval_data import IntervalVars


def test_interval_vars() -> None:
    """Tests the `epl.IntervalVars` object."""
    ds = epl.data_generation.generate_random_ev_input_data(
        48, n_chargers=3, charge_length=3, n_charge_events=12, seed=42
    )
    asset = epl.EVs(**ds)
    asset_two = epl.EVs(**ds, name="evs-two")
    optimizer = epl.Optimizer()

    evs = asset.one_interval(optimizer, i=0, freq=epl.Freq(60))
    evs_two = asset_two.one_interval(optimizer, i=0, freq=epl.Freq(60))
    site = asset.site.one_interval(
        optimizer, asset.site.cfg, i=0, freq=epl.freq.Freq(60)
    )

    ivars = IntervalVars()
    ivars.append([site, *evs, *evs_two])

    assert (
        len(
            ivars.filter_objective_variables(
                instance_type=epl.assets.evs.EVOneInterval, i=0
            )
        )
        == 3 * 12 * 2
    )

    assert (
        len(
            ivars.filter_objective_variables(
                instance_type=epl.assets.evs.EVSpillOneInterval, i=0
            )
        )
        == 1 * 12 * 2
    )

    assert (
        len(
            ivars.filter_objective_variables(
                instance_type=epl.assets.evs.EVOneInterval,
                i=0,
                asset_name=asset.cfg.name,
            )
        )
        == 3 * 12
    )
    ivars[0]
    ivars[-1]
