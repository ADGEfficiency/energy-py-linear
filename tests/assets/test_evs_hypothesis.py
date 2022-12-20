import hypothesis
import numpy as np
import pandas as pd

import energypylinear as epl


@hypothesis.settings(
    print_blob=True,
    max_examples=200,
    verbosity=hypothesis.Verbosity.verbose,
    deadline=1000,
)
@hypothesis.given(
    idx_length=hypothesis.strategies.integers(min_value=10, max_value=24),
    n_chargers=hypothesis.strategies.integers(min_value=4, max_value=24),
    charger_turndown=hypothesis.strategies.floats(min_value=0.1, max_value=0.4),
    charge_length=hypothesis.strategies.integers(min_value=2, max_value=22),
    prices_mu=hypothesis.strategies.floats(min_value=-1000, max_value=1000),
    prices_std=hypothesis.strategies.floats(min_value=0.1, max_value=25),
)
def test_evs_hypothesis(
    idx_length, n_chargers, charger_turndown, charge_length, prices_mu, prices_std
):
    #  TODO add given for electricity_prices mean + stdev
    ds = epl.data_generation.generate_random_ev_input_data(
        idx_length,
        n_chargers,
        charge_length,
        prices_mu=prices_mu,
        prices_std=prices_std,
    )
    evs = epl.evs.EVs(
        charger_mws=ds["charger_mws"].tolist(), charger_turndown=charger_turndown
    )
    ds.pop("charger_mws")
    results = evs.optimize(**ds)
