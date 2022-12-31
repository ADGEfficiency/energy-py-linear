import hypothesis
import numpy as np

import energypylinear as epl


def test_evs_optimization_price():
    evs = epl.evs.EVs(
        charger_mws=[100, 100],
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


def test_evs_optimization_carbon():
    evs = epl.evs.EVs(
        charger_mws=[100, 100],
    )
    charge_event_mwh = [50, 100, 30, 40]
    results = evs.optimize(
        electricity_prices=[-100, 50, 30, 50, 40, 10],
        electricity_carbon_intensities=[0.1, 0.3, -0.5, 0.9, 0.9, 0.0],
        charge_events=[
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
        ],
        charge_event_mwh=charge_event_mwh,
        objective="carbon",
    )
    #  test total import power equal to total charge event mwh
    #  requires efficiency to be 100%
    np.testing.assert_equal(results["import_power_mwh"].sum(), sum(charge_event_mwh))

    #  no exporting at all
    np.testing.assert_equal(results["export_power_mwh"].sum(), 0)

    #  test dispatch exactly as we expect
    np.testing.assert_array_equal(
        results["import_power_mwh"], [50.0, 0.0, 100.0, 0.0, 30.0, 40]
    )


@hypothesis.settings(
    print_blob=True,
    max_examples=200,
    verbosity=hypothesis.Verbosity.verbose,
    deadline=4000,
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
