"""Test electric vehicle asset."""
import hypothesis
import numpy as np

import energypylinear as epl
from energypylinear.flags import Flags


def test_evs_optimization_price() -> None:
    """Test EV optimization for price."""

    charge_events_capacity_mwh = [50, 100, 30, 40]
    evs = epl.evs.EVs(
        chargers_power_mw=[100, 100],
        charge_events_capacity_mwh=charge_events_capacity_mwh,
        charger_turndown=0.0,
    )
    results = evs.optimize(
        electricity_prices=[-100, 50, 30, 50, 40],
        charge_events=[
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0],
        ],
        flags=Flags(
            allow_evs_discharge=False,
            fail_on_spill_asset_use=True,
            allow_infeasible=False,
        ),
        freq_mins=60,
    )
    simulation = results.simulation
    print(simulation[[c for c in simulation.columns if "soc" in c]])
    breakpoint()  # fmt: skip

    #  test total import power equal to total charge event mwh
    #  requires efficiency to be 100%
    np.testing.assert_equal(
        simulation["site-import_power_mwh"].sum(), sum(charge_events_capacity_mwh)
    )

    #  no exporting at all
    np.testing.assert_equal(simulation["site-export_power_mwh"].sum(), 0)

    #  test dispatch exactly as we expect
    np.testing.assert_array_equal(
        simulation["site-import_power_mwh"], [50, 40, 100, 0, 30]
    )


def test_evs_optimization_carbon() -> None:
    """Test EV optimization for carbon."""
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
    simulation = results.simulation
    #  test total import power equal to total charge event mwh
    #  requires efficiency to be 100%
    np.testing.assert_equal(
        simulation["site-import_power_mwh"].sum(), sum(charge_event_mwh)
    )

    #  no exporting at all
    np.testing.assert_equal(simulation["site-export_power_mwh"].sum(), 0)

    #  test dispatch exactly as we expect
    np.testing.assert_array_equal(
        simulation["site-import_power_mwh"], [50.0, 0.0, 100.0, 0.0, 30.0, 40]
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
    idx_length: int,
    n_chargers: int,
    charger_turndown: float,
    charge_length: int,
    prices_mu: float,
    prices_std: float,
) -> None:
    """Test EV optimization using hypothesis."""
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
    evs.optimize(**ds)
