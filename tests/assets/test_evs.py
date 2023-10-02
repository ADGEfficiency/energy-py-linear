"""Test electric vehicle asset."""
import collections
import statistics
from concurrent.futures import ProcessPoolExecutor

import hypothesis
import numpy as np
import pytest

import energypylinear as epl
from energypylinear.flags import Flags


def test_evs_optimization_price() -> None:
    """Test EV optimization for price."""

    charge_events_capacity_mwh = [50.0, 100, 30, 40]
    evs = epl.EVs(
        chargers_power_mw=[100, 100],
        charge_events_capacity_mwh=charge_events_capacity_mwh,
        charger_turndown=0.0,
        charge_event_efficiency=1.0,
        electricity_prices=[-100, 50, 30, 50, 40],
        charge_events=[
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0],
        ],
        freq_mins=60,
    )
    simulation = evs.optimize(
        flags=Flags(
            allow_evs_discharge=False,
            fail_on_spill_asset_use=True,
            allow_infeasible=False,
        ),
        verbose=False,
    )
    # print(simulation[[c for c in simulation.columns if "soc" in c]])

    #  test total import power equal to total charge event mwh
    #  requires efficiency to be 100%
    np.testing.assert_equal(
        simulation.results["site-import_power_mwh"].sum(),
        sum(charge_events_capacity_mwh),
    )

    #  no exporting at all
    np.testing.assert_equal(simulation.results["site-export_power_mwh"].sum(), 0)

    #  test dispatch exactly as we expect
    np.testing.assert_array_equal(
        simulation.results["site-import_power_mwh"], [50, 40, 100, 0, 30]
    )


def test_evs_optimization_carbon() -> None:
    """Test EV optimization for carbon."""
    charge_events_capacity_mwh = [50.0, 100, 30, 40]
    evs = epl.EVs(
        chargers_power_mw=[100, 100],
        charge_events_capacity_mwh=charge_events_capacity_mwh,
        charger_turndown=0.0,
        charge_event_efficiency=1.0,
        electricity_prices=[-100, 50, 30, 50, 40, 10],
        electricity_carbon_intensities=[0.1, 0.3, -0.5, 0.95, 0.9, 0.0],
        charge_events=[
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
        ],
        freq_mins=60,
    )
    simulation = evs.optimize(
        flags=Flags(
            allow_evs_discharge=False,
            fail_on_spill_asset_use=True,
            allow_infeasible=False,
        ),
        objective="carbon",
        verbose=False,
    )
    #  test total import power equal to total charge event mwh
    #  requires efficiency to be 100%
    np.testing.assert_equal(
        simulation.results["site-import_power_mwh"].sum(),
        sum(charge_events_capacity_mwh),
    )

    #  no exporting at all
    np.testing.assert_equal(simulation.results["site-export_power_mwh"].sum(), 0)

    #  test dispatch exactly as we expect
    np.testing.assert_array_equal(
        simulation.results["site-import_power_mwh"], [50.0, 0.0, 100.0, 0.0, 30.0, 40]
    )


@pytest.mark.parametrize(
    "efficiency",
    [
        1.0,
        0.9,
        0.5,
        0.2,
        # 0.1, causes spill - not enough power to charge
    ],
)
def test_evs_efficiency_losses(efficiency: float) -> None:
    """Test the EV charge event losses."""
    charge_events_capacity_mwh: list[float] = [50, 100, 30, 40]
    evs = epl.EVs(
        chargers_power_mw=[500, 500],
        charge_events_capacity_mwh=charge_events_capacity_mwh,
        charger_turndown=0.0,
        charge_event_efficiency=efficiency,
        electricity_prices=[-100, 50, 30, 50, 40],
        charge_events=[
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0],
        ],
        freq_mins=60,
    )
    simulation = evs.optimize(
        flags=Flags(
            allow_evs_discharge=False,
            fail_on_spill_asset_use=True,
            allow_infeasible=False,
        ),
        verbose=False,
    )

    #  test total import versus charge event mwh
    #  a less efficient charge event should mean we import more power
    #  our expected losses are a function of the amount we charge a charge event

    #  TODO
    _ = simulation.results["total-electric_loss_mwh"].sum()
    # np.testing.assert_equal(
    #     simulation.results["site-import_power_mwh"].sum(),
    #     sum(charge_events_capacity_mwh) + losses_mwh,
    # )
    # print(
    #     f'import power: {simulation.results["site-import_power_mwh"].sum()}, charge event {sum(charge_events_capacity_mwh)}, {losses_mwh=}'
    # )

    np.testing.assert_allclose(
        simulation.results["total-electric_charge_mwh"] * (1 - efficiency),
        simulation.results["total-electric_loss_mwh"],
    )
    #  TODO in future add charger efficiency
    #  this would allow matching of different efficiency chargers and charge events


def _one_v2g(args: tuple) -> tuple:
    """
    Runs a single opmitization with a randomly generated EV dataset.

    Helper for `test_evs_v2g_fast`.
    """
    seed, charge_event_length = args

    ds = epl.data_generation.generate_random_ev_input_data(
        48,
        n_chargers=3,
        charge_length=charge_event_length,
        n_charge_events=12,
        prices_mu=500,
        prices_std=250,
        seed=seed,
    )
    evs = epl.EVs(
        **ds,
        charger_turndown=0.0,
        charge_event_efficiency=1.0,
        freq_mins=60,
    )
    charge_events_capacity_mwh = ds["charge_events_capacity_mwh"]
    simulation = evs.optimize(
        flags=Flags(
            allow_evs_discharge=True,
            fail_on_spill_asset_use=True,
            allow_infeasible=False,
        ),
        verbose=False,
    )

    #  test that the initial and final socs are correct
    cols = [
        c
        for c in simulation.results
        if "evs-charge-event" in c and "final_soc_mwh" in c
    ]
    subset = simulation.results[cols]
    np.testing.assert_array_almost_equal(
        subset.iloc[-1, :].values, charge_events_capacity_mwh
    )
    return (simulation.results, charge_event_length, seed)


def test_v2g() -> None:
    """Test that we discharge more power as the charge events get longer."""
    num_trials = 25
    charge_event_lengths = range(3, 24, 5)
    args = [
        (seed, charge_event_length)
        for charge_event_length in charge_event_lengths
        for seed in np.random.randint(0, 1000, size=num_trials)
    ]
    with ProcessPoolExecutor() as executor:
        trials = list(executor.map(_one_v2g, args))

    discharge = collections.defaultdict(list)
    for charge_event_length in charge_event_lengths:
        trial_results = [
            x[0]["total-electric_discharge_mwh"].sum()
            for x in trials
            if x[1] == charge_event_length
        ]
        discharge["mean"].append(statistics.mean(trial_results))
        discharge["stdev"].append(statistics.stdev(trial_results))

    assert all(
        x <= y
        for x, y in zip(discharge["mean"][:-1], discharge["mean"][1:], strict=True)
    ), "discharge should be increasing always as charge event increases"


@hypothesis.settings(
    print_blob=True,
    max_examples=10,
    verbosity=hypothesis.Verbosity.verbose,
    deadline=200000,  # enough for v2g ?
)
@hypothesis.given(
    idx_length=hypothesis.strategies.integers(min_value=10, max_value=24),
    n_charge_events=hypothesis.strategies.integers(min_value=4, max_value=24),
    charger_turndown=hypothesis.strategies.floats(min_value=0.1, max_value=0.4),
    charge_length=hypothesis.strategies.integers(min_value=2, max_value=22),
    prices_mu=hypothesis.strategies.floats(min_value=-1000, max_value=1000),
    prices_std=hypothesis.strategies.floats(min_value=0.1, max_value=25),
    v2g=hypothesis.strategies.booleans(),
)
def test_evs_hypothesis(
    idx_length: int,
    n_charge_events: int,
    charger_turndown: float,
    charge_length: int,
    prices_mu: float,
    prices_std: float,
    v2g: bool,
) -> None:
    """Test EV optimization using hypothesis."""
    ds = epl.data_generation.generate_random_ev_input_data(
        idx_length,
        n_chargers=3,
        charge_length=charge_length,
        n_charge_events=n_charge_events,
        prices_mu=prices_mu,
        prices_std=prices_std,
    )
    evs = epl.EVs(
        **ds,
        charger_turndown=charger_turndown,
    )
    evs.optimize(
        verbose=True,
        flags=Flags(
            allow_evs_discharge=v2g,
            fail_on_spill_asset_use=False,
            allow_infeasible=False,
        ),
    )
