"""Test electric vehicle asset."""
import collections
import timeit

import hypothesis
import matplotlib.pyplot as plt
import numpy as np
import pytest

import energypylinear as epl
from energypylinear.flags import Flags


def test_evs_optimization_price() -> None:
    """Test EV optimization for price."""

    charge_events_capacity_mwh = [50, 100, 30, 40]
    evs = epl.evs.EVs(
        chargers_power_mw=[100, 100],
        charge_events_capacity_mwh=charge_events_capacity_mwh,
        charger_turndown=0.0,
        charge_event_efficiency=1.0,
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
    # print(simulation[[c for c in simulation.columns if "soc" in c]])

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
    charge_events_capacity_mwh = [50, 100, 30, 40]
    evs = epl.evs.EVs(
        chargers_power_mw=[100, 100],
        charge_events_capacity_mwh=charge_events_capacity_mwh,
        charger_turndown=0.0,
        charge_event_efficiency=1.0,
    )
    results = evs.optimize(
        electricity_prices=[-100, 50, 30, 50, 40, 10],
        electricity_carbon_intensities=[0.1, 0.3, -0.5, 0.95, 0.9, 0.0],
        charge_events=[
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
        ],
        flags=Flags(
            allow_evs_discharge=False,
            fail_on_spill_asset_use=True,
            allow_infeasible=False,
        ),
        freq_mins=60,
        objective="carbon",
    )
    simulation = results.simulation
    #  test total import power equal to total charge event mwh
    #  requires efficiency to be 100%
    np.testing.assert_equal(
        simulation["site-import_power_mwh"].sum(), sum(charge_events_capacity_mwh)
    )

    #  no exporting at all
    np.testing.assert_equal(simulation["site-export_power_mwh"].sum(), 0)

    #  test dispatch exactly as we expect
    np.testing.assert_array_equal(
        simulation["site-import_power_mwh"], [50.0, 0.0, 100.0, 0.0, 30.0, 40]
    )


@pytest.mark.parametrize(
    "efficiency",
    [
        1.0,
        0.9,
        0.5,
        # 0.1,  TODO causes problems
        # 0.0,
        # -0.1
    ],
)
def test_evs_efficiency_losses(efficiency: float) -> None:
    """Test the EV charge event losses."""
    charge_events_capacity_mwh = [50, 100, 30, 40]
    evs = epl.evs.EVs(
        chargers_power_mw=[500, 500],
        charge_events_capacity_mwh=charge_events_capacity_mwh,
        charger_turndown=0.0,
        charge_event_efficiency=efficiency,
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
    # print(simulation[[c for c in simulation.columns if "soc" in c]])

    #  test total import versus charge event mwh
    #  a less efficient charge event should mean we import more power
    #  our expected losses are a function of the amount we charge a charge event

    _ = simulation["total-electric_loss_mwh"].sum()
    # np.testing.assert_equal(
    #     simulation["site-import_power_mwh"].sum(),
    #     sum(charge_events_capacity_mwh) + losses_mwh,
    # )
    # print(
    #     f'import power: {simulation["site-import_power_mwh"].sum()}, charge event {sum(charge_events_capacity_mwh)}, {losses_mwh=}'
    # )

    np.testing.assert_allclose(
        simulation["total-electric_charge_mwh"] * (1 - efficiency),
        simulation["total-electric_loss_mwh"],
    )
    #  TODO in future add charger efficiency
    #  this would allow matching of different efficiency chargers and charge events


def test_v2g():
    """
    this test is stochastic

    control seeds so that we run the same seed in each trial over different charge event lengths

    this reduces the variance between trials - each trial uses the same prices
    """
    import collections
    import statistics

    num_trials = 25
    seeds = np.random.randint(0, 1000, size=num_trials)
    discharge = collections.defaultdict(list)
    for charge_event_length in range(3, 24, 2):

        trials = collections.defaultdict(list)
        for n_trial, seed in enumerate(seeds):
            #  here we are resampling prices each time
            #  really shouldn't - would need a bit of work TODO
            ds = epl.data_generation.generate_random_ev_input_data(
                48,
                n_chargers=3,
                charge_length=charge_event_length,
                n_charge_events=12,
                prices_mu=500,
                prices_std=250,
                seed=seed,
            )
            # print(
            #     f"{charge_event_length=} {n_trial=} prices: {np.mean(ds['electricity_prices'])}"
            # )

            evs = epl.evs.EVs(
                chargers_power_mw=ds["charger_mws"].tolist(),
                charge_events_capacity_mwh=ds["charge_events_capacity_mwh"].tolist(),
                charger_turndown=0.0,
                charge_event_efficiency=1.0,
            )
            ds.pop("charger_mws")
            ds.pop("charge_events_capacity_mwh")
            results = evs.optimize(
                **ds,
                flags=Flags(
                    allow_evs_discharge=True,
                    fail_on_spill_asset_use=True,
                    allow_infeasible=False,
                ),
                freq_mins=60,
                verbose=False,
            )
            simulation = results.simulation
            trials["discharge"].append(simulation["total-electric_discharge_mwh"].sum())

        discharge["mean"].append(statistics.mean(trials["discharge"]))
        discharge["stdev"].append(statistics.stdev(trials["discharge"]))

        #  check at each step to fail early
        # print(discharge)
        assert all(
            x <= y
            for x, y in zip(discharge["mean"][:-1], discharge["mean"][1:], strict=True)
        ), f"discharge should be increasing - it decreased on {charge_event_length=}"

    # from energypylinear.debug import debug_simulation
    # debug_simulation(results.simulation)
    """
    how to actuall test this
    - check that some of the electric_discharge_mwh is positive

    check final soc are correct
    """


@hypothesis.settings(
    print_blob=True,
    max_examples=200,
    verbosity=hypothesis.Verbosity.verbose,
    # deadline=4000,  # with no v2g
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
    evs = epl.evs.EVs(
        chargers_power_mw=ds["charger_mws"].tolist(),
        charge_events_capacity_mwh=ds["charge_events_capacity_mwh"].tolist(),
        charger_turndown=charger_turndown,
    )
    ds.pop("charger_mws")
    ds.pop("charge_events_capacity_mwh")
    evs.optimize(
        **ds,
        verbose=True,
        flags=Flags(
            allow_evs_discharge=v2g,
            fail_on_spill_asset_use=False,
            allow_infeasible=False,
        ),
    )


def test_evs_performance():
    """Test the Battery run time perforamnce."""
    idx_lengths = [
        6,
        #  one day 60 min freq
        24,
        #  one week 60 min freq
        168,
        #  one week 15 min freq
        672,
        #  two weeks
        1344,
    ]
    data = collections.defaultdict(list)
    for flag in [False, True]:
        for idx_length in idx_lengths:
            start_time = timeit.default_timer()
            # print(f"idx_length: {idx_length}")

            ds = epl.data_generation.generate_random_ev_input_data(
                idx_length,
                n_chargers=2,
                charge_length=10,
                n_charge_events=24,
                prices_mu=500,
                prices_std=10,
            )
            asset = epl.evs.EVs(
                chargers_power_mw=ds["charger_mws"].tolist(),
                charge_events_capacity_mwh=ds["charge_events_capacity_mwh"].tolist(),
                charger_turndown=0.2,
            )
            ds.pop("charger_mws")
            ds.pop("charge_events_capacity_mwh")
            asset.optimize(
                **ds,
                verbose=False,
                flags=Flags(
                    allow_evs_discharge=True,
                    fail_on_spill_asset_use=False,
                    allow_infeasible=False,
                    limit_charge_variables_to_valid_events=flag,
                ),
            )

            elapsed = timeit.default_timer() - start_time
            data["time"].append(elapsed)
            data["pkg"].append(
                {"idx_length": idx_length, "time": elapsed, "flag": flag}
            )
            # print(
            #     f"idx_length: {idx_length}, elapsed: {elapsed:2.2f} sec, flag: {flag}"
            # )

    plt.figure()
    for flag in [True, False]:
        subset = [p for p in data["pkg"] if p["flag"] == flag]
        plt.plot(
            [p["idx_length"] for p in subset],
            [p["time"] for p in subset],
            "o-",
            label=f"prune charge variable to only valid charge events: {flag}",
        )
        plt.xlabel("Index Length")
        plt.ylabel("Run Time (seconds)")
        plt.legend()
        plt.title(asset.__repr__())
        plt.grid(True)
        plt.savefig("./docs/docs/static/evs-performance.png")
