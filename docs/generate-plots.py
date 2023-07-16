"""Generate plots for the documentation."""
import collections
import statistics
import time
import timeit

import matplotlib.pyplot as plt
import numpy as np
from rich import print

import energypylinear as epl
from energypylinear.flags import Flags
from energypylinear.logger import logger


def test_battery_performance() -> None:
    """Test the Battery run time perforamnce."""
    idx_lengths = [
        6,
        # one day 60 min freq
        24,
        # one week 60 min freq
        168,
        # one week 15 min freq
        672,
        # two weeks
        1344,
    ]
    num_trials = 15

    run_times = collections.defaultdict(list)
    for flag in [False, True]:
        for idx_length in idx_lengths:
            trial_times = collections.defaultdict(list)

            for n_trial in range(num_trials):
                print(f"idx_length: {idx_length} trial {n_trial}")
                st = time.perf_counter()

                ds = {"electricity_prices": np.random.uniform(-1000, 1000, idx_length)}
                asset = epl.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)

                asset.optimize(
                    electricity_prices=ds["electricity_prices"],
                    verbose=False,
                    flags=Flags(
                        allow_evs_discharge=True,
                        fail_on_spill_asset_use=True,
                        allow_infeasible=False,
                        include_charge_discharge_binary_variables=flag,
                    ),
                )

                trial_times["time"].append(time.perf_counter() - st)

            run_times["time"].append(
                {
                    "mean": statistics.mean(trial_times["time"]),
                    "std": statistics.stdev(trial_times["time"]),
                    "flag": flag,
                    "idx_length": idx_length,
                }
            )
            print(run_times["time"])

    fig, axes = plt.subplots(nrows=1, sharex=True)
    print("[red]final run times:[/]")
    print(run_times)
    for flag in [False, True]:

        subset: list = [p for p in run_times["time"] if p["flag"] == flag]
        axes.plot(
            [p["idx_length"] for p in subset],
            [p["mean"] for p in subset],
            marker="o",
            label=f"include_charge_discharge_binary_variables={flag}",
        )
        axes.set_title(asset.__repr__())
        axes.set_ylabel("Run Time (seconds)")
        axes.legend()
        axes.grid(True)
        plt.xlabel("Index Length")
        plt.tight_layout()
        fig.savefig("./docs/docs/static/battery-performance.png")


def test_evs_performance() -> None:
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

            ds = epl.data_generation.generate_random_ev_input_data(
                idx_length,
                n_chargers=2,
                charge_length=10,
                n_charge_events=24,
                prices_mu=500,
                prices_std=10,
            )
            asset = epl.EVs(
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
            data["pkg"].append(
                {"idx_length": idx_length, "time": elapsed, "flag": flag}
            )
            logger.info(
                "test_evs_performance",
                idx_length=idx_length,
                elapsed=elapsed,
                flag=flag,
            )

    plt.figure()
    for flag in [True, False]:
        subset: list = [p for p in data["pkg"] if p["flag"] == flag]
        plt.plot(
            [p["idx_length"] for p in subset],
            [p["time"] for p in subset],
            "o-",
            label=f"limit_charge_variables_to_valid_events: {flag}",
        )
        plt.xlabel("Index Length")
        plt.ylabel("Run Time (seconds)")
        plt.legend()
        plt.title(asset.__repr__())
        plt.grid(True)
        plt.savefig("./docs/docs/static/evs-performance.png")


if __name__ == "__main__":
    test_battery_performance()
    test_evs_performance()
