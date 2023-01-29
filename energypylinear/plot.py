import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

import energypylinear as epl

mpl.rcParams["axes.titlesize"] = 10


def find_column(df: pd.DataFrame, start: str, end: str) -> str:
    cols = [c for c in df.columns if c.startswith(start) and c.endswith(end)]
    assert len(cols) == 1
    return cols[0]


def plot_battery(results: "epl.results.SimulationResult", path: pathlib.Path):
    fig, axes = plt.subplots(nrows=5, sharex=True, figsize=(12, 8))
    results = results.simulation

    results["Index"] = np.arange(results.shape[0]).tolist()

    results["import-export-balance"] = (
        results["import_power_mwh"] - results["export_power_mwh"]
    )
    results.plot(
        ax=axes[0],
        x="Index",
        y="import-export-balance",
    )
    axes[0].set_title("Power Balance MWh (Import Positive)")
    axes[0].set_ylabel("MWh")
    #  TODO will need some work in a multi-battery world
    results["net-battery-charge"] = (
        results[find_column(results, "battery-", "-charge_mwh")]
        - results[find_column(results, "battery-", "-discharge_mwh")]
    )
    results.plot(
        ax=axes[1],
        x="Index",
        y="net-battery-charge",
    )
    axes[1].set_title("Battery Charge / Discharge MWh (Charge Positive)")
    axes[1].set_ylabel("MWh")

    results.plot(
        "Index",
        "battery-final_charge_mwh",
        ax=axes[2],
        kind="bar",
    )
    axes[2].set_title("Battery SOC MWh (End of Interval)")
    axes[2].set_ylabel("MWh")

    results.plot(
        "Index",
        "electricity_prices",
        ax=axes[3],
    )
    axes[3].set_title("Electricity Prices")
    axes[3].set_ylabel("$/MWh")
    results.plot(
        "Index",
        "electricity_carbon_intensities",
        ax=axes[4],
        label="Carbon Intensities",
    )
    axes[4].set_title("Carbon Intensities")
    axes[4].set_ylabel("tC/MWh")

    for ax in axes:
        ax.get_legend().remove()

    plt.tight_layout()

    if path.is_dir():
        fig.savefig(path / "battery.png")
    else:
        fig.savefig(path)


def plot_evs(results: "epl.results.SimulationResult", path: pathlib.Path):
    fig, axes = plt.subplots(nrows=5)
    interval_data = results.interval_data
    results = results.simulation

    charger_usage = results[
        [
            c
            for c in results.columns
            if c.startswith("charger-") and c.endswith("-charge_mwh")
        ]
    ].values
    charge_event_usage = results[
        [
            c
            for c in results.columns
            if c.startswith("charge-event-") and c.endswith("total-charge_mwh")
        ]
    ].values

    fig, axes = plt.subplots(
        ncols=4, figsize=(14, 6), width_ratios=(5, 10, 1, 1), sharey=True
    )

    heatmap_config = {
        "annot_kws": {
            "size": 6,
        },
        "annot": True,
    }
    data = charger_usage
    seaborn.heatmap(
        data, ax=axes[0], **heatmap_config, mask=data == 0, fmt="g", cbar=False
    )
    axes[0].set_ylabel("Time")
    axes[0].set_xlabel("Chargers")

    data = charge_event_usage
    #  want to unmask out the periods where charge_event was positive
    seaborn.heatmap(
        data,
        ax=axes[1],
        **heatmap_config,
        mask=interval_data.evs.charge_events == 0,
        fmt="g",
        cbar_kws={"label": "Charge MWh", "location": "left"}
    )
    axes[1].set_xlabel("Charge Events")

    spill_charge_usage = results["charger-spill-charge_mwh"].values.reshape(-1, 1)
    data = spill_charge_usage
    seaborn.heatmap(
        data, ax=axes[2], **heatmap_config, xticklabels=["spill"], fmt="g", cbar=False
    )

    data = np.array(results["electricity_prices"]).reshape(-1, 1)
    seaborn.heatmap(
        data, ax=axes[3], **heatmap_config, xticklabels=["price"], fmt="g", cbar=False
    )

    plt.tight_layout()
    pathlib.Path("./figs").mkdir(
        exist_ok=True,
        parents=True,
    )

    if path.is_dir():
        fig.savefig(path / "evs.png")
    else:
        fig.savefig(path)


def plot_chp(results: "epl.results.SimulationResult", path: pathlib.Path):
    results = results.simulation
    results.to_csv("temp.csv")
    fig, axes = plt.subplots(nrows=5)
    results["idx"] = np.arange(results.shape[0]).tolist()

    results["import-export-balance"] = (
        results["import_power_mwh"] - results["export_power_mwh"]
    )
    results.plot(
        ax=axes[0],
        x="idx",
        y="import-export-balance",
        label="Power Balance MWh (Import Positive)",
    )
    results.plot(
        ax=axes[1],
        x="idx",
        y="boiler-high_temperature_generation_mwh",
        label="Boiler Heat Generation MWh",
    )
    results.plot(
        ax=axes[2],
        x="idx",
        y="spill-default-low_temperature_load_mwh",
        label="Low Temperature Heat Dump MWh",
    )
    results.plot(
        "idx",
        "electricity_prices",
        ax=axes[3],
        label="Electricity Prices",
    )
    results.plot(
        "idx",
        "electricity_carbon_intensities",
        ax=axes[4],
        label="Carbon Intensities",
    )

    if path.is_dir():
        fig.savefig(path / "battery.png")
    else:
        fig.savefig(path)
