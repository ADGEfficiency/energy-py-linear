import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

import energypylinear as epl


def find_column(df: pd.DataFrame, start: str, end: str) -> str:
    cols = [c for c in df.columns if c.startswith(start) and c.endswith(end)]
    assert len(cols) == 1
    return cols[0]


def plot_battery(results: pd.DataFrame, path: pathlib.Path):
    fig, axes = plt.subplots(nrows=5)

    results["import-export-balance"] = (
        results["import_power_mwh"] - results["export_power_mwh"]
    )
    results.plot(
        ax=axes[0],
        y="import-export-balance",
        label="Power Balance MWh (Import Positive)",
    )
    #  TODO will need some work in a multi-battery world
    results["net-battery-charge"] = (
        results[find_column(results, "battery-", "-charge_mwh")]
        - results[find_column(results, "battery-", "-discharge_mwh")]
    )
    results.plot(
        ax=axes[1],
        y="net-battery-charge",
        label="Battery Charge / Discharge MWh (Charge Positive)",
    )

    results["idx"] = np.arange(results.shape[0]).tolist()
    results.plot(
        "idx",
        "battery-final_charge_mwh",
        ax=axes[2],
        label="Battery SOC MWh (End of Interval)",
        kind="bar",
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


def plot_evs(
    results: pd.DataFrame, interval_data: epl.data.IntervalData, path: pathlib.Path
):
    fig, axes = plt.subplots(nrows=5)

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
    seaborn.heatmap(data, ax=axes[0], **heatmap_config, mask=data == 0)

    data = charge_event_usage
    #  want to unmask out the periods where charge_event was positive
    seaborn.heatmap(
        data, ax=axes[1], **heatmap_config, mask=interval_data.evs.charge_events == 0
    )

    spill_charge_usage = results["charger-spill-charge_mwh"].values.reshape(-1, 1)
    data = spill_charge_usage
    seaborn.heatmap(data, ax=axes[2], **heatmap_config, xticklabels=["spill"])

    data = np.array(results["electricity_prices"]).reshape(-1, 1)
    seaborn.heatmap(data, ax=axes[3], **heatmap_config, xticklabels=["price"])

    plt.tight_layout()
    pathlib.Path("./figs").mkdir(
        exist_ok=True,
        parents=True,
    )

    if path.is_dir():
        fig.savefig(path / "evs.png")
    else:
        fig.savefig(path)


def plot_chp(results: pd.DataFrame, path: pathlib.Path):
    results.to_csv("temp.csv")
    fig, axes = plt.subplots(nrows=5)
    results["idx"] = np.arange(results.shape[0]).tolist()

    results["import-export-balance"] = (
        results["import_power_mwh"] - results["export_power_mwh"]
    )
    results.plot(
        ax=axes[0],
        y="import-export-balance",
        label="Power Balance MWh (Import Positive)",
    )
    results.plot(
        ax=axes[1],
        y="boiler-high_temperature_generation_mwh",
        label="Boiler Heat Generation MWh",
    )
    results.plot(
        ax=axes[2],
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
        fig.savefig(path / "chp.png")
    else:
        fig.savefig(path)

    fig.savefig("temp.png")
