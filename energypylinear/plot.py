"""Plot functions for the energy asset linear programming simulation results."""
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

import energypylinear as epl

mpl.rcParams["axes.titlesize"] = 10


def find_column(df: pd.DataFrame, start: str, end: str) -> str:
    """Finds a column based on the start and end of the column name."""
    cols: list[str] = [
        c for c in df.columns.tolist() if c.startswith(start) and c.endswith(end)
    ]
    assert len(cols) == 1
    return cols[0]


def plot_battery(
    results: "epl.results.SimulationResult", path: pathlib.Path | str
) -> None:
    """Plot battery simulation results."""
    path = pathlib.Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(nrows=5, sharex=True, figsize=(12, 8))
    simulation = results.simulation

    simulation["Index"] = np.arange(simulation.shape[0]).tolist()

    simulation["import-export-balance"] = (
        simulation["site-import_power_mwh"] - simulation["site-export_power_mwh"]
    )
    simulation.plot(
        ax=axes[0],
        x="Index",
        y="import-export-balance",
    )
    axes[0].set_ylim()
    axes[0].set_title("Power Balance MWh (Import Positive)")
    axes[0].set_ylabel("MWh")

    #  TODO will need some work in a multi-battery world
    simulation["net-battery-charge"] = (
        simulation[find_column(simulation, "battery-", "-electric_charge_mwh")]
        - simulation[find_column(simulation, "battery-", "-electric_discharge_mwh")]
    )
    simulation.plot(
        ax=axes[1],
        x="Index",
        y="net-battery-charge",
    )
    axes[1].set_title("Battery Charge / Discharge MWh (Charge Positive)")
    axes[1].set_ylabel("MWh")

    simulation.plot(
        "Index",
        "battery-final_charge_mwh",
        ax=axes[2],
        kind="bar",
    )
    axes[2].set_title("Battery SOC MWh (End of Interval)")
    axes[2].set_ylabel("MWh")

    simulation.plot(
        "Index",
        "electricity_prices",
        ax=axes[3],
    )
    axes[3].set_title("Electricity Prices")
    axes[3].set_ylabel("$/MWh")
    simulation.plot(
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
        path = path / "battery.png"
    fig.savefig(path)


def plot_evs(results: "epl.results.SimulationResult", path: pathlib.Path) -> None:
    """Plot electric vehicle simulation results."""
    simulation = results.simulation

    charger_usage = simulation[
        [
            c
            for c in simulation.columns
            if c.startswith("charger-") and c.endswith("-electric_charge_mwh")
        ]
    ].values
    charge_event_usage = simulation[
        [
            c
            for c in simulation.columns
            if c.startswith("charge-event-") and c.endswith("electric_charge_mwh")
        ]
    ].values
    discharge_event_usage = simulation[
        [
            c
            for c in simulation.columns
            if c.startswith("charge-event-") and c.endswith("electric_discharge_mwh")
        ]
    ].values

    fig, axes = plt.subplots(
        ncols=5, figsize=(14, 6), width_ratios=(5, 10, 10, 1, 1), sharey=True
    )

    heatmap_config = {
        "annot_kws": {
            "size": 12,
        },
        "annot": True,
        "cbar": False,
        "cmap": "tab20c",
    }
    data = charger_usage
    seaborn.heatmap(data, ax=axes[0], **heatmap_config, mask=data == 0, fmt="g")
    axes[0].set_ylabel("Interval")
    axes[0].set_xlabel("Chargers")

    data = charge_event_usage
    #  unmask out the periods where charge_event was positive
    assert results.interval_data.evs is not None
    seaborn.heatmap(
        data,
        ax=axes[1],
        **heatmap_config,
        mask=results.interval_data.evs.charge_events == 0,
        fmt="g",
    )
    axes[1].set_xlabel("Charge Events")

    data = discharge_event_usage
    #  want to unmask out the periods where charge_event was positive
    assert results.interval_data.evs is not None
    seaborn.heatmap(
        data,
        ax=axes[2],
        **heatmap_config,
        mask=results.interval_data.evs.charge_events == 0,
        fmt="g",
    )
    axes[2].set_xlabel("Discharge Events")

    spill_charge_usage = simulation["charger-spill-electric_charge_mwh"].values.reshape(
        -1, 1
    )
    data = spill_charge_usage
    seaborn.heatmap(
        data,
        ax=axes[3],
        **(heatmap_config | {"cmap": ["white"]}),
        xticklabels=["spill"],
        fmt="g",
    )

    data = np.array(simulation["electricity_prices"]).reshape(-1, 1)
    seaborn.heatmap(
        data,
        ax=axes[4],
        **(heatmap_config | {"cmap": ["white"]}),
        xticklabels=["price"],
        fmt="g",
    )
    # for ax in axes:
    #     ax.grid(True)

    plt.tight_layout()

    if path.is_dir():
        path = path / "evs.png"
    fig.savefig(path)


def plot_chp(results: "epl.results.SimulationResult", path: pathlib.Path) -> None:
    """Plot CHP generator simulation results."""
    simulation = results.simulation
    fig, axes = plt.subplots(nrows=5, sharex=True, figsize=(12, 8))
    simulation["Index"] = np.arange(simulation.shape[0]).tolist()

    simulation["import-export-balance"] = (
        simulation["site-import_power_mwh"] - simulation["site-export_power_mwh"]
    )
    simulation.plot(
        ax=axes[0],
        x="Index",
        y="import-export-balance",
    )
    axes[0].set_ylabel("MWh")
    axes[0].set_title("Power Balance MWh (Import Positive)")
    simulation.plot(
        ax=axes[1],
        x="Index",
        y="boiler-high_temperature_generation_mwh",
    )
    axes[1].set_ylabel("MWh")
    axes[1].set_title("Boiler Heat Generation MWh")
    simulation.plot(
        ax=axes[2],
        x="Index",
        y="spill-low_temperature_load_mwh",
    )
    axes[2].set_ylabel("MWh")
    axes[2].set_title("Low Temperature Heat Dump MWh")
    simulation.plot(
        "Index",
        "electricity_prices",
        ax=axes[3],
    )
    axes[3].set_ylabel("$/MWh")
    axes[3].set_title("Electricity Prices")
    simulation.plot(
        "Index",
        "electricity_carbon_intensities",
        ax=axes[4],
    )
    axes[4].set_ylabel("tC/MWh")
    axes[4].set_title("Carbon Intensities")

    for ax in axes:
        ax.get_legend().remove()
    plt.tight_layout()

    if path.is_dir():
        path = path / "chp.png"
    fig.savefig(path)
