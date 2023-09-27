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


def plot_battery(simulation: "epl.SimulationResult", path: pathlib.Path | str) -> None:
    """Plot battery simulation results."""
    path = pathlib.Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(nrows=5, sharex=True, figsize=(12, 8))

    results = simulation.results

    results["Index"] = np.arange(results.shape[0]).tolist()

    results["import-export-balance"] = (
        results["site-import_power_mwh"] - results["site-export_power_mwh"]
    )
    results.plot(
        ax=axes[0],
        x="Index",
        y="import-export-balance",
    )
    axes[0].set_ylim()
    axes[0].set_title("Power Balance MWh (Import Positive)")
    axes[0].set_ylabel("MWh")

    results["net-battery-charge"] = (
        results[find_column(results, "battery-", "-electric_charge_mwh")]
        - results[find_column(results, "battery-", "-electric_discharge_mwh")]
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
        "battery-electric_final_charge_mwh",
        ax=axes[2],
        kind="bar",
    )
    axes[2].set_title("Battery SOC MWh (End of Interval)")
    axes[2].set_ylabel("MWh")

    results.plot(
        "Index",
        f"{simulation.site.cfg.name}-electricity_prices",
        ax=axes[3],
    )
    axes[3].set_title("Electricity Prices")
    axes[3].set_ylabel("$/MWh")
    results.plot(
        "Index",
        f"{simulation.site.cfg.name}-electricity_carbon_intensities",
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


def plot_evs(
    simulation: "epl.SimulationResult", path: pathlib.Path, asset_name: str
) -> None:
    """Plot electric vehicle simulation results."""
    results = simulation.results

    charger_charge = results[
        [
            c
            for c in results.columns
            if "charger-" in c and c.endswith("-electric_charge_mwh")
        ]
    ].values
    charger_discharge = results[
        [
            c
            for c in results.columns
            if "charger-" in c and c.endswith("-electric_discharge_mwh")
        ]
    ].values
    charger_usage = charger_charge - charger_discharge

    charge_event_usage = results[
        [
            c
            for c in results.columns
            if "charge-event-" in c and c.endswith("electric_charge_mwh")
        ]
    ].values
    discharge_event_usage = results[
        [
            c
            for c in results.columns
            if "charge-event-" in c and c.endswith("electric_discharge_mwh")
        ]
    ].values
    charge_event_usage = charge_event_usage - discharge_event_usage

    charge_event_initial_soc = results[
        [
            c
            for c in results.columns
            if "charge-event-" in c and c.endswith("initial_soc_mwh")
        ]
    ].values
    charge_event_final_soc = results[
        [
            c
            for c in results.columns
            if "charge-event-" in c and c.endswith("final_soc_mwh")
        ]
    ].values

    fig, axes = plt.subplots(
        ncols=4, figsize=(14, 6), width_ratios=(5, 10, 1, 1), sharey=True
    )

    data = [
        *charger_usage.flatten(),
        *charge_event_usage.flatten(),
    ]
    global_vmin = np.min(data)
    global_vmax = np.max(data)

    heatmap_config = {
        "annot_kws": {
            "size": 12,
        },
        "annot": True,
        "cbar": False,
        "cmap": "coolwarm",
        "vmin": global_vmin,
        "vmax": global_vmax,
    }
    data = charger_usage
    seaborn.heatmap(data, ax=axes[0], **heatmap_config, mask=data == 0, fmt="g")
    axes[0].set_ylabel("Interval")
    axes[0].set_xlabel("Chargers")

    data = charge_event_usage
    charge_event_heatmap_config = heatmap_config.copy()
    result_array = np.empty_like(charge_event_initial_soc, dtype=object)

    for i, (a, b, c) in enumerate(
        zip(charge_event_initial_soc.T, charge_event_usage.T, charge_event_final_soc.T)
    ):
        for j, values in enumerate(zip(a, b, c)):
            result_array[
                j, i
            ] = f"initial: {values[0]:3.1f}\ncharge: {values[1]:3.1f}\nfinal: {values[2]:3.1f}"

    print(result_array)
    charge_event_heatmap_config["annot"] = result_array

    assets = [a for a in simulation.assets if a.cfg.name == asset_name]
    assert len(assets) == 1
    asset = assets[0]
    seaborn.heatmap(
        data,
        ax=axes[1],
        **charge_event_heatmap_config,
        #  unmask out the periods where charge_event was positive
        # mask=results.interval_data.evs.charge_events == 0,
        mask=asset.cfg.charge_events == 0,
        fmt="",
    )
    axes[1].set_xlabel("Charge Events Net Charge (Discharge is Negative)")

    spill_charge_usage = results[
        f"{asset.cfg.name}-charger-spill-evs-electric_charge_mwh"
    ].values.reshape(-1, 1)
    data = spill_charge_usage
    seaborn.heatmap(
        data,
        ax=axes[2],
        **(heatmap_config | {"cmap": ["white"]}),
        xticklabels=["spill"],
        fmt="g",
    )

    seaborn.heatmap(
        np.array(results[f"{simulation.site.cfg.name}-electricity_prices"]).reshape(
            -1, 1
        ),
        ax=axes[3],
        **(heatmap_config | {"cmap": ["white"]}),
        xticklabels=["price"],
        fmt="g",
    )
    # for ax in axes:
    #     ax.grid(True)
    # for ax in axes:
    #     for spine in ax.spines.values():
    #         spine.set_edgecolor('gray')
    #         spine.set_linewidth(2)
    from matplotlib.patches import Rectangle

    for ax in axes:
        border = Rectangle(
            (0, 0),
            1,
            1,
            edgecolor="gray",
            facecolor="none",
            transform=ax.transAxes,
            figure=ax.figure,
            linewidth=2,
        )
        ax.add_patch(border)

    plt.tight_layout()

    if path.is_dir():
        path = path / "evs.png"
    fig.savefig(path)


def plot_chp(simulation: "epl.SimulationResult", path: pathlib.Path) -> None:
    """Plot CHP generator simulation results."""
    results = simulation.results
    fig, axes = plt.subplots(nrows=5, sharex=True, figsize=(12, 8))
    results["Index"] = np.arange(results.shape[0]).tolist()

    results["import-export-balance"] = (
        results["site-import_power_mwh"] - results["site-export_power_mwh"]
    )
    results.plot(
        ax=axes[0],
        x="Index",
        y="import-export-balance",
    )
    axes[0].set_ylabel("MWh")
    axes[0].set_title("Power Balance MWh (Import Positive)")
    results.plot(
        ax=axes[1],
        x="Index",
        y="boiler-high_temperature_generation_mwh",
    )
    axes[1].set_ylabel("MWh")
    axes[1].set_title("Boiler Heat Generation MWh")
    results.plot(
        ax=axes[2],
        x="Index",
        y="spill-low_temperature_load_mwh",
    )
    axes[2].set_ylabel("MWh")
    axes[2].set_title("Low Temperature Heat Dump MWh")
    results.plot(
        "Index",
        f"{simulation.site.cfg.name}-electricity_prices",
        ax=axes[3],
    )
    axes[3].set_ylabel("$/MWh")
    axes[3].set_title("Electricity Prices")
    results.plot(
        "Index",
        f"{simulation.site.cfg.name}-electricity_carbon_intensities",
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


def plot_heat_pump(
    simulation: "epl.SimulationResult", path: pathlib.Path | str, asset_name: str
) -> None:
    """Plot Heat Pump simulation results."""
    path = pathlib.Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(12, 12))
    results = simulation.results
    results["index"] = np.arange(results.shape[0]).tolist()

    for col in [
        f"{simulation.site.cfg.name}-high_temperature_load_mwh",
        f"{asset_name}-high_temperature_generation_mwh",
    ]:
        results.plot(ax=axes[0], x="index", y=col)
    axes[0].set_ylabel("MWh")
    axes[0].set_title("Heat Pump")

    box = axes[0].get_position()
    axes[0].set_position([box.x0, box.y0 + box.y0 * 0.1, box.width, box.height * 0.8])
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2)

    out_cols = [
        "spill-low_temperature_load_mwh",
        f"{simulation.site.cfg.name}-low_temperature_load_mwh",
        f"{asset_name}-low_temperature_load_mwh",
    ]
    neg_bottom = np.zeros_like(results["index"]).astype(float)
    width = 0.3
    adj = 0.15
    for col in out_cols:
        results[col] = results[col].round(4)
        axes[1].bar(
            results["index"] - adj,
            results[col],
            width,
            label=col,
            bottom=neg_bottom,
        )
        neg_bottom += results[col]

    bottom = np.zeros_like(results["index"]).astype(float)
    in_cols = [
        f"{simulation.site.cfg.name}-low_temperature_generation_mwh",
        "spill-low_temperature_generation_mwh",
        "valve-low_temperature_generation_mwh",
    ]
    for col in in_cols:
        results[col] = results[col].round(4)
        axes[1].bar(
            results["index"] + adj, results[col], width, label=col, bottom=bottom
        )
        bottom += results[col]

    axes[1].set_ylabel("MWh")
    axes[1].set_title("Low Temperature Heat Balance")

    box = axes[1].get_position()
    axes[1].set_position([box.x0, box.y0 + box.y0 * 0.2, box.width, box.height * 0.8])
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=2)

    for col in [
        f"{simulation.site.cfg.name}-electricity_prices",
        f"{simulation.site.cfg.name}-electricity_carbon_intensities",
    ]:
        results.plot(ax=axes[2], x="index", y=col)

    axes[2].set_ylabel("$/MWh or tC/MWh")
    axes[2].set_title("Electricity Prices & Carbon Intensities")

    if path.is_dir():
        path = path / "heat-pump.png"
    fig.savefig(path)
