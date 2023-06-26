"""Extract results from a solved linear program to pd.DataFrame's."""
import collections

import numpy as np
import pandas as pd
import pandera as pa
import pydantic
from rich import print

import energypylinear as epl
from energypylinear.defaults import defaults
from energypylinear.flags import Flags
from energypylinear.interval_data import IntervalData
from energypylinear.optimizer import Optimizer

optimizer = Optimizer()


def check_electricity_balance(simulation: pd.DataFrame) -> pd.DataFrame:
    """Checks the electricity balance."""
    inp = (
        simulation["site-import_power_mwh"]
        + simulation["total-electric_generation_mwh"]
    )
    out = simulation["site-export_power_mwh"] + simulation["total-electric_load_mwh"]
    accumulation = simulation["total-discharge_mwh"] - simulation["total-charge_mwh"]
    balance = abs(inp + accumulation - out) < 1e-4
    data = pd.DataFrame(
        {
            "import": simulation["site-import_power_mwh"],
            "generation": simulation["total-electric_generation_mwh"],
            "export": simulation["site-export_power_mwh"],
            "load": simulation["total-electric_load_mwh"],
            "charge": simulation["total-charge_mwh"],
            "discharge": simulation["total-discharge_mwh"],
            "balance": balance,
            "loss": simulation["total-losses_mwh"],
            "spills": simulation["total-spills_mwh"],
        }
    )
    print("Electricity Balance")
    print(data)
    assert balance.all()
    return data


def check_high_temperature_heat_balance(simulation: pd.DataFrame) -> None:
    """Checks the high temperature heat balance."""
    inp = simulation[["total-high_temperature_generation_mwh"]].sum(axis=1)
    out = simulation[
        [
            "total-high_temperature_load_mwh",
            "load-high_temperature_load_mwh",
            #  valve naming - hmmmmmmmmmmmmmmmmmmmmmm
        ]
    ].sum(axis=1)
    balance = abs(inp - out) < 1e-4
    data = pd.DataFrame(
        {
            "in": inp,
            "out": out,
            "balance": balance,
        }
    )
    col = "valve-high_temperature_load_mwh"
    if col in simulation.columns:
        data["valve"] = simulation[col]
    print("High Temperature Heat Balance")
    print(data)
    assert balance.all()


def check_low_temperature_heat_balance(simulation: pd.DataFrame) -> None:
    """Checks the low temperature heat balance."""
    inp = simulation[
        [
            "total-low_temperature_generation_mwh",
        ]
    ].sum(axis=1)
    out = simulation[
        [
            "total-low_temperature_load_mwh",
            "load-low_temperature_load_mwh",
        ]
    ].sum(axis=1)
    balance = abs(inp - out) < 1e-4
    #  used for debug
    data = pd.DataFrame(
        {
            "in": inp,
            "out": out,
            "balance": balance,
        }
    )

    for name, col in [
        ("valve", "valve-low_temperature_generation_mwh"),
        ("load", "load-low_temperature_load_mwh"),
        ("assets-load", "total-low_temperature_load_mwh"),
        ("generator-load", "generator-low_temperature_load_mwh"),
        ("generator-generation", "generator-low_temperature_generation_mwh"),
    ]:
        if col in simulation.columns:
            data[name] = simulation[col]

    print("Low Temperature Heat Balance")
    print(data)
    # print(simulation[[c for c in simulation.columns if "low_temperature" in c]])
    assert balance.all()
