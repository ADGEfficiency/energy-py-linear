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


schema = {
    "site-import_power_mwh": pa.Column(
        pa.Float,
        checks=[pa.Check.ge(defaults.epsilon)],
        title="Site Import Power MWh",
        coerce=True,
    ),
    "site-export_power_mwh": pa.Column(
        pa.Float,
        checks=[pa.Check.ge(defaults.epsilon)],
        title="Site Export Power MWh",
        coerce=True,
    ),
}
#  maybe could get this from epl.assets.AssetOneInterval ?
quantities = [
    "electric_generation_mwh",
    "electric_load_mwh",
    "high_temperature_generation_mwh",
    "low_temperature_generation_mwh",
    "high_temperature_load_mwh",
    "low_temperature_load_mwh",
    "gas_consumption_mwh",
    "electric_charge_mwh",
    "electric_discharge_mwh",
]
for qu in quantities:
    schema[rf"\w+-{qu}"] = pa.Column(
        pa.Float, checks=[pa.Check.ge(defaults.epsilon)], coerce=True, regex=True
    )
    schema[f"total-{qu}"] = pa.Column(
        pa.Float,
        checks=[pa.Check.ge(defaults.epsilon)],
        coerce=True,
        regex=True,
        required=True,
    )
simulation_schema = pa.DataFrameSchema(schema)


spill_quantities = [
    "electric_generation_mwh",
    "electric_load_mwh",
    "high_temperature_generation_mwh",
    "low_temperature_generation_mwh",
    "high_temperature_load_mwh",
    "low_temperature_load_mwh",
    "gas_consumption_mwh",
]
