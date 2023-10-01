"""Schema for simulation results."""
import pandera as pa

from energypylinear.assets.asset import AssetOneInterval
from energypylinear.defaults import defaults
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

for qu in [q for q in AssetOneInterval.__fields__ if (q != "cfg") and (q != "binary")]:
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
