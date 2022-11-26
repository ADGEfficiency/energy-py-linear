import collections
import typing

import pulp

floats = typing.Sequence[float]
import pydantic

from energypylinear.defaults import defaults


class IntervalData(pydantic.BaseModel):
    electricity_prices: floats
    gas_prices: typing.Union[float, floats, None] = None
    carbon_intensities: typing.Union[float, floats, None] = None

    high_temperature_load_mwh: typing.Union[float, floats, None] = None
    low_temperature_load_mwh: typing.Union[float, floats, None] = None
    idx: typing.Any = []

    @pydantic.validator(
        "gas_prices",
        "carbon_intensities",
        "high_temperature_load_mwh",
        "low_temperature_load_mwh",
        pre=True,
    )
    def validate_float(cls, value, values, field):
        if isinstance(value, (float, int)):
            return [value] * len(values["electricity_prices"])

        elif value is None:
            return [getattr(defaults, field.name)] * len(values["electricity_prices"])

        else:
            assert len(value) == len(values["electricity_prices"])
            return value

    @pydantic.validator("idx", always=True)
    def setup_idx(cls, value, values):
        return list(range(len(values["electricity_prices"])))
