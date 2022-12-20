import typing

import numpy as np
import pandas as pd
import pydantic

from energypylinear.defaults import defaults

floats = typing.Sequence[float]


class EVIntervalData(pydantic.BaseModel):
    charge_events: typing.Union[list[list[int]], np.ndarray]
    idx: typing.Any = []
    charge_event_mwh: typing.Union[list[int], np.ndarray]

    class Config:
        arbitrary_types_allowed = True

    @pydantic.validator("idx", always=True, pre=True)
    def setup_idx(cls, value, values):
        return np.arange(values["charge_events"].shape[0])

    @pydantic.root_validator()
    def validate_all_things(cls, values):

        #  only_positive or zero charge_event_mwh
        assert all(
            values["charge_event_mwh"] >= 0
        ), "charge_event_mwh has negative values"

        assert all(
            np.array(values["charge_events"]).sum(axis=0) > 0
        ), "sum across axis=0"

        assert (
            values["idx"].shape[0] == values["charge_events"].shape[0]
        ), "charge_event_mwh not equal to length of electricitiy prices."
        assert (
            values["charge_events"].shape[1] == values["charge_event_mwh"].shape[0]
        ), "charge_events not equal to charge_event_mwh"
        return values


class IntervalData(pydantic.BaseModel):
    electricity_prices: floats
    gas_prices: typing.Union[float, floats, None] = None
    electricity_carbon_intensities: typing.Union[float, floats, None] = None

    high_temperature_load_mwh: typing.Union[float, floats, None] = None
    low_temperature_load_mwh: typing.Union[float, floats, None] = None
    idx: typing.Any = []

    evs: typing.Union[EVIntervalData, None] = None

    @pydantic.validator("evs")
    def validate_evs(cls, evs, values, field):
        assert all(evs.idx == values["idx"])
        return evs

    @pydantic.validator(
        "gas_prices",
        "electricity_carbon_intensities",
        "high_temperature_load_mwh",
        "low_temperature_load_mwh",
        pre=True,
        always=True,
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
