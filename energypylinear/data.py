import typing

import numpy as np
import pydantic

from energypylinear.defaults import defaults

floats = typing.Union[np.ndarray, typing.Sequence[float]]


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
    electricity_carbon_intensities: typing.Union[
        float, floats, None
    ] = defaults.electricity_carbon_intensities
    electricity_prices: typing.Union[float, floats, None] = defaults.electricity_prices
    gas_prices: typing.Union[float, floats, None] = None

    high_temperature_load_mwh: typing.Union[float, floats, None] = None
    low_temperature_load_mwh: typing.Union[float, floats, None] = None
    idx: typing.Any = []

    evs: typing.Union[EVIntervalData, None] = None

    class Config:
        arbitrary_types_allowed = True

    @pydantic.validator("evs")
    def validate_evs(cls, evs, values):
        assert all(evs.idx == values["idx"])
        return evs

    @pydantic.root_validator(pre=True)
    def validate_all_things(cls, values):

        if "electricity_prices" in values.keys():
            base_field = "electricity_prices"
        else:
            base_field = "electricity_carbon_intensities"

        fields = [
            "electricity_prices",
            "gas_prices",
            "electricity_carbon_intensities",
            "high_temperature_load_mwh",
            "low_temperature_load_mwh",
        ]
        fields.remove(base_field)
        for field in fields:
            value = values.get(field)
            if isinstance(value, (float, int)):
                values[field] = [value] * len(values[base_field])

            elif value is None:
                values[field] = [getattr(defaults, field)] * len(values[base_field])

            else:
                assert len(value) == len(values[base_field])
                values[field] = value
        return values

    @pydantic.validator("idx", always=True)
    def setup_idx(cls, value, values):
        return list(range(len(values["electricity_prices"])))
