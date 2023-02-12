"""Models for interval data for electricity & gas prices, thermal loads and carbon intensities."""
import typing

import numpy as np
import pandas as pd
import pydantic

import energypylinear as epl
from energypylinear.defaults import defaults

floats = typing.Union[float, np.ndarray, typing.Sequence[float], list[float]]


class EVIntervalData(pydantic.BaseModel):
    """
    Interval data for electric vehicle (EV) simulation.

    Attributes:
        charge_events: a list or numpy array of charge events for each time step
        idx: index of the time steps (default is an empty list)
        charge_event_mwh: total energy consumption in each time step

    Methods:
        setup_idx: sets up the index of the time steps based on the shape of `charge_events`
        validate_all_things: validates the input data
    """

    charge_events: typing.Union[list[list[int]], np.ndarray]
    idx: typing.Any = []
    charge_event_mwh: typing.Union[list[int], np.ndarray]

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed = True

    @pydantic.validator("idx", always=True, pre=True)
    def setup_idx(cls, value: list, values: dict) -> np.ndarray:
        """Create an integer index."""
        return np.arange(values["charge_events"].shape[0])

    @pydantic.root_validator()
    def validate_all_things(cls, values: dict) -> dict:
        """Validate all input data in a single step."""
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
    """
    Interval data input into a simulation.

    `electricity_prices` must always be given - it acts as the source of truth
    for the number of intervals in the interval data.  It can be supplied
    as either a list or numpy array.

    The other data such as `electricity_carbon_intensities` or `gas_prices` can
    be given as:
        - list or numpy array (same length as `electricity_prices`),
        - single float or int to for a constant value for the each interval,
        - None to use the default value for this field (commonly 0).

    ```python
    IntervalData(
        electricity_prices=[10, 20, 30],
        gas_prices=[30, 20, 10],
    )
    IntervalData(
        electricity_prices=[10, 20, 30],
        gas_prices=20
    )
    IntervalData(
        electricity_prices=[10, 20, 30],
        gas_prices=None
    )
    ```

    Attributes:
        electricity_prices: electricity prices in each time step,
        electricity_carbon_intensities:  carbon intensity of electricity in each time step,
        gas_prices: gas prices in each time step,
        high_temperature_load_mwh: high temperature load in each time step,
        low_temperature_load_mwh: low temperature load in each time step,
        idx: integer index of the time steps,
        evs: an instance of `EVIntervalData` for simulating EVs.

    Methods:
        validate_evs: validates the EVIntervalData instance,
        validate_all_things: validates the input data,
        setup_idx: sets up the index of the time steps based on the length of `electricity_prices`.
    """

    electricity_prices: np.ndarray
    electricity_carbon_intensities: typing.Union[np.ndarray, None]
    gas_prices: np.ndarray
    high_temperature_load_mwh: np.ndarray
    low_temperature_load_mwh: np.ndarray
    idx: list[int] = []

    evs: typing.Union[EVIntervalData, None] = None

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed = True

    @pydantic.validator("evs")
    def validate_evs(
        cls, evs: "epl.interval_data.EVIntervalData", values: dict
    ) -> "epl.interval_data.EVIntervalData":
        """Validate our indexes are the same with our parent index."""
        assert all(evs.idx == values["idx"])
        return evs

    @pydantic.root_validator(pre=True)
    def validate_all_things(cls, values: dict) -> dict:
        """Validate all input data in a single step."""
        base_field = "electricity_prices"
        prices = values[base_field]

        if isinstance(prices, pd.Series):
            values[base_field] = prices.values

        if isinstance(prices, list):
            values[base_field] = np.array(prices)

        fields = [
            "gas_prices",
            "electricity_carbon_intensities",
            "high_temperature_load_mwh",
            "low_temperature_load_mwh",
        ]
        for field in fields:
            value = values.get(field)
            if isinstance(value, (float, int)):
                values[field] = np.array([value] * len(values[base_field]))

            elif value is None:
                values[field] = np.array(
                    [getattr(defaults, field)] * len(values[base_field])
                )

            else:
                assert len(value) == len(values[base_field])
                values[field] = np.array(value)

            assert values[field] is not None
            assert isinstance(values[field], np.ndarray)
            assert np.isnan(values[field]).sum() == 0

        return values

    @pydantic.validator("idx", always=True)
    def setup_idx(cls, value: list, values: dict) -> list:
        """Create an integer index."""
        return list(range(len(values["electricity_prices"])))

    def to_dataframe(self) -> pd.DataFrame:
        """Save all interval data to a pandas DataFrame."""
        data = self.dict()

        expected_len = len(data["idx"])

        df = {}
        for name, data in data.items():
            if data is not None:
                if len(data) == expected_len:
                    df[name] = data
        return pd.DataFrame(df)
