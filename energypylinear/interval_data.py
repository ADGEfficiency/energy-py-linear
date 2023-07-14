"""Models for interval data for electricity & gas prices, thermal loads and carbon intensities."""
import collections
import typing

import numpy as np
import pandas as pd
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval
from energypylinear.assets.evs import EVsArrayOneInterval
from energypylinear.assets.site import SiteOneInterval
from energypylinear.defaults import defaults

floats = typing.Union[float, np.ndarray, typing.Sequence[float], list[float]]


class EVIntervalData(pydantic.BaseModel):
    """
    Interval data for electric vehicle (EV) simulation.

    Attributes:
        charge_events: a list or numpy array of charge events for each time step
        idx: index of the time steps (default is an empty list)
        charge_events_capacity_mwh:

    Methods:
        setup_idx: sets up the index of the time steps based on the shape of `charge_events`
        validate_all_things: validates the input data
    """

    charge_events: np.ndarray
    charge_events_capacity_mwh: list
    idx: typing.Any = []

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
        assert all(
            np.array(values["charge_events"]).sum(axis=0) > 0
        ), "sum across axis=0"

        assert (
            values["idx"].shape[0] == values["charge_events"].shape[0]
        ), "charge_event_mwh not equal to length of electricitiy prices."
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
    electricity_carbon_intensities: np.ndarray | None = None
    gas_prices: np.ndarray | None = None
    high_temperature_load_mwh: np.ndarray | None = None
    low_temperature_load_mwh: np.ndarray | None = None
    electricity_load_mwh: np.ndarray | None = None
    idx: list[int] = []

    evs: typing.Union[EVIntervalData, None] = None

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed = True

    def __str__(self) -> str:
        """A string representation of self."""
        assert isinstance(self.electricity_prices, np.ndarray)
        return f"<epl.IntervalData n: {self.electricity_prices.shape[0]} electricity_prices: {self.electricity_prices.mean():2.1f}>"

    @pydantic.validator("evs")
    def validate_evs(
        cls, evs: "epl.interval_data.EVIntervalData", values: dict
    ) -> typing.Optional["epl.interval_data.EVIntervalData"]:
        """Validate our indexes are the same with our parent index."""
        if evs:
            assert all(evs.idx == values["idx"])
            return evs
        return None

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
            "electricity_load_mwh",
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
    def setup_idx(cls, _: list, values: dict) -> list:
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


class IntervalVars:
    def __init__(self) -> None:
        #  not every lp variable - only the ones we want to iterate over
        #  in the objective functions (price, carbon etc)
        self.objective_variables: list[list[AssetOneInterval]] = []
        self.asset: collections.defaultdict = collections.defaultdict(
            lambda: {"evs_array": [], "spill_evs_array": [], "site": []}
        )

    def __repr__(self) -> str:
        return f"<epl.IntervalVars i: {len(self.objective_variables)}>"

    def append(
        self, one_interval: AssetOneInterval | SiteOneInterval | list[AssetOneInterval]
    ) -> None:
        #  some OneInterval objects are special
        #  is this case it is the Array EV data structures
        #  TODO in future don't save these separately and
        #  dynamically create as needed from the objective variables
        if isinstance(one_interval, EVsArrayOneInterval):
            if one_interval.is_spill:
                self.asset[one_interval.cfg.name]["spill_evs_array"].append(
                    one_interval
                )
            else:
                self.asset[one_interval.cfg.name]["evs_array"].append(one_interval)
        elif isinstance(one_interval, SiteOneInterval):
            self.asset[one_interval.cfg.name]["site"].append(one_interval)

        else:
            assert isinstance(one_interval, list)
            self.objective_variables.append(one_interval)

    def filter_evs_array(
        self, is_spill: bool, i: int, asset_name: str
    ) -> EVsArrayOneInterval:
        if is_spill:
            return self.asset[asset_name]["spill_evs_array"][i]
        else:
            return self.asset[asset_name]["evs_array"][i]

    def filter_all_evs_array(
        self, is_spill: bool, asset_name: str
    ) -> list[EVsArrayOneInterval]:
        if is_spill:
            return self.asset[asset_name]["spill_evs_array"]
        else:
            return self.asset[asset_name]["evs_array"]

    def filter_site(self, i: int, site_name: str) -> SiteOneInterval:
        return self.asset[site_name]["site"][i]

    def filter_objective_variables(
        self,
        instance_type: type[AssetOneInterval],
        i: int | None = None,
        asset_name: str | None = None,
    ) -> list[list[AssetOneInterval]]:
        #  here we return data for all intervals
        if i is None:
            pkg = []
            for i, assets_one_interval in enumerate(self.objective_variables):
                pkg.append(
                    [
                        asset
                        for asset in assets_one_interval
                        if isinstance(asset, instance_type)
                        and (
                            asset.cfg.name == asset_name
                            if asset_name is not None
                            else True
                        )
                    ]
                )
            return pkg

        #  here we return data for one interval
        else:
            assets = self.objective_variables[i]
            return [
                [
                    asset
                    for asset in assets
                    if isinstance(asset, instance_type)
                    and (
                        asset.cfg.name == asset_name if asset_name is not None else True
                    )
                ]
            ]
