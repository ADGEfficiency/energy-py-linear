"""Contains AssetOneInterval - used as the base for all single interval energy assets data samples."""
import abc
import typing

import pulp
import pydantic

import energypylinear as epl


class Asset(abc.ABC):
    """Abstract Base Class for an Asset."""

    @abc.abstractmethod
    def __init__(self) -> None:
        """Initializes the asset."""
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        """A string representation of self."""
        pass

    @abc.abstractmethod
    def one_interval(
        self, optimizer: "epl.Optimizer", i: int, freq: "epl.Freq", flags: "epl.Flags"
    ) -> typing.Any:
        """Generate linear program data for one interval."""
        pass

    @abc.abstractmethod
    def constrain_within_interval(
        self,
        optimizer: "epl.Optimizer",
        ivars: "epl.IntervalVars",
        i: int,
        freq: "epl.Freq",
        flags: "epl.Flags",
    ) -> None:
        """Constrain asset within an interval."""
        pass

    @abc.abstractmethod
    def constrain_after_intervals(
        self, optimizer: "epl.Optimizer", ivars: "epl.IntervalVars"
    ) -> None:
        """Constrain asset after all intervals."""
        pass


#  TODO - maybe have a separate OptimizeableAsset
#     @abc.abstractmethod
#     def optimize(self) -> typing.Any | None:
#         """Optimize the asset."""
#         pass


class AssetOneInterval(pydantic.BaseModel):
    """Generic energy asset that contains data for a single interval.

    Brought to you by the energy balance:
        input - output = accumulation

    Defines the quantities we care about in our energy model for one timestep:
        - electricity,
        - high temperature heat,
        - low temperature heat,
        - charge & discharge of electricity,
        - gas consumption.

    These quantities are considered as both generation and consumption (load).

    Charge and discharge are handled as explicit accumulation terms.
    """

    cfg: typing.Any = None

    electric_generation_mwh: pulp.LpVariable | float = 0
    high_temperature_generation_mwh: pulp.LpVariable | float = 0
    low_temperature_generation_mwh: pulp.LpVariable | float = 0
    electric_load_mwh: pulp.LpVariable | float = 0
    high_temperature_load_mwh: pulp.LpVariable | float = 0
    low_temperature_load_mwh: pulp.LpVariable | float = 0
    electric_charge_mwh: pulp.LpVariable | float = 0
    electric_discharge_mwh: pulp.LpVariable | float = 0
    gas_consumption_mwh: pulp.LpVariable | float = 0

    binary: pulp.LpVariable | int = 0
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
