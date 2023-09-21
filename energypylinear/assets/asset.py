"""Contains AssetOneInterval - used as the base for all single interval energy assets data samples."""
import abc
import typing

import pulp
import pydantic


class Asset(abc.ABC):
    """Abstract Base Class for an Asset."""

    def __init__(self):
        """Initializes the asset."""
        pass

    @abc.abstractmethod
    def __repr__(self):
        """A string representation of self."""
        pass

    @abc.abstractmethod
    def one_interval(self):
        """Generate linear program data for one interval."""
        pass

    @abc.abstractmethod
    def constrain_within_interval(self):
        """Constrain asset within an interval."""
        pass

    @abc.abstractmethod
    def constrain_after_intervals(self):
        """Constrain asset after all intervals."""
        pass

    @abc.abstractmethod
    def optimize(self):
        """Optimize the asset."""
        pass


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

    cfg: typing.Any

    electric_generation_mwh: pulp.LpVariable | float = 0
    high_temperature_generation_mwh: pulp.LpVariable | float = 0
    low_temperature_generation_mwh: pulp.LpVariable | float = 0
    electric_load_mwh: pulp.LpVariable | float = 0
    high_temperature_load_mwh: pulp.LpVariable | float = 0
    low_temperature_load_mwh: pulp.LpVariable | float = 0
    electric_charge_mwh: pulp.LpVariable | float = 0
    electric_discharge_mwh: pulp.LpVariable | float = 0
    gas_consumption_mwh: pulp.LpVariable | float = 0

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed: bool = True
