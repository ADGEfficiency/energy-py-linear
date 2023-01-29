"""Base AssetOneInterval class which is used as the base for all single interval energy assets models."""
import typing

import pulp
import pydantic


class AssetOneInterval(pydantic.BaseModel):
    """Generic Energy Asset for a single interval.

    Brought to you by the energy balance:
        input - output = accumulation

    Defines the quantities we care about in our energy model for one timestep:
        - electricity,
        - high temperature heat,
        - low temperature heat,
        - charge & discharge of electricity,
        - gas consumption.

    These quantities are considered as both generation and consumption (load).
    Charge and discharge are accumulation terms.
    """

    electric_generation_mwh: typing.Union[pulp.LpVariable, float] = 0
    high_temperature_generation_mwh: typing.Union[pulp.LpVariable, float] = 0
    low_temperature_generation_mwh: typing.Union[pulp.LpVariable, float] = 0
    #  add cooling generation here TODO

    electric_load_mwh: typing.Union[pulp.LpVariable, float] = 0
    high_temperature_load_mwh: typing.Union[pulp.LpVariable, float] = 0
    low_temperature_load_mwh: typing.Union[pulp.LpVariable, float] = 0
    #  add cooling load here TODO

    #  maybe should be electric charge TODO
    charge_mwh: typing.Union[pulp.LpVariable, float] = 0
    discharge_mwh: typing.Union[pulp.LpVariable, float] = 0

    gas_consumption_mwh: typing.Union[pulp.LpVariable, float] = 0

    class Config:
        arbitrary_types_allowed: bool = True
