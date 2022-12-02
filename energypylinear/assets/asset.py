import typing

import pulp
import pydantic


class Asset(pydantic.BaseModel):
    electric_generation_mwh: typing.Union[pulp.LpVariable, float] = 0
    high_temperature_generation_mwh: typing.Union[pulp.LpVariable, float] = 0
    low_temperature_generation_mwh: typing.Union[pulp.LpVariable, float] = 0
    #  add cooling generation here TODO

    electric_load_mwh: typing.Union[pulp.LpVariable, float] = 0
    high_temperature_load_mwh: typing.Union[pulp.LpVariable, float] = 0
    low_temperature_load_mwh: typing.Union[pulp.LpVariable, float] = 0
    #  add cooling load here TODO

    charge_mwh: typing.Union[pulp.LpVariable, float] = 0
    discharge_mwh: typing.Union[pulp.LpVariable, float] = 0

    class Config:
        arbitrary_types_allowed: bool = True
