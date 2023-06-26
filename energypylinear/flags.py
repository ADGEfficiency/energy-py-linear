"""Toggles to change simulation behaviour."""
import pydantic


class Flags(pydantic.BaseModel):
    """Toggles to change simulation behaviour."""

    #  general
    fail_on_spill_asset_use: bool = False

    #  battery
    include_charge_discharge_binary_variables: bool = False

    #  evs
    allow_evs_discharge: bool = False

    #  whether to throw a Python error on an infeasible linear program
    allow_infeasible: bool = False
