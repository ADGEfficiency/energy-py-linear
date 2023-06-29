"""Toggles to change simulation behaviour."""
import pydantic


class Flags(pydantic.BaseModel):
    """Toggles to change simulation behaviour."""

    include_charge_discharge_binary_variables: bool = False
    fail_on_spill_asset_use: bool = False

    #  whether to throw a Python error on an infeasible linear program
    allow_infeasible: bool = False
