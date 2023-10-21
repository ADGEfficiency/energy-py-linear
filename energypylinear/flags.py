"""Toggles to change simulation behaviour."""
import pydantic


class Flags(pydantic.BaseModel):
    """Toggles to change simulation behaviour."""

    #  general
    fail_on_spill_asset_use: bool = False
    allow_infeasible: bool = False

    #  evs
    allow_evs_discharge: bool = False
    limit_charge_variables_to_valid_events: bool = False
