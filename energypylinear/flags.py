import pydantic


class Flags(pydantic.BaseModel):
    include_charge_discharge_binary_variables: bool = False
    fail_on_spill_asset_use: bool = False
