import pydantic


class Flags(pydantic.BaseModel):
    include_charge_discharge_binary_variables: bool = False
