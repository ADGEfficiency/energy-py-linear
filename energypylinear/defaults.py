import pydantic


class Defaults(pydantic.BaseModel):
    carbon_intensities: float = 0.1
    gas_prices: float = 20
    freq_mins: int = 30

    high_temperature_load_mwh: float = 0
    low_temperature_load_mwh: float = 0

    spill_objective_penalty: float = 1e17

    default_boiler_efficiency_pct: float = 0.8


defaults = Defaults()
