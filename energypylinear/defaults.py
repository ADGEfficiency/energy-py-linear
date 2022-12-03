import pydantic


class Defaults(pydantic.BaseModel):
    carbon_intensities: float = 0.1
    gas_prices: float = 20
    freq_mins: int = 60

    high_temperature_load_mwh: float = 0
    low_temperature_load_mwh: float = 0

    spill_objective_penalty: float = 1e10

    default_boiler_efficiency_pct: float = 0.8

    gas_carbon_intensity_tc_per_mwh: float = 0.185

    spill_charge_max_mw: float = 1e10

    decimal_tolerance: int = 4


defaults = Defaults()
