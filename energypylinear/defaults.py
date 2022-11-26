import pydantic


class Defaults(pydantic.BaseModel):
    carbon_intensities: float = 0.1
    gas_prices: float = 20
    freq_mins: int = 30

    high_temperature_load_mwh: float = 0
    low_temperature_load_mwh: float = 0

    #  high temperature
    spill_boiler_efficiency_pct: float = 0.05
    spill_boiler_size_mw: float = 100000

    #  electricity
    spill_generator_efficiency_pct: float = 0.05
    spill_generator_size_mw: float = 100000


defaults = Defaults()
