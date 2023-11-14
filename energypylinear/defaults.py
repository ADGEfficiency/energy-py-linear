"""Constant values for the library."""
import pydantic


class Defaults(pydantic.BaseModel):
    """Collection of constant values."""

    electricity_prices: float = 100.0
    export_electricity_prices: float = 100.0
    electricity_carbon_intensities: float = 0.1
    gas_prices: float = 20
    freq_mins: int = 60

    high_temperature_load_mwh: float = 0
    low_temperature_load_mwh: float = 0
    low_temperature_generation_mwh: float = 0
    electric_load_mwh: float = 0

    #   setting this too high will break the evs... was as 1e10
    spill_objective_penalty: float = 1e11

    boiler_efficiency_pct: float = 0.8
    boiler_high_temperature_generation_max_mw: float = 10000.0

    gas_carbon_intensity: float = 0.185

    spill_charge_max_mw: float = 1e4
    spill_capacity_mwh: float = 1e4

    decimal_tolerance: int = 4

    #  used for < 0 stuff
    epsilon: float = -1e-4

    log_level: int = 2


defaults = Defaults()
