import typing

import pydantic

from energypylinear.battery import Asset


class GeneratorConfig(Asset):
    electric_efficiency_pct: float
    high_temperature_efficiency_pct: float
    low_temperature_efficiency_pct: float
    cooling_efficiency_pct: float


class GeneratorOneInterval(Asset):
    pass


class Generator:
    def __init__(
        self,
        electric_efficiency_pct: float,
        high_temperature_efficiency_pct: float,
        low_temperature_efficiency_pct: float,
        cooling_efficiency_pct: float,
    ):
        self.cfg = GeneratorConfig(
            electric_efficiency_pct=electric_efficiency_pct,
            high_temperature_efficiency_pct=high_temperature_efficiency_pct,
            low_temperature_efficiency_pct=low_temperature_efficiency_pct,
            cooling_efficiency_pct=cooling_efficiency_pct,
        )

    def optimize(
        self,
        freq_mins: int,
        prices: typing.Optional[list[float]],
        forecasts: typing.Optional[list[float]] = None,
        carbon_intensities: typing.Optional[list[float]] = None,
        objective: typing.Literal["price", "forecast", "carbon"] = "price",
    ):
        freq = Freq(freq_mins)
        interval_data = IntervalData(
            prices=prices, forecasts=forecasts, carbon_intensities=carbon_intensities
        )
        self.site_cfg = SiteConfig()
