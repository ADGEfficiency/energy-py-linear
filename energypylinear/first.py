import collections
import typing

import pulp
import pydantic

from energypylinear import battery, site
from energypylinear.frameworks import Pulp
from energypylinear.freq import Freq


class IntervalData:
    default_price: float = 100
    default_carbon_intensity: float = 0.5

    def __init__(
        self,
        prices: typing.Optional[typing.Union[float, list[float]]] = None,
        forecasts: typing.Optional[list[float]] = None,
        carbon_intensities: typing.Optional[typing.Union[float, list[float]]] = None,
    ):
        assert any((prices, carbon_intensities))

        if prices is None:
            assert isinstance(carbon_intensities, list)
            prices = [self.default_price for _ in range(len(carbon_intensities))]

        if carbon_intensities is None:
            assert isinstance(prices, list)
            carbon_intensities = [
                self.default_carbon_intensity for _ in range(len(prices))
            ]

        if forecasts is None:
            assert isinstance(prices, list)
            forecasts = prices

        assert isinstance(prices, list)
        assert isinstance(carbon_intensities, list)
        assert isinstance(forecasts, list)
        assert len(prices) == len(forecasts) == len(carbon_intensities)

        self.prices = prices
        self.forecasts = forecasts
        self.carbon_intensities = carbon_intensities
        self.idx = range(len(self.prices))


class Battery:
    def __init__(self, power_mw: float, capacity_mwh: float, efficiency: float = 0.9):
        self.cfg = battery.BatteryConfig(
            power_mw=power_mw,
            capacity_mwh=capacity_mwh,
            efficiency_pct=efficiency,
        )
        self.framework = Pulp()

    def optimize(
        self,
        freq_mins: int,
        prices: typing.Optional[list[float]],
        forecasts: typing.Optional[list[float]] = None,
        carbon_intensities: typing.Optional[list[float]] = None,
        initial_charge_mwh: float = 0.0,
        objective: typing.Literal["price", "forecast", "carbon"] = "price",
    ):
        freq = Freq(freq_mins)
        interval_data = IntervalData(
            prices=prices, forecasts=forecasts, carbon_intensities=carbon_intensities
        )
        self.site_cfg = site.SiteConfig()
        self.cfg.initial_charge_mwh = initial_charge_mwh
        self.cfg.final_charge_mwh = initial_charge_mwh

        assets = collections.defaultdict(list)
        for i in interval_data.idx:

            #  linear program variables
            assets["sites"].append(
                site.site_one_interval(self.framework, self.site_cfg, i, freq)
            )
            batteries = [
                battery.battery_one_interval(self.framework, self.cfg, i, freq)
            ]
            assets["batteries"].append(batteries)
            assets["assets"].append(batteries)

            #  site constraints
            site.constrain_within_interval(self.framework, assets)
            battery.constrain_within_interval(self.framework, assets)

        battery.constrain_after_intervals(self.framework, assets, [self.cfg])

        assert (
            len(interval_data.idx)
            == len(assets["assets"])
            == len(assets["batteries"])
            == len(assets["sites"])
        )

        #  objective functions
        sites = assets["sites"]
        price_objective = self.framework.sum(
            sites[i].import_power_mwh * interval_data.prices[i]
            - sites[i].export_power_mwh * interval_data.prices[i]
            for i in interval_data.idx
        )
        forecast_objective = self.framework.sum(
            sites[i].import_power_mwh * interval_data.forecasts[i]
            - sites[i].export_power_mwh * interval_data.forecasts[i]
            for i in interval_data.idx
        )
        carbon_objective = self.framework.sum(
            sites[i].import_power_mwh * interval_data.carbon_intensities[i]
            - sites[i].export_power_mwh * interval_data.carbon_intensities[i]
            for i in interval_data.idx
        )

        objectives = {
            "price": price_objective,
            "forecast": forecast_objective,
            "carbon": carbon_objective,
        }

        self.framework.objective(objectives[objective])
        status = self.framework.solve()
        print(status)


asset = Battery(power_mw=2, capacity_mwh=4, efficiency=0.95)
results = asset.optimize(prices=[10, 50, 10, 50, 10], freq_mins=30)
