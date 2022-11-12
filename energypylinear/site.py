import collections
import typing

import pulp
import pydantic


class SiteConfig(pydantic.BaseModel):
    import_limit_mw: float = 10000
    export_limit_mw: float = 10000


class SiteOneInterval(pydantic.BaseModel):
    import_power_mwh: pulp.LpVariable
    export_power_mwh: pulp.LpVariable
    import_power_bin: pulp.LpVariable
    export_power_bin: pulp.LpVariable

    import_limit_mwh: float
    export_limit_mwh: float

    class Config:
        arbitrary_types_allowed: bool = True


def site_one_interval(
    framework: Pulp, site: SiteConfig, i: int, freq: Freq
) -> SiteOneInterval:
    return SiteOneInterval(
        import_power_mwh=framework.continuous(
            f"import_power_mw-{i}", up=freq.mw_to_mwh(site.import_limit_mw)
        ),
        export_power_mwh=framework.continuous(
            f"export_power_mw-{i}", up=freq.mw_to_mwh(site.import_limit_mw)
        ),
        import_power_bin=framework.binary(f"import_power_bin-{i}"),
        export_power_bin=framework.binary(f"export_power_bin-{i}"),
        import_limit_mwh=freq.mw_to_mwh(site.import_limit_mw),
        export_limit_mwh=freq.mw_to_mwh(site.export_limit_mw),
    )


def constrain_site_electricity_balance(framework, site, assets):
    """
    in = out + accumulation
    import + generation = (export + load) + (charge - discharge)
    import + generation - (export + load) - (charge - discharge) = 0

    should losses occur here?
    """
    assets = assets["assets"][-1]
    framework.constrain(
        site.import_power_mwh
        + framework.sum([a.generation_mwh for a in assets])
        - (site.export_power_mwh + framework.sum([a.load_mwh for a in assets]))
        - (
            framework.sum([a.charge_mwh for a in assets])
            - framework.sum([a.discharge_mwh for a in assets])
        )
        == 0
    )


def constrain_site_import_export(framework, site):
    framework.constrain(
        site.import_power_mwh - site.import_limit_mwh * site.import_power_bin <= 0
    )
    framework.constrain(
        site.export_power_mwh - site.export_limit_mwh * site.export_power_bin <= 0
    )
    framework.constrain(site.import_power_bin + site.export_power_bin == 1)


class Site:
    def __init__(self, asset_configs: list):
        self.cfg = SiteConfig()
        self.asset_configs = asset_configs
        self.battery_configs = [
            cfg for cfg in self.asset_configs if isinstance(cfg, BatteryConfig)
        ]

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

        assets = collections.defaultdict(list)
        for i in interval_data.idx:

            site = site_one_interval(self.framework, self.site_cfg, i, freq)
            assets["sites"].append(site)

            #  battery variables
            assets["batteries"].append(
                battery_one_interval(self.framework, cfg, i, freq)
                for cfg in self.battery_configs
            )

            #  generator valiables
            assets["generators"].append(
                generator_one_interval(self.framework, cfg, i, freq)
                for cfg in self.generator_configs
            )

            #  site constraints

            battery.constraints_within_interval(framework, assets)

            #  generation constraints

        battery.constraints_after_intervals(framework, assets)
