import collections
import typing

import pulp
import pydantic

from energypylinear.freq import Freq
from energypylinear.optimizer import Pulp


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


def constrain_site_electricity_balance(framework, assets):
    """
    in = out + accumulation
    import + generation = (export + load) + (charge - discharge)
    import + generation - (export + load) - (charge - discharge) = 0

    TODO add a spill import + export
    """
    assets_one_interval = assets["assets"][-1]
    site_one_interval = assets["sites"][-1]
    framework.constrain(
        site_one_interval.import_power_mwh
        + framework.sum([a.electric_generation_mwh for a in assets_one_interval])
        - (
            site_one_interval.export_power_mwh
            + framework.sum([a.electric_load_mwh for a in assets_one_interval])
        )
        - (
            framework.sum([a.electric_charge_mwh for a in assets_one_interval])
            - framework.sum([a.electric_discharge_mwh for a in assets_one_interval])
        )
        == 0
    )


def constrain_site_import_export(framework, assets):
    site_one_interval = assets["sites"][-1]
    framework.constrain(
        site_one_interval.import_power_mwh
        - site_one_interval.import_limit_mwh * site_one_interval.import_power_bin
        <= 0
    )
    framework.constrain(
        site_one_interval.export_power_mwh
        - site_one_interval.export_limit_mwh * site_one_interval.export_power_bin
        <= 0
    )
    framework.constrain(
        site_one_interval.import_power_bin + site_one_interval.export_power_bin == 1
    )


def constrain_site_high_temperature_heat(optimizer, vars):
    """
    in = out + accumulation
    generation = load
    generation - load = 0
    """
    assets = vars["assets"][-1]
    optimizer.constrain(
        optimizer.sum([a.high_temperature_generation_mwh for a in assets])
        - optimizer.sum([a.high_temperature_load_mwh for a in assets])
        == 0
    )


def constrain_site_low_temperature_heat(optimizer, vars):
    """
    in = out + accumulation
    generation = load
    generation - load = 0
    """
    assets = vars["assets"][-1]
    optimizer.constrain(
        optimizer.sum([a.low_temperature_generation_mwh for a in assets])
        - optimizer.sum([a.low_high_temperature_load_mwh for a in assets])
        == 0
    )


def constrain_within_interval(framework, assets):
    constrain_site_electricity_balance(framework, assets)
    constrain_site_import_export(framework, assets)
    constrain_site_high_temperature_heat(framework, assets)
    constrain_site_low_temperature_heat(framework, assets)
    #  add cooling constraint here TODO


class Site:
    def __init__(self, asset_configs: list):
        self.cfg = SiteConfig()
        self.asset_configs = asset_configs
        self.battery_configs = [
            cfg for cfg in self.asset_configs if isinstance(cfg, BatteryConfig)
        ]
