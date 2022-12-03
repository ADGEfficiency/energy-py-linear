import collections
import typing

import pulp
import pydantic

import energypylinear as epl
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
    optimizer: Pulp, site: SiteConfig, i: int, freq: Freq
) -> SiteOneInterval:
    return SiteOneInterval(
        import_power_mwh=optimizer.continuous(
            f"import_power_mw-{i}", up=freq.mw_to_mwh(site.import_limit_mw)
        ),
        export_power_mwh=optimizer.continuous(
            f"export_power_mw-{i}", up=freq.mw_to_mwh(site.import_limit_mw)
        ),
        import_power_bin=optimizer.binary(f"import_power_bin-{i}"),
        export_power_bin=optimizer.binary(f"export_power_bin-{i}"),
        import_limit_mwh=freq.mw_to_mwh(site.import_limit_mw),
        export_limit_mwh=freq.mw_to_mwh(site.export_limit_mw),
    )


def constrain_site_electricity_balance(optimizer, vars):
    """
    in = out + accumulation
    import + generation = (export + load) + (charge - discharge)
    import + generation - (export + load) - (charge - discharge) = 0
    """
    assets = vars["assets"][-1]
    site = vars["sites"][-1]
    spill = vars["spills"][-1]

    optimizer.constrain(
        site.import_power_mwh
        - site.export_power_mwh
        + spill.electric_generation_mwh
        - spill.electric_load_mwh
        + optimizer.sum([a.electric_generation_mwh for a in assets])
        - optimizer.sum([a.electric_load_mwh for a in assets])
        - optimizer.sum([a.charge_mwh for a in assets])
        + optimizer.sum([a.discharge_mwh for a in assets])
        == 0
    )


def constrain_site_import_export(optimizer, vars):
    site = vars["sites"][-1]
    optimizer.constrain(
        site.import_power_mwh - site.import_limit_mwh * site.import_power_bin <= 0
    )
    optimizer.constrain(
        site.export_power_mwh - site.export_limit_mwh * site.export_power_bin <= 0
    )
    optimizer.constrain(site.import_power_bin + site.export_power_bin == 1)


def constrain_site_high_temperature_heat_balance(optimizer, vars, interval_data, i):
    """
    in = out + accumulation
    generation = load
    generation - load = 0
    """
    assets = vars["assets"][-1]
    spill = vars["spills"][-1]
    valve = vars["valves"][-1]
    optimizer.constrain(
        spill.high_temperature_generation_mwh
        - valve.high_temperature_load_mwh
        + optimizer.sum([a.high_temperature_generation_mwh for a in assets])
        - optimizer.sum([a.high_temperature_load_mwh for a in assets])
        - interval_data.high_temperature_load_mwh[i]
        == 0
    )


def constrain_site_low_temperature_heat_balance(optimizer, vars, interval_data, i):
    """
    in = out + accumulation
    generation = load
    generation - load = 0
    """
    assets = vars["assets"][-1]
    spill = vars["spills"][-1]
    valve = vars["valves"][-1]
    optimizer.constrain(
        optimizer.sum([a.low_temperature_generation_mwh for a in assets])
        + valve.low_temperature_generation_mwh
        - optimizer.sum([a.low_temperature_load_mwh for a in assets])
        - interval_data.low_temperature_load_mwh[i]
        - spill.low_temperature_load_mwh
        == 0
    )


def constrain_within_interval(optimizer, vars, interval_data, i):
    constrain_site_electricity_balance(optimizer, vars)
    constrain_site_import_export(optimizer, vars)
    constrain_site_high_temperature_heat_balance(optimizer, vars, interval_data, i)
    constrain_site_low_temperature_heat_balance(optimizer, vars, interval_data, i)
