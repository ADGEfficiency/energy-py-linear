import collections
import typing

import pulp
import pydantic

from energypylinear.frameworks import Pulp
from energypylinear.freq import Freq


class Asset(pydantic.BaseModel):
    generation_mwh: typing.Union[pulp.LpVariable, float] = 0
    load_mwh: typing.Union[pulp.LpVariable, float] = 0
    charge_mwh: typing.Union[pulp.LpVariable, float] = 0
    discharge_mwh: typing.Union[pulp.LpVariable, float] = 0

    class Config:
        arbitrary_types_allowed: bool = True


class BatteryConfig(pydantic.BaseModel):
    power_mw: float
    capacity_mwh: float
    efficiency_pct: float
    initial_charge_mwh: float = 0
    final_charge_mwh: float = 0


class BatteryModel(Asset):
    charge_mwh: pulp.LpVariable
    discharge_mwh: pulp.LpVariable
    losses_mwh: pulp.LpVariable
    initial_charge_mwh: pulp.LpVariable
    final_charge_mwh: pulp.LpVariable
    efficiency_pct: float


def battery_one_interval(
    framework: Pulp, cfg: BatteryConfig, i: int, freq: Freq
) -> BatteryModel:
    return BatteryModel(
        charge_mwh=framework.continuous(
            f"charge_mwh-{i}", up=freq.mw_to_mwh(cfg.power_mw)
        ),
        discharge_mwh=framework.continuous(
            f"discharge_mwh-{i}", up=freq.mw_to_mwh(cfg.power_mw)
        ),
        losses_mwh=framework.continuous(f"losses_mwh-{i}"),
        initial_charge_mwh=framework.continuous(f"initial_charge_mwh-{i}"),
        final_charge_mwh=framework.continuous(f"final_charge_mwh-{i}"),
        efficiency_pct=cfg.efficiency_pct,
    )


def constrain_battery_electricity_balance(
    framework: Pulp, assets: collections.defaultdict
) -> None:
    for battery in assets["batteries"][-1]:
        framework.constrain(
            battery.initial_charge_mwh
            + battery.charge_mwh
            - battery.discharge_mwh
            - battery.losses_mwh
            == battery.final_charge_mwh
        )
        framework.constrain(
            battery.losses_mwh == battery.charge_mwh * (1 - battery.efficiency_pct)
        )


def constrain_connection_batteries_between_intervals(
    framework: Pulp, assets: collections.defaultdict
) -> None:
    batteries = assets["batteries"]

    #  if in first interval, do nothing
    #  could also do something based on `i` here...
    if len(batteries) < 2:
        return None

    else:
        old = batteries[-2]
        new = batteries[-1]
        for alt, neu in zip(old, new, strict=True):
            framework.constrain(alt.final_charge_mwh == neu.initial_charge_mwh)


def constrain_initial_final_charge(
    framework: Pulp, assets: collections.defaultdict, battery_cfgs: list[BatteryConfig]
) -> None:

    batteries = assets["batteries"]
    first = batteries[0]
    for battery, cfg in zip(first, battery_cfgs, strict=True):
        framework.constrain(battery.initial_charge_mwh == cfg.initial_charge_mwh)

    last = batteries[-1]
    for battery, cfg in zip(last, battery_cfgs, strict=True):
        framework.constrain(battery.final_charge_mwh == cfg.final_charge_mwh)


def constrain_within_interval(framework, assets):
    constrain_battery_electricity_balance(framework, assets)
    constrain_connection_batteries_between_intervals(framework, assets)


def constrain_after_intervals(framework, assets, configs):
    constrain_initial_final_charge(framework, assets, configs)
