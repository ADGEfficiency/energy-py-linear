import collections
import typing

import pulp
import pydantic

import energypylinear as epl
from energypylinear import battery, objectives, site
from energypylinear.assets.asset import Asset
from energypylinear.defaults import defaults
from energypylinear.freq import Freq
from energypylinear.optimizer import Optimizer


class BatteryConfig(pydantic.BaseModel):
    name: str
    power_mw: float
    capacity_mwh: float
    efficiency_pct: float
    initial_charge_mwh: float = 0
    final_charge_mwh: float = 0


class BatteryOneInterval(Asset):
    cfg: BatteryConfig
    charge_mwh: pulp.LpVariable
    discharge_mwh: pulp.LpVariable
    losses_mwh: pulp.LpVariable
    initial_charge_mwh: pulp.LpVariable
    final_charge_mwh: pulp.LpVariable
    efficiency_pct: float


def battery_one_interval(
    optimizer: Optimizer, cfg: BatteryConfig, i: int, freq: Freq
) -> BatteryOneInterval:
    return BatteryOneInterval(
        cfg=cfg,
        charge_mwh=optimizer.continuous(
            f"charge_mwh-{i}", up=freq.mw_to_mwh(cfg.power_mw)
        ),
        discharge_mwh=optimizer.continuous(
            f"discharge_mwh-{i}", up=freq.mw_to_mwh(cfg.power_mw)
        ),
        losses_mwh=optimizer.continuous(f"losses_mwh-{i}"),
        initial_charge_mwh=optimizer.continuous(f"initial_charge_mwh-{i}"),
        final_charge_mwh=optimizer.continuous(f"final_charge_mwh-{i}"),
        efficiency_pct=cfg.efficiency_pct,
    )


def constrain_battery_electricity_balance(
    optimizer: Optimizer, vars: collections.defaultdict
) -> None:
    for battery in vars["batteries"][-1]:
        optimizer.constrain(
            battery.initial_charge_mwh
            + battery.charge_mwh
            - battery.discharge_mwh
            - battery.losses_mwh
            == battery.final_charge_mwh
        )
        optimizer.constrain(
            battery.losses_mwh == battery.charge_mwh * (1 - battery.efficiency_pct)
        )


def constrain_connection_batteries_between_intervals(
    optimizer: Optimizer, vars: collections.defaultdict
) -> None:
    batteries = vars["batteries"]

    #  if in first interval, do nothing
    #  could also do something based on `i` here...
    if len(batteries) < 2:
        return None

    else:
        old = batteries[-2]
        new = batteries[-1]
        for alt, neu in zip(old, new, strict=True):
            optimizer.constrain(alt.final_charge_mwh == neu.initial_charge_mwh)


def constrain_initial_final_charge(
    optimizer: Optimizer,
    vars: collections.defaultdict,
    battery_cfgs: list[BatteryConfig],
) -> None:

    batteries = vars["batteries"]
    first = batteries[0]
    for battery, cfg in zip(first, battery_cfgs, strict=True):
        optimizer.constrain(battery.initial_charge_mwh == cfg.initial_charge_mwh)

    last = batteries[-1]
    for battery, cfg in zip(last, battery_cfgs, strict=True):
        optimizer.constrain(battery.final_charge_mwh == cfg.final_charge_mwh)


def constrain_within_interval(optimizer, vars):
    constrain_battery_electricity_balance(optimizer, vars)
    constrain_connection_batteries_between_intervals(optimizer, vars)


def constrain_after_intervals(optimizer, vars, configs):
    constrain_initial_final_charge(optimizer, vars, configs)


class Battery:
    def __init__(
        self,
        power_mw: float,
        capacity_mwh: float,
        efficiency: float = 0.9,
        battery_name: str = "battery-alpha",
    ):
        self.cfg = battery.BatteryConfig(
            name=battery_name,
            power_mw=power_mw,
            capacity_mwh=capacity_mwh,
            efficiency_pct=efficiency,
        )
        self.optimizer = Optimizer()

    def optimize(
        self,
        electricity_prices,
        gas_prices=None,
        carbon_intensities=None,
        high_temperature_load_mwh=None,
        low_temperature_load_mwh=None,
        freq_mins: int = defaults.freq_mins,
        initial_charge_mwh: float = 0.0,
        final_charge_mwh: typing.Union[float, None] = None,
    ):
        freq = Freq(freq_mins)
        interval_data = epl.data.IntervalData(
            electricity_prices=electricity_prices,
            gas_prices=gas_prices,
            electricity_carbon_intensities=carbon_intensities,
            high_temperature_load_mwh=high_temperature_load_mwh,
            low_temperature_load_mwh=low_temperature_load_mwh,
        )
        self.site_cfg = epl.site.SiteConfig()
        self.spill_cfg = epl.spill.SpillConfig()
        self.valve_cfg = epl.valve.ValveConfig(name="valve-alpha")

        self.cfg.initial_charge_mwh = initial_charge_mwh
        self.cfg.final_charge_mwh = (
            initial_charge_mwh if final_charge_mwh is None else final_charge_mwh
        )

        vars = collections.defaultdict(list)
        for i in interval_data.idx:

            vars["sites"].append(
                site.site_one_interval(self.optimizer, self.site_cfg, i, freq)
            )
            vars["spills"].append(
                epl.spill.spill_one_interval(self.optimizer, self.spill_cfg, i, freq)
            )
            vars["valves"].append(
                epl.valve.valve_one_interval(self.optimizer, self.valve_cfg, i, freq)
            )

            batteries = [
                battery.battery_one_interval(self.optimizer, self.cfg, i, freq)
            ]
            #  do I need both??? can I just get away with `assets`
            vars["batteries"].append(batteries)
            vars["assets"].append(batteries)

            site.constrain_within_interval(self.optimizer, vars, interval_data, i)
            battery.constrain_within_interval(self.optimizer, vars)

        battery.constrain_after_intervals(self.optimizer, vars, [self.cfg])

        assert (
            len(interval_data.idx)
            == len(vars["assets"])
            == len(vars["batteries"])
            == len(vars["sites"])
        )
        self.optimizer.objective(
            objectives.price_objective(self.optimizer, vars, interval_data)
        )
        status = self.optimizer.solve()
        return epl.data.extract_results(interval_data, vars)