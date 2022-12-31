import collections

import pandas as pd
import pulp
import pydantic

import energypylinear as epl
from energypylinear import objectives
from energypylinear.assets.asset import Asset
from energypylinear.defaults import defaults
from energypylinear.freq import Freq
from energypylinear.optimizer import Optimizer


class BoilerConfig(Asset):
    name: str
    high_temperature_generation_max_mw: float = 0
    high_temperature_generation_min_mw: float = 0
    high_temperature_efficiency_pct: float = 0


class BoilerOneInterval(Asset):
    cfg: BoilerConfig
    high_temperature_generation_mwh: pulp.LpVariable
    gas_consumption_mwh: pulp.LpVariable
    binary: pulp.LpVariable


class GeneratorConfig(Asset):
    name: str
    electric_power_max_mw: float = 0
    electric_power_min_mw: float = 0

    electric_efficiency_pct: float = 0
    high_temperature_efficiency_pct: float = 0
    low_temperature_efficiency_pct: float = 0
    #  add cooling efficieny here TODO

    @pydantic.validator("name")
    def ensure_generator_in_name(cls, name):
        assert "generator" in name
        return name


class GeneratorOneInterval(Asset):
    electric_generation_mwh: pulp.LpVariable
    gas_consumption_mwh: pulp.LpVariable
    high_temperature_generation_mwh: pulp.LpVariable
    low_temperature_generation_mwh: pulp.LpVariable
    binary: pulp.LpVariable
    cfg: GeneratorConfig


def boiler_one_interval(
    optimizer: Optimizer, cfg: BoilerConfig, i: int, freq: Freq
) -> BoilerOneInterval:
    return BoilerOneInterval(
        high_temperature_generation_mwh=optimizer.continuous(
            f"{cfg.name}-high_temperature_generation_mwh-{i}",
            low=freq.mw_to_mwh(cfg.high_temperature_generation_min_mw),
            up=freq.mw_to_mwh(cfg.high_temperature_generation_max_mw),
        ),
        binary=optimizer.binary(
            f"{cfg.name}-binary_mwh-{i}",
        ),
        gas_consumption_mwh=optimizer.continuous(f"{cfg.name}-gas_consumption_mwh-{i}"),
        cfg=cfg,
    )


def constrain_within_interval_boilers(
    optimizer: Optimizer, vars: dict, freq: Freq
) -> None:
    for asset in vars["boilers"][-1]:
        optimizer.constrain(
            asset.gas_consumption_mwh
            == asset.high_temperature_generation_mwh
            * (1 / asset.cfg.high_temperature_efficiency_pct)
        )

        optimizer.constrain_max(
            asset.high_temperature_generation_mwh,
            asset.binary,
            freq.mw_to_mwh(asset.cfg.high_temperature_generation_max_mw),
        )
        optimizer.constrain_min(
            asset.high_temperature_generation_mwh,
            asset.binary,
            freq.mw_to_mwh(asset.cfg.high_temperature_generation_min_mw),
        )


def generator_one_interval(
    optimizer: Optimizer, cfg: GeneratorConfig, i: int, freq: Freq
) -> GeneratorOneInterval:
    #  probably need to include more from config here???
    #  maybe just include the config itself????
    return GeneratorOneInterval(
        electric_generation_mwh=optimizer.continuous(
            f"{cfg.name}-electric_generation_mwh-{i}",
            low=0,
            up=freq.mw_to_mwh(cfg.electric_power_max_mw),
        ),
        binary=optimizer.binary(
            f"{cfg.name}-binary_mwh-{i}",
        ),
        gas_consumption_mwh=optimizer.continuous(f"{cfg.name}-gas_consumption_mwh-{i}"),
        high_temperature_generation_mwh=optimizer.continuous(
            f"{cfg.name}-high_temperature_generation_mwh-{i}",
            low=0,
        ),
        low_temperature_generation_mwh=optimizer.continuous(
            f"{cfg.name}-low_temperature_generation_mwh-{i}",
            low=0,
        ),
        cfg=cfg,
    )


def constrain_within_interval_generators(
    optimizer: Optimizer, vars: dict, freq: Freq
) -> None:
    for asset in vars["generators"][-1]:
        optimizer.constrain(
            asset.gas_consumption_mwh
            == asset.electric_generation_mwh * (1 / asset.cfg.electric_efficiency_pct)
        )
        optimizer.constrain(
            asset.high_temperature_generation_mwh
            == asset.gas_consumption_mwh * asset.cfg.high_temperature_efficiency_pct
        )
        optimizer.constrain(
            asset.low_temperature_generation_mwh
            == asset.gas_consumption_mwh * asset.cfg.low_temperature_efficiency_pct
        )
        #  add cooling constraint here TODO

        optimizer.constrain_max(
            asset.electric_generation_mwh,
            asset.binary,
            freq.mw_to_mwh(asset.cfg.electric_power_max_mw),
        )
        optimizer.constrain_min(
            asset.electric_generation_mwh,
            asset.binary,
            freq.mw_to_mwh(asset.cfg.electric_power_min_mw),
        )


class Generator:
    def __init__(
        self,
        electric_power_max_mw: float = 0.0,
        electric_power_min_mw: float = 0.0,
        electric_efficiency_pct: float = 0.0,
        high_temperature_efficiency_pct: float = 0.0,
        low_temperature_efficiency_pct: float = 0.0,
    ):
        """
        Make sure to get your efficiencies and gas prices on the same basis LHV or HHV!
        """
        self.cfg = GeneratorConfig(
            name="generator-alpha",
            electric_power_min_mw=electric_power_min_mw,
            electric_power_max_mw=electric_power_max_mw,
            electric_efficiency_pct=electric_efficiency_pct,
            high_temperature_efficiency_pct=high_temperature_efficiency_pct,
            low_temperature_efficiency_pct=low_temperature_efficiency_pct,
        )

    def optimize(
        self,
        electricity_prices,
        gas_prices=None,
        electricity_carbon_intensities=None,
        high_temperature_load_mwh=None,
        low_temperature_load_mwh=None,
        freq_mins: int = defaults.freq_mins,
        objective: str = "price",
    ) -> pd.DataFrame:
        self.optimizer = Optimizer()
        freq = Freq(freq_mins)
        interval_data = epl.data.IntervalData(
            electricity_prices=electricity_prices,
            gas_prices=gas_prices,
            electricity_carbon_intensities=electricity_carbon_intensities,
            high_temperature_load_mwh=high_temperature_load_mwh,
            low_temperature_load_mwh=low_temperature_load_mwh,
        )
        self.site_cfg = epl.site.SiteConfig()
        self.spill_cfg = epl.spill.SpillConfig()
        self.valve_cfg = epl.valve.ValveConfig(name="valve-alpha")

        default_boiler_size = freq.mw_to_mwh(
            max(interval_data.high_temperature_load_mwh)
            + max(interval_data.low_temperature_load_mwh)
        )
        self.default_boiler_cfg = BoilerConfig(
            name="boiler-alpha",
            high_temperature_generation_max_mw=default_boiler_size,
            high_temperature_efficiency_pct=defaults.default_boiler_efficiency_pct,
        )
        vars = collections.defaultdict(list)

        for i in interval_data.idx:
            vars["sites"].append(
                epl.site.site_one_interval(self.optimizer, self.site_cfg, i, freq)
            )
            vars["spills"].append(
                epl.spill.spill_one_interval(self.optimizer, self.spill_cfg, i, freq)
            )
            vars["valves"].append(
                epl.valve.valve_one_interval(self.optimizer, self.valve_cfg, i, freq)
            )

            generators = [
                generator_one_interval(self.optimizer, self.cfg, i, freq),
            ]
            boilers = [
                boiler_one_interval(self.optimizer, self.default_boiler_cfg, i, freq),
            ]
            vars["generators"].append(generators)
            vars["boilers"].append(boilers)
            vars["assets"].append([*generators, *boilers])

            epl.site.constrain_within_interval(self.optimizer, vars, interval_data, i)
            constrain_within_interval_generators(self.optimizer, vars, freq)
            constrain_within_interval_boilers(self.optimizer, vars, freq)
            epl.valve.constrain_within_interval_valve(self.optimizer, vars)

        assert (
            len(interval_data.idx)
            == len(vars["assets"])
            == len(vars["generators"])
            == len(vars["boilers"])
        )

        objective_fn = objectives[objective]
        self.optimizer.objective(objective_fn(self.optimizer, vars, interval_data))
        self.optimizer.solve()
        return epl.results.extract_results(interval_data, vars)
