import collections
import typing

import pulp
import pydantic

import energypylinear as epl
from energypylinear.battery import Asset
from energypylinear.defaults import defaults
from energypylinear.freq import Freq
from energypylinear.optimizer import Pulp


class BoilerConfig(Asset):
    high_temperature_generation_max_mw: float = 0
    high_temperature_generation_min_mw: float = 0
    high_temperature_efficiency_pct: float = 0


class BoilerOneInterval(Asset):
    high_temperature_generation_mwh: pulp.LpVariable
    binary: pulp.LpVariable


def boiler_one_interval(
    optimizer: Pulp, cfg: BoilerConfig, i: int, freq: Freq
) -> BoilerOneInterval:
    #  probably need to include more from config here???
    #  maybe just include the config itself????

    #  probably need variables for the heat + cooling output
    #  these would be set / controlled by constraints????
    return BoilerOneInterval(
        high_temperature_generation_mwh=optimizer.continuous(
            f"electric_generation_mwh-{i}",
            low=freq.mw_to_mwh(cfg.high_temperature_generation_max_mw),
            up=freq.mw_to_mwh(cfg.high_temperature_generation_min_mw),
        ),
        binary=optimizer.binary(
            f"binary_mwh-{i}",
        ),
    )

def constrain_within_interval_boilers(optimizer: Pulp, vars: dict, freq: Freq):
    for asset in vars["boilers"][-1]:
        #  fuel consumption + min and max


class GeneratorConfig(Asset):
    electric_power_max_mw: float = 0
    electric_power_min_mw: float = 0

    electric_efficiency_pct: float = 0
    high_temperature_efficiency_pct: float = 0
    low_temperature_efficiency_pct: float = 0
    #  add cooling efficieny here TODO


class GeneratorOneInterval(Asset):
    electric_generation_mwh: pulp.LpVariable
    fuel_consumption_mwh: pulp.LpVariable
    binary: pulp.LpVariable
    cfg: GeneratorConfig


def generator_one_interval(
    optimizer: Pulp, cfg: GeneratorConfig, i: int, freq: Freq
) -> GeneratorOneInterval:
    #  probably need to include more from config here???
    #  maybe just include the config itself????
    return GeneratorOneInterval(
        electric_generation_mwh=optimizer.continuous(
            f"electric_generation_mwh-{i}",
            low=freq.mw_to_mwh(cfg.electric_power_min_mw),
            up=freq.mw_to_mwh(cfg.electric_power_max_mw),
        ),
        binary=optimizer.binary(
            f"binary_mwh-{i}",
        ),
        fuel_consumption_mwh=optimizer.continuous(
            f"electric_generation_mwh-{i}",
            low=freq.mw_to_mwh(cfg.electric_power_min_mw),
            up=freq.mw_to_mwh(cfg.electric_power_max_mw),
        ),
        cfg=cfg,
    )


def constrain_within_interval(optimizer: Pulp, vars: dict, freq: Freq) -> None:
    for asset in vars["generators"][-1]:
        optimizer.constrain(
            asset.fuel_consumption_mwh
            == asset.electric_generation_mwh * asset.cfg.electric_efficiency_pct
        )
        optimizer.constrain(
            asset.high_temperature_generation_mwh
            == asset.fuel_consumption_mwh * asset.cfg.high_temperature_efficiency_pct
        )
        optimizer.constrain(
            asset.low_temperature_generation_mwh
            == asset.fuel_consumption_mwh * asset.cfg.low_temperature_efficiency_pct
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
        electric_power_mw: float,
        electric_efficiency_pct: float = 0.0,
        high_temperature_efficiency_pct: float = 0.0,
        low_temperature_efficiency_pct: float = 0.0,
        cooling_efficiency_pct: float = 0.0,
    ):
        """
        Make sure to get your efficiencies and gas prices on the same basis LHV or HHV!
        """
        self.cfg = GeneratorConfig(
            electric_power_mw=electric_power_mw,
            electric_efficiency_pct=electric_efficiency_pct,
            high_temperature_efficiency_pct=high_temperature_efficiency_pct,
            low_temperature_efficiency_pct=low_temperature_efficiency_pct,
            cooling_efficiency_pct=cooling_efficiency_pct,
        )
        self.spill_boiler_cfg = BoilerConfig(
            high_temperature_output_max_mw=defaults.spill_boiler_size_mw,
            high_temperature_efficiency_pct=defaults.spill_boiler_efficiency_pct,
        )
        self.optimizer = Pulp()

    def optimize(
        self,
        electricity_prices,
        gas_prices=None,
        carbon_intensities=None,
        high_temperature_load_mwh=None,
        low_temperature_load_mwh=None,
        freq_mins: int = defaults.freq_mins,
    ):
        freq = Freq(freq_mins)

        interval_data = epl.data.IntervalData(
            electricity_prices=electricity_prices,
            gas_prices=gas_prices,
            carbon_intensities=carbon_intensities,
            high_temperature_load_mwh=high_temperature_load_mwh,
            low_temperature_load_mwh=low_temperature_load_mwh,
        )
        self.site_cfg = epl.site.SiteConfig()
        vars = collections.defaultdict(list)

        for i in interval_data.idx:
            #  setup linear program variables
            vars["sites"].append(
                epl.site.site_one_interval(self.optimizer, self.site_cfg, i, freq)
            )
            generators = [
                generator_one_interval(self.optimizer, self.cfg, i, freq),
            ]
            boilers = [
                boiler_one_interval(self.optimizer, self.spill_boiler_cfg, i, freq),
            ]
            vars["generators"].append(generators)
            vars["boilers"].append(boilers)
            vars["assets"].append([*generators, *boilers])
            epl.site.constrain_within_interval(self.optimizer, vars)

            #  generator constraints
            constrain_within_interval(self.optimizer, vars, freq)

            #  boiler constraints - TODO
            constrain_within_interval_boilers(self.optimizer, vars, freq)
