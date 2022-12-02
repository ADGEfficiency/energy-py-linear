import collections
import typing

import pandas as pd
import pulp
import pydantic

import energypylinear as epl
from energypylinear.battery import Asset
from energypylinear.defaults import defaults
from energypylinear.freq import Freq
from energypylinear.optimizer import Pulp


class SpillConfig(Asset):
    name: str = "spill"


class SpillOneInterval(Asset):
    cfg: SpillConfig
    electric_generation_mwh: pulp.LpVariable
    high_temperature_generation_mwh: pulp.LpVariable

    electric_load_mwh: pulp.LpVariable
    high_temperature_load_mwh: pulp.LpVariable


def spill_one_interval(
    optimizer: Pulp, cfg: SpillConfig, i: int, freq: Freq
) -> SpillOneInterval:
    return SpillOneInterval(
        cfg=cfg,
        electric_generation_mwh=optimizer.continuous(
            f"{cfg.name}-electric_generation_mwh-{i}",
        ),
        high_temperature_generation_mwh=optimizer.continuous(
            f"{cfg.name}-high_temperature_generation_mwh-{i}",
        ),
        electric_load_mwh=optimizer.continuous(
            f"{cfg.name}-electric_load_mwh-{i}",
        ),
        high_temperature_load_mwh=optimizer.continuous(
            f"{cfg.name}-high_temperature_load_mwh-{i}",
        ),
    )


class BoilerConfig(Asset):
    name: str
    high_temperature_generation_max_mw: float = 0
    high_temperature_generation_min_mw: float = 0
    high_temperature_efficiency_pct: float = 0


class BoilerOneInterval(Asset):
    high_temperature_generation_mwh: pulp.LpVariable
    gas_consumption_mwh: pulp.LpVariable
    binary: pulp.LpVariable
    cfg: BoilerConfig


def boiler_one_interval(
    optimizer: Pulp, cfg: BoilerConfig, i: int, freq: Freq
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


def constrain_within_interval_boilers(optimizer: Pulp, vars: dict, freq: Freq):
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


class GeneratorConfig(Asset):
    name: str
    electric_power_max_mw: float = 0
    electric_power_min_mw: float = 0

    electric_efficiency_pct: float = 0
    high_temperature_efficiency_pct: float = 0
    low_temperature_efficiency_pct: float = 0
    #  add cooling efficieny here TODO


class GeneratorOneInterval(Asset):
    electric_generation_mwh: pulp.LpVariable
    gas_consumption_mwh: pulp.LpVariable
    binary: pulp.LpVariable
    cfg: GeneratorConfig


def generator_one_interval(
    optimizer: Pulp, cfg: GeneratorConfig, i: int, freq: Freq
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
            f"{cfg.name}-high_temperature_generation_mwh-{i}"
        ),
        cfg=cfg,
    )


def constrain_within_interval_generators(
    optimizer: Pulp, vars: dict, freq: Freq
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
        electric_power_max_mw: float,
        electric_power_min_mw: float = 0.0,
        electric_efficiency_pct: float = 0.0,
        high_temperature_efficiency_pct: float = 0.0,
        low_temperature_efficiency_pct: float = 0.0,
        cooling_efficiency_pct: float = 0.0,
    ):
        """
        Make sure to get your efficiencies and gas prices on the same basis LHV or HHV!
        """
        self.cfg = GeneratorConfig(
            name="generator",
            electric_power_min_mw=electric_power_min_mw,
            electric_power_max_mw=electric_power_max_mw,
            electric_efficiency_pct=electric_efficiency_pct,
            high_temperature_efficiency_pct=high_temperature_efficiency_pct,
            low_temperature_efficiency_pct=low_temperature_efficiency_pct,
            cooling_efficiency_pct=cooling_efficiency_pct,
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

        default_boiler_size = freq.mw_to_mwh(
            max(interval_data.high_temperature_load_mwh)
            + max(interval_data.low_temperature_load_mwh)
        )
        self.default_boiler_cfg = BoilerConfig(
            name="boiler",
            high_temperature_generation_max_mw=default_boiler_size,
            high_temperature_efficiency_pct=defaults.default_boiler_efficiency_pct,
        )
        self.spill_cfg = SpillConfig()
        vars = collections.defaultdict(list)

        for i in interval_data.idx:
            vars["sites"].append(
                epl.site.site_one_interval(self.optimizer, self.site_cfg, i, freq)
            )
            vars["spills"].append(
                spill_one_interval(self.optimizer, self.spill_cfg, i, freq)
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

        assert (
            len(interval_data.idx)
            == len(vars["assets"])
            == len(vars["generators"])
            == len(vars["boilers"])
        )

        from energypylinear import objectives

        self.optimizer.objective(
            objectives.price_objective(self.optimizer, vars, interval_data)
        )
        status = self.optimizer.solve()
        print(status)

        #  extract results step

        results = collections.defaultdict(list)
        for i in interval_data.idx:
            site = vars["sites"][i]

            results["import_power_mwh"].append(site.import_power_mwh.value())
            results["export_power_mwh"].append(site.export_power_mwh.value())

            spill = vars["spills"][i]
            for attr in [
                "electric_generation_mwh",
                "high_temperature_generation_mwh",
                "electric_load_mwh",
                "high_temperature_load_mwh",
            ]:
                name = f"{spill.cfg.name}"
                results[f"{name}-{attr}"].append(getattr(spill, attr).value())

            generators = vars["generators"][i]
            for generator in generators:
                name = f"{generator.cfg.name}"
                for attr in [
                    "electric_generation_mwh",
                    "gas_consumption_mwh",
                    "high_temperature_generation_mwh",
                ]:
                    results[f"{name}-{attr}"].append(getattr(generator, attr).value())

            boilers = vars["boilers"][i]
            for boiler in boilers:
                name = f"{boiler.cfg.name}"
                for attr in ["high_temperature_generation_mwh", "gas_consumption_mwh"]:
                    results[f"{name}-{attr}"].append(getattr(boiler, attr).value())

        #  add totals
        #  can I do this without pandas??
        results = pd.DataFrame(results)
        for col in [
            "electric_generation_mwh",
            "gas_consumption_mwh",
            "high_temperature_generation_mwh",
        ]:
            cols = [c for c in results.columns if (col in c)]
            results[f"total-{col}"] = results[cols].sum(axis=1)

        #  add balances + check them - TODO
        epl.accounting.validate_results(results)

        #  add warnings on the spill generator + boiler use

        return results
