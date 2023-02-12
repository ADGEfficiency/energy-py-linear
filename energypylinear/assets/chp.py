"""CHP asset for optimizing dispatch of combined heat and power (CHP) generators."""
import collections
import pathlib
import typing

import numpy as np
import pulp
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval
from energypylinear.defaults import defaults
from energypylinear.flags import Flags
from energypylinear.freq import Freq
from energypylinear.optimizer import Optimizer


class BoilerConfig(AssetOneInterval):
    """Gas boiler configuration."""

    name: str
    high_temperature_generation_max_mw: float = 0
    high_temperature_generation_min_mw: float = 0
    high_temperature_efficiency_pct: float = 0


class GeneratorConfig(pydantic.BaseModel):
    """CHP generator configuration."""

    name: str
    electric_power_max_mw: float = 0
    electric_power_min_mw: float = 0

    electric_efficiency_pct: float = 0
    high_temperature_efficiency_pct: float = 0
    low_temperature_efficiency_pct: float = 0
    #  add cooling efficieny here TODO

    @pydantic.validator("name")
    def check_name(cls, name: str) -> str:
        """Ensure we can identify this asset correctly."""
        assert "generator" in name
        return name


class BoilerOneInterval(AssetOneInterval):
    """Boiler data for a single interval."""

    cfg: BoilerConfig
    high_temperature_generation_mwh: pulp.LpVariable
    gas_consumption_mwh: pulp.LpVariable
    binary: pulp.LpVariable


class GeneratorOneInterval(AssetOneInterval):
    """CHP generator data for a single interval."""

    electric_generation_mwh: pulp.LpVariable
    gas_consumption_mwh: pulp.LpVariable
    high_temperature_generation_mwh: pulp.LpVariable
    low_temperature_generation_mwh: pulp.LpVariable
    binary: pulp.LpVariable
    cfg: GeneratorConfig


def boiler_one_interval(
    optimizer: Optimizer, cfg: BoilerConfig, i: int, freq: Freq
) -> BoilerOneInterval:
    """Create Boiler asset data for a single interval."""
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


def generator_one_interval(
    optimizer: Optimizer, cfg: GeneratorConfig, i: int, freq: Freq
) -> GeneratorOneInterval:
    """Create a Generator asset model for one interval."""
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


def constrain_within_interval_boilers(
    optimizer: Optimizer, vars: dict, freq: Freq
) -> None:
    """Constrain boiler upper and lower bounds for generating high & low temperature heat."""
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


def constrain_within_interval_generators(
    optimizer: Optimizer, vars: dict, freq: Freq
) -> None:
    """Constrain generator upper and lower bounds for generating electricity, high & low temperature heat."""
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
    """CHP generator asset - handles optimization and plotting of results over many intervals.

    Args:
        electric_power_max_mw - maximum electric power output of the generator in mega-watts.
        electric_power_min_mw - minimum electric power output of the generator in mega-watts.
        electric_efficiency_pct - electric efficiency of the generator, measured in percentage.
        high_temperature_efficiency_pct - high temperature efficiency of the generator, measured in percentage.
        low_temperature_efficiency_pct - the low temperature efficiency of the generator, measured in percentage.

    Make sure to get your efficiencies and gas prices on the same basis (HHV or LHV).
    """

    def __init__(
        self,
        electric_power_max_mw: float = 0.0,
        electric_power_min_mw: float = 0.0,
        electric_efficiency_pct: float = 0.0,
        high_temperature_efficiency_pct: float = 0.0,
        low_temperature_efficiency_pct: float = 0.0,
    ):
        """Initialize a Battery asset model."""
        self.cfg = GeneratorConfig(
            name="generator",
            electric_power_min_mw=electric_power_min_mw,
            electric_power_max_mw=electric_power_max_mw,
            electric_efficiency_pct=electric_efficiency_pct,
            high_temperature_efficiency_pct=high_temperature_efficiency_pct,
            low_temperature_efficiency_pct=low_temperature_efficiency_pct,
        )

    def optimize(
        self,
        electricity_prices: typing.Union[np.ndarray, list[float]],
        gas_prices: typing.Union[None, np.ndarray, float] = None,
        electricity_carbon_intensities: typing.Union[
            None, np.ndarray, list[float], float
        ] = None,
        #  should these go in here?  TODO
        high_temperature_load_mwh: typing.Union[None, np.ndarray, list[float]] = None,
        low_temperature_load_mwh: typing.Union[None, np.ndarray, list[float]] = None,
        freq_mins: int = defaults.freq_mins,
        objective: str = "price",
        flags: Flags = Flags(),
    ) -> "epl.results.SimulationResult":
        """
        Optimize the CHP generator's dispatch using a mixed-integer linear program.

        Args:
            electricity_prices: the price of electricity in each interval.
            gas_prices: the prices of natural gas, used in CHP and boilers in each interval.
            electricity_carbon_intensities: carbon intensity of electricity in each interval.
            high_temperature_load_mwh: high temperature load of the site in mega-watt hours.
            low_temperature_load_mwh: low temperature load of the site in mega-watt hours.
            freq_mins: the size of an interval in minutes.
            objective: the optimization objective - either "price" or "carbon".
            flags: boolean flags to change simulation and results behaviour.
        """
        self.optimizer = Optimizer()
        freq = Freq(freq_mins)
        interval_data = epl.interval_data.IntervalData(
            electricity_prices=electricity_prices,
            gas_prices=gas_prices,
            electricity_carbon_intensities=electricity_carbon_intensities,
            high_temperature_load_mwh=high_temperature_load_mwh,
            low_temperature_load_mwh=low_temperature_load_mwh,
        )
        self.site_cfg = epl.site.SiteConfig()
        self.spill_cfg = epl.spill.SpillConfig()
        self.valve_cfg = epl.valve.ValveConfig(name="valve")

        default_boiler_size = freq.mw_to_mwh(
            max(interval_data.high_temperature_load_mwh)
            + max(interval_data.low_temperature_load_mwh)
        )
        self.default_boiler_cfg = BoilerConfig(
            name="boiler",
            high_temperature_generation_max_mw=default_boiler_size,
            high_temperature_efficiency_pct=defaults.default_boiler_efficiency_pct,
        )

        #  TODO - difficult to type the list of list thing
        #  maybe sign something should be reworked
        vars: collections.defaultdict[str, typing.Any] = collections.defaultdict(list)
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

        objective_fn = epl.objectives[objective]
        self.optimizer.objective(objective_fn(self.optimizer, vars, interval_data))
        _, feasible = self.optimizer.solve()
        self.interval_data = interval_data
        return epl.results.extract_results(
            interval_data, vars, feasible=feasible, flags=flags
        )

    def plot(
        self,
        results: "epl.results.SimulationResult",
        path: typing.Union[pathlib.Path, str],
    ) -> None:
        """Plot simulation results."""
        return epl.plot.plot_chp(results, pathlib.Path(path))
