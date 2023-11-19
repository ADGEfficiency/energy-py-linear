"""Asset for optimizing combined heat and power (CHP) generators."""
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


class CHPConfig(pydantic.BaseModel):
    """CHP configuration."""

    name: str
    electric_power_max_mw: float = 0
    electric_power_min_mw: float = 0

    electric_efficiency_pct: float = 0
    high_temperature_efficiency_pct: float = 0
    low_temperature_efficiency_pct: float = 0

    freq_mins: int

    @pydantic.field_validator("name")
    @classmethod
    def check_name(cls, name: str) -> str:
        """Ensure we can identify this asset correctly."""
        assert "chp" in name
        return name


class CHPOneInterval(AssetOneInterval):
    """CHP generator data for a single interval."""

    cfg: CHPConfig

    binary: pulp.LpVariable
    electric_generation_mwh: pulp.LpVariable
    gas_consumption_mwh: pulp.LpVariable
    high_temperature_generation_mwh: pulp.LpVariable
    low_temperature_generation_mwh: pulp.LpVariable


class CHP(epl.Asset):
    """CHP asset - handles optimization and plotting of results over many intervals.

    A CHP (combined heat and power) generator generates electricity, high temperature
    and low temperature heat from natural gas.

    This asset can be used to model gas turbines, gas engines or open cycle generators
    like diesel generators.

    Args:
        electric_power_max_mw - maximum electric power output of the generator in mega-watts.
        electric_power_min_mw - minimum electric power output of the generator in mega-watts.
        electric_efficiency_pct - electric efficiency of the generator, measured in percentage.
        high_temperature_efficiency_pct - high temperature efficiency of the generator, measured in percentage.
        low_temperature_efficiency_pct - the low temperature efficiency of the generator, measured in percentage.
        electricity_prices: the price of electricity in each interval.
        gas_prices: the prices of natural gas, used in CHP and boilers in each interval.
        electricity_carbon_intensities: carbon intensity of electricity in each interval.
        high_temperature_load_mwh: high temperature load of the site in mega-watt hours.
        low_temperature_load_mwh: low temperature load of the site in mega-watt hours.
        freq_mins: the size of an interval in minutes.

    Make sure to get your efficiencies and gas prices on the same basis (HHV or LHV).
    """

    def __init__(
        self,
        electric_efficiency_pct: float = 0.0,
        electric_power_max_mw: float = 0.0,
        electric_power_min_mw: float = 0.0,
        high_temperature_efficiency_pct: float = 0.0,
        low_temperature_efficiency_pct: float = 0.0,
        name: str = "chp",
        freq_mins: int = defaults.freq_mins,
        electricity_prices: np.ndarray | list[float] | float | None = None,
        export_electricity_prices: np.ndarray | list[float] | float | None = None,
        electricity_carbon_intensities: np.ndarray | list[float] | float | None = None,
        gas_prices: np.ndarray | list[float] | float | None = None,
        high_temperature_load_mwh: np.ndarray | list[float] | float | None = None,
        low_temperature_load_mwh: np.ndarray | list[float] | float | None = None,
        low_temperature_generation_mwh: np.ndarray | list[float] | float | None = None,
    ):
        """Initializes the asset."""
        self.cfg = CHPConfig(
            name=name,
            electric_power_min_mw=electric_power_min_mw,
            electric_power_max_mw=electric_power_max_mw,
            electric_efficiency_pct=electric_efficiency_pct,
            high_temperature_efficiency_pct=high_temperature_efficiency_pct,
            low_temperature_efficiency_pct=low_temperature_efficiency_pct,
            freq_mins=freq_mins,
        )

        if electricity_prices is not None or electricity_carbon_intensities is not None:
            assets = [self, epl.Spill(), epl.Valve(), epl.Boiler()]
            self.site = epl.Site(
                assets=assets,
                electricity_prices=electricity_prices,
                export_electricity_prices=export_electricity_prices,
                electricity_carbon_intensities=electricity_carbon_intensities,
                gas_prices=gas_prices,
                high_temperature_load_mwh=high_temperature_load_mwh,
                low_temperature_load_mwh=low_temperature_load_mwh,
                low_temperature_generation_mwh=low_temperature_generation_mwh,
                freq_mins=self.cfg.freq_mins,
            )

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<energypylinear.CHP {self.cfg.electric_power_max_mw=}>"

    def one_interval(
        self, optimizer: Optimizer, i: int, freq: Freq, flags: Flags = Flags()
    ) -> CHPOneInterval:
        """Generate linear program data for one interval."""
        return CHPOneInterval(
            electric_generation_mwh=optimizer.continuous(
                f"{self.cfg.name}-electric_generation_mwh-{i}",
                low=0,
                up=freq.mw_to_mwh(self.cfg.electric_power_max_mw),
            ),
            binary=optimizer.binary(
                f"{self.cfg.name}-binary_mwh-{i}",
            ),
            gas_consumption_mwh=optimizer.continuous(
                f"{self.cfg.name}-gas_consumption_mwh-{i}"
            ),
            high_temperature_generation_mwh=optimizer.continuous(
                f"{self.cfg.name}-high_temperature_generation_mwh-{i}",
                low=0,
            ),
            low_temperature_generation_mwh=optimizer.continuous(
                f"{self.cfg.name}-low_temperature_generation_mwh-{i}",
                low=0,
            ),
            cfg=self.cfg,
        )

    def constrain_within_interval(
        self,
        optimizer: Optimizer,
        ivars: "epl.interval_data.IntervalVars",
        i: int,
        freq: Freq,
        flags: Flags,
    ) -> None:
        """Constrain generator upper and lower bounds for generating electricity, high
        and low temperature heat within a single interval."""
        chp = ivars.filter_objective_variables(
            instance_type=CHPOneInterval, i=i, asset_name=self.cfg.name
        )[0]
        assert isinstance(chp, CHPOneInterval)
        if chp.cfg.electric_efficiency_pct > 0:
            optimizer.constrain(
                chp.gas_consumption_mwh
                == chp.electric_generation_mwh * (1 / chp.cfg.electric_efficiency_pct)
            )
        optimizer.constrain(
            chp.high_temperature_generation_mwh
            == chp.gas_consumption_mwh * chp.cfg.high_temperature_efficiency_pct
        )
        optimizer.constrain(
            chp.low_temperature_generation_mwh
            == chp.gas_consumption_mwh * chp.cfg.low_temperature_efficiency_pct
        )
        optimizer.constrain_max(
            chp.electric_generation_mwh,
            chp.binary,
            freq.mw_to_mwh(chp.cfg.electric_power_max_mw),
        )
        optimizer.constrain_min(
            chp.electric_generation_mwh,
            chp.binary,
            freq.mw_to_mwh(chp.cfg.electric_power_min_mw),
        )

    def constrain_after_intervals(
        self, *args: typing.Tuple[typing.Any], **kwargs: typing.Any
    ) -> None:
        """Constrain the asset after all intervals."""
        return

    def optimize(
        self,
        objective: "str | dict | epl.objectives.CustomObjectiveFunction" = "price",
        verbose: int | bool = 2,
        flags: Flags = Flags(),
        optimizer_config: "epl.OptimizerConfig | dict" = epl.optimizer.OptimizerConfig(),
    ) -> "epl.SimulationResult":
        """
        Optimize the CHP generator's dispatch using a mixed-integer linear program.

        Args:
            objective: the optimization objective - either "price" or "carbon".
            verbose: level of printing.
            flags: boolean flags to change simulation and results behaviour.
            optimizer_config: configuration options for the optimizer.

        Returns:
            epl.results.SimulationResult
        """
        return self.site.optimize(
            objective=objective,
            flags=flags,
            verbose=verbose,
            optimizer_config=optimizer_config,
        )

    def plot(self, results: "epl.SimulationResult", path: pathlib.Path | str) -> None:
        """Plot simulation results."""
        return epl.plot.plot_chp(results, pathlib.Path(path))
