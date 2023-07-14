"""CHP asset for optimizing dispatch of combined heat and power (CHP) generators."""
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


class GeneratorOneInterval(AssetOneInterval):
    """CHP generator data for a single interval."""

    electric_generation_mwh: pulp.LpVariable
    gas_consumption_mwh: pulp.LpVariable
    high_temperature_generation_mwh: pulp.LpVariable
    low_temperature_generation_mwh: pulp.LpVariable
    binary: pulp.LpVariable
    cfg: GeneratorConfig


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
        name: str = "generator",
    ):
        """Initialize a Battery asset model."""
        self.cfg = GeneratorConfig(
            name=name,
            electric_power_min_mw=electric_power_min_mw,
            electric_power_max_mw=electric_power_max_mw,
            electric_efficiency_pct=electric_efficiency_pct,
            high_temperature_efficiency_pct=high_temperature_efficiency_pct,
            low_temperature_efficiency_pct=low_temperature_efficiency_pct,
        )

    def one_interval(
        self, optimizer: Optimizer, i: int, freq: Freq, flags: Flags = Flags()
    ) -> GeneratorOneInterval:
        """Create a Generator asset model for one interval."""
        return GeneratorOneInterval(
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
        interval_data: "epl.IntervalData",
        i: int,
        freq: Freq,
        flags: Flags = Flags(),
    ) -> None:
        """Constrain generator upper and lower bounds for generating electricity, high & low temperature heat."""
        assets = ivars.objective_variables[-1]
        generators = [a for a in assets if isinstance(a, epl.chp.GeneratorOneInterval)]
        for asset in generators:
            if asset.cfg.electric_efficiency_pct > 0:
                optimizer.constrain(
                    asset.gas_consumption_mwh
                    == asset.electric_generation_mwh
                    * (1 / asset.cfg.electric_efficiency_pct)
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

    def constrain_after_intervals(
        self, *args: typing.Tuple[typing.Any], **kwargs: typing.Any
    ) -> None:
        """Constrain asset after all interval asset models are created."""
        return

    def optimize(
        self,
        electricity_prices: np.ndarray | typing.Sequence[float],
        gas_prices: np.ndarray | typing.Sequence[float] | float | None = None,
        # fmt: off
        electricity_carbon_intensities: np.ndarray| typing.Sequence[float] | float | None = None,
        high_temperature_load_mwh: np.ndarray| typing.Sequence[float] | float| None = None,
        low_temperature_load_mwh: np.ndarray| typing.Sequence[float] | float | None = None,
        # fmt: on
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
        self.site = epl.Site()
        self.spill = epl.spill.Spill()
        self.valve = epl.valve.Valve()

        assert interval_data.high_temperature_load_mwh is not None
        assert interval_data.low_temperature_load_mwh is not None
        default_boiler_size = freq.mw_to_mwh(
            max(interval_data.high_temperature_load_mwh)
            + max(interval_data.low_temperature_load_mwh)
        )
        self.boiler = epl.Boiler(
            high_temperature_generation_max_mw=default_boiler_size,
            high_temperature_efficiency_pct=defaults.default_boiler_efficiency_pct,
        )

        ivars = epl.interval_data.IntervalVars()
        for i in interval_data.idx:
            ivars.append(self.site.one_interval(self.optimizer, self.site.cfg, i, freq))
            ivars.append(
                [
                    self.one_interval(self.optimizer, i, freq),
                    self.boiler.one_interval(self.optimizer, i, freq),
                    self.valve.one_interval(self.optimizer, i, freq),
                    self.spill.one_interval(self.optimizer, i, freq),
                ]
            )

            self.site.constrain_within_interval(self.optimizer, ivars, interval_data, i)
            self.constrain_within_interval(
                self.optimizer, ivars, interval_data, i, freq
            )
            self.boiler.constrain_within_interval(
                self.optimizer, ivars, interval_data, i, freq
            )
            self.valve.constrain_within_interval(
                self.optimizer, ivars, interval_data, i, freq
            )
            self.spill.constrain_within_interval(
                self.optimizer, ivars, interval_data, i, freq
            )

        assert len(interval_data.idx) == len(ivars.objective_variables)

        objective_fn = epl.objectives[objective]
        self.optimizer.objective(objective_fn(self.optimizer, ivars, interval_data))
        status = self.optimizer.solve()
        self.interval_data = interval_data
        return epl.results.extract_results(
            interval_data, ivars, feasible=status.feasible, flags=flags
        )

    def plot(
        self,
        results: "epl.results.SimulationResult",
        path: pathlib.Path | str
    ) -> None:
        """Plot simulation results."""
        return epl.plot.plot_chp(results, pathlib.Path(path))
