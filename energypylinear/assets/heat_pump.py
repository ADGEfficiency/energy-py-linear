"""Heat Pump asset."""
import pathlib

import numpy as np
import pulp
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval
from energypylinear.defaults import defaults
from energypylinear.flags import Flags


class HeatPumpConfig(pydantic.BaseModel):
    """Heat Pump asset configuration."""

    cop: float
    electric_power_mw: float
    freq_mins: int
    include_valve: bool
    name: str

    @pydantic.validator("cop", pre=True, always=True)
    def validate_cop(cls, value: float) -> float:
        """Check COP is greater than 1.0.

        Heat balance doesn't make sense with a COP less than 1.0.

        The HT heat output is the sum of the LT heat and electricity input.

        Introducing losses to the heat pump would allow COPs less than 1.0.
        """
        if value < 1:
            raise ValueError("COP must be 1 or above")
        return value


class HeatPumpOneInterval(AssetOneInterval):
    """Heat pump asset data for a single interval."""

    cfg: HeatPumpConfig
    electric_load_mwh: pulp.LpVariable
    electric_load_binary: pulp.LpVariable
    low_temperature_load_mwh: pulp.LpVariable
    high_temperature_generation_mwh: pulp.LpVariable


class HeatPump(epl.Asset):
    """Heat pump asset - handles optimization and plotting of results over many intervals.

    A heat pump generates high temperature heat from low temperature heat and electricity.

    Args:
        electric_power_mw: the maximum power input of the heat pump.
            Measured in in mega-watts.
        cop: the coefficient of performance of the heat pump.
            The ratio of high temperature heat output over input electricity.
        name: the asset name.
        freq_mins: the size of an interval in minutes.

        include_valve: whether to allow heat to flow from high to low temperature.
        electricity_prices: the price of electricity in each interval.
        gas_prices: the prices of natural gas, used in CHP and boilers in each interval.
        electricity_carbon_intensities: carbon intensity of electricity in each interval.
        high_temperature_load_mwh: high temperature load of the site in mega-watt hours.
        low_temperature_load_mwh: low temperature load of the site in mega-watt hours.
        low_temperature_generation_mwh: low temperature heat generated by the site in mega-watt hours.
    """

    def __init__(
        self,
        cop: float = 3.0,
        electric_power_mw: float = 1.0,
        freq_mins: int = defaults.freq_mins,
        include_valve: bool = True,
        name: str = "heat-pump",
        electricity_prices: np.ndarray | list[float] | float | None = None,
        export_electricity_prices: np.ndarray | list[float] | float | None = None,
        electricity_carbon_intensities: np.ndarray | list[float] | float | None = None,
        gas_prices: np.ndarray | list[float] | float | None = None,
        high_temperature_load_mwh: np.ndarray | list[float] | float | None = None,
        low_temperature_load_mwh: np.ndarray | list[float] | float | None = None,
        low_temperature_generation_mwh: np.ndarray | list[float] | float | None = None,
    ):
        """Initializes the asset."""
        self.cfg = HeatPumpConfig(
            cop=cop,
            electric_power_mw=electric_power_mw,
            freq_mins=freq_mins,
            include_valve=include_valve,
            name=name,
        )

        if electricity_prices is not None or electricity_carbon_intensities is not None:
            assets = [
                self,
                epl.Spill(),
                epl.Boiler(),
            ]
            if include_valve:
                assets.append(epl.Valve())

            self.site = epl.Site(
                assets=assets,
                electricity_prices=electricity_prices,
                electricity_carbon_intensities=electricity_carbon_intensities,
                export_electricity_prices=export_electricity_prices,
                gas_prices=gas_prices,
                high_temperature_load_mwh=high_temperature_load_mwh,
                low_temperature_load_mwh=low_temperature_load_mwh,
                low_temperature_generation_mwh=low_temperature_generation_mwh,
                freq_mins=self.cfg.freq_mins,
            )

    def __repr__(self) -> str:
        """A string representation of self."""
        return (
            f"<energypylinear.HeatPump {self.cfg.electric_power_mw=}, {self.cfg.cop=}>"
        )

    def one_interval(
        self, optimizer: "epl.Optimizer", i: int, freq: "epl.Freq", flags: "epl.Flags"
    ) -> HeatPumpOneInterval:
        """Create asset data for a single interval."""
        name = f"i:{i},asset:{self.cfg.name}"
        return HeatPumpOneInterval(
            cfg=self.cfg,
            electric_load_mwh=optimizer.continuous(
                f"electric_load_mwh,{name}",
                low=0,
                up=freq.mw_to_mwh(self.cfg.electric_power_mw),
            ),
            electric_load_binary=optimizer.binary(
                f"electric_load_binary,{name}",
            ),
            low_temperature_load_mwh=optimizer.continuous(
                f"low_temperature_load_mwh,{name}",
                low=0,
            ),
            high_temperature_generation_mwh=optimizer.continuous(
                f"high_temperature_generation_mwh,{name}",
                low=0,
            ),
        )

    def constrain_within_interval(
        self,
        optimizer: "epl.Optimizer",
        ivars: "epl.IntervalVars",
        i: int,
        freq: "epl.Freq",
        flags: "epl.Flags",
    ) -> None:
        """Constrain asset within a single interval."""
        heat_pump = ivars.filter_objective_variables(
            HeatPumpOneInterval, i=i, asset_name=self.cfg.name
        )[0][0]
        assert isinstance(heat_pump, HeatPumpOneInterval)
        optimizer.constrain_max(
            heat_pump.electric_load_mwh,
            heat_pump.electric_load_binary,
            freq.mw_to_mwh(heat_pump.cfg.electric_power_mw),
        )
        optimizer.constrain(
            heat_pump.high_temperature_generation_mwh
            == heat_pump.electric_load_mwh * heat_pump.cfg.cop
        )
        optimizer.constrain(
            heat_pump.low_temperature_load_mwh + heat_pump.electric_load_mwh
            == heat_pump.high_temperature_generation_mwh
        )

    def constrain_after_intervals(
        self,
        optimizer: "epl.Optimizer",
        ivars: "epl.IntervalVars",
    ) -> None:
        """Constrain the asset after all intervals."""
        pass

    def optimize(
        self,
        objective: str = "price",
        verbose: bool = True,
        flags: Flags = Flags(),
        optimizer_config: "epl.OptimizerConfig | dict" = epl.optimizer.OptimizerConfig(),
    ) -> "epl.SimulationResult":
        """Optimize the asset dispatch using a mixed-integer linear program.

        Args:
            objective: the optimization objective - either "price" or "carbon".
            flags: boolean flags to change simulation and results behaviour.
            verbose: level of printing.
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
        return epl.plot.plot_heat_pump(
            results, pathlib.Path(path), asset_name=self.cfg.name
        )
