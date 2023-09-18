""""""
import pathlib
import typing

import numpy as np
import pulp
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval
from energypylinear.assets.chp import get_default_boiler_size
from energypylinear.defaults import defaults
from energypylinear.flags import Flags


class HeatPumpConfig(pydantic.BaseModel):
    """Heat pump asset configuration."""

    name: str
    electric_power_mw: float
    cop: float

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


class HeatPump:
    """Heat pump asset - handles optimization and plotting of results over many intervals.

    A heat pump generates high temperature heat from low temperature heat and electricity.

    Args:
        electric_power_mw: the maximum power input of the heat pump.
            Measured in in mega-watts.
        cop: the coefficient of performance of the heat pump.
            The ratio of high temperature heat output over input electricity.
        name: the asset name.
    """

    def __init__(self, electric_power_mw: float, cop: float, name: str = "heat-pump"):
        """Initializes the asset."""
        self.cfg = HeatPumpConfig(
            name=name, electric_power_mw=electric_power_mw, cop=cop
        )
        self.site = epl.Site()
        self.spill = epl.spill.Spill()
        self.valve = epl.valve.Valve()

    def __repr__(self) -> str:
        """A string representation of self."""
        return (
            f"<energypylinear.HeatPump {self.cfg.electric_power_mw=}, {self.cfg.cop=}>"
        )

    def one_interval(
        self,
        optimizer: "epl.Optimizer",
        i: int,
        freq: "epl.Freq",
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> HeatPumpOneInterval:
        """Creates asset data for a single interval."""
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
        ivars: "epl.interval_data.IntervalVars",
        i: int,
        freq: "epl.Freq",
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        """Constrain asset dispatch within a single interval."""
        for heat_pump in ivars.filter_objective_variables(HeatPumpOneInterval, i=i)[0]:
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
        self, *args: typing.Any, **kwargs: typing.Any
    ) -> None:
        """Constrain dispatch after all interval asset models are created."""
        pass

    def optimize(
        self,
        electricity_prices: typing.Sequence[float] | np.ndarray,
        gas_prices: float | typing.Sequence[float] | np.ndarray | None = None,
        electricity_carbon_intensities: float
        | typing.Sequence[float]
        | np.ndarray
        | None = None,
        high_temperature_load_mwh: float
        | typing.Sequence[float]
        | np.ndarray
        | None = None,
        low_temperature_load_mwh: float
        | typing.Sequence[float]
        | np.ndarray
        | None = None,
        low_temperature_generation_mwh: float
        | typing.Sequence[float]
        | np.ndarray
        | None = None,
        freq_mins: int = defaults.freq_mins,
        objective: str = "price",
        verbose: bool = True,
        flags: Flags = Flags(),
        include_valve: bool = True,
    ) -> "epl.results.SimulationResult":
        """Optimize the asset dispatch using a mixed-integer linear program.

        Args:
            electricity_prices: the price of electricity in each interval.
            gas_prices: the prices of natural gas, used in CHP and boilers in each interval.
            electricity_carbon_intensities: carbon intensity of electricity in each interval.
            high_temperature_load_mwh: high temperature load of the site in mega-watt hours.
            low_temperature_load_mwh: low temperature load of the site in mega-watt hours.
            low_temperature_generation_mwh: low temperature heat generated by the site in mega-watt hours.
            freq_mins: the size of an interval in minutes.
            objective: the optimization objective - either "price" or "carbon".
            verbose: level of printing.
            flags: boolean flags to change simulation and results behaviour.
            include_valve: whether to allow heat to flow from high to low temperature.

        Returns:
            epl.results.SimulationResult
        """

        self.optimizer = epl.Optimizer()
        freq = epl.Freq(freq_mins)

        self.idata = epl.interval_data.IntervalData(
            electricity_prices=electricity_prices,
            gas_prices=gas_prices,
            electricity_carbon_intensities=electricity_carbon_intensities,
            high_temperature_load_mwh=high_temperature_load_mwh,
            low_temperature_load_mwh=low_temperature_load_mwh,
            low_temperature_generation_mwh=low_temperature_generation_mwh,
        )

        self.boiler = epl.Boiler(
            high_temperature_generation_max_mw=get_default_boiler_size(
                freq, self.idata
            ),
            high_temperature_efficiency_pct=defaults.default_boiler_efficiency_pct,
        )

        self.ivars = epl.interval_data.IntervalVars()

        for i in self.idata.idx:
            self.ivars.append(
                self.site.one_interval(self.optimizer, self.site.cfg, i, freq)
            )

            ivars_interval = [
                self.one_interval(self.optimizer, i, freq, flags=flags),
                self.boiler.one_interval(self.optimizer, i, freq),
                self.spill.one_interval(self.optimizer, i, freq),
            ]

            if include_valve:
                ivars_interval.append(self.valve.one_interval(self.optimizer, i, freq))

            self.ivars.append(ivars_interval)

            self.site.constrain_within_interval(
                self.optimizer, self.ivars, self.idata, i
            )

            self.constrain_within_interval(self.optimizer, self.ivars, i, freq=freq)

            self.spill.constrain_within_interval(
                self.optimizer, self.ivars, self.idata, i, freq
            )
            self.boiler.constrain_within_interval(
                self.optimizer, self.ivars, self.idata, i, freq
            )
            if include_valve:
                self.valve.constrain_within_interval(
                    self.optimizer, self.ivars, self.idata, i, freq
                )

        #  does nothing at the moment
        self.constrain_after_intervals(i)

        assert len(self.idata.idx) == len(self.ivars.objective_variables)

        objective_fn = epl.objectives[objective]
        self.optimizer.objective(
            objective_fn(
                self.optimizer,
                self.ivars,
                self.idata,
            )
        )

        status = self.optimizer.solve(verbose=verbose)

        #  needed as why??? there is a reason...
        self.interval_data = self.idata
        return epl.results.extract_results(
            self.idata, self.ivars, feasible=status.feasible, verbose=verbose
        )

    def plot(
        self, results: "epl.results.SimulationResult", path: pathlib.Path | str
    ) -> None:
        """Plot simulation results."""
        return epl.plot.plot_heat_pump(
            results, pathlib.Path(path), asset_name=self.cfg.name
        )
