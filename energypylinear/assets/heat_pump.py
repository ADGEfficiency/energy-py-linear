""""""
import typing
import pathlib

import pydantic
from energypylinear.assets.asset import AssetOneInterval
import numpy as np

import energypylinear as epl
from energypylinear.assets.chp import get_default_boiler_size
from energypylinear.defaults import defaults
from energypylinear.flags import Flags
import pulp


class HeatPumpConfig(pydantic.BaseModel):
    name: str = "heat-pump"

    electric_power_mw: float
    cop: float


class HeatPumpOneInterval(AssetOneInterval):
    cfg: HeatPumpConfig

    electric_load_mwh: pulp.LpVariable
    electric_load_binary: pulp.LpVariable
    low_temperature_load_mwh: pulp.LpVariable
    high_temperature_generation_mwh: pulp.LpVariable


class HeatPump:
    def __init__(self, electric_power_mw: float, cop: float):
        self.cfg = HeatPumpConfig(electric_power_mw=electric_power_mw, cop=cop)
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
                # up=self.idata.low_temperature_load_mwh[i],
            ),
            high_temperature_generation_mwh=optimizer.continuous(
                f"high_temperature_generation_mwh,{name}",
                low=0,
                #  could set the upper bound here
                #  but the balance should take care of it
                #  same for the `up` for lt load
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
        pass

    def optimize(
        self,
        electricity_prices: list[float] | np.ndarray,
        electricity_carbon_intensities: float | list[float] | np.ndarray | None = None,
        gas_prices: float | list[float] | np.ndarray | None = None,
        freq_mins: int = defaults.freq_mins,
        objective: str = "price",
        verbose: bool = True,
        flags: Flags = Flags(),
        high_temperature_load_mwh: float | list[float] | np.ndarray | None = None,
        low_temperature_load_mwh: float | list[float] | np.ndarray | None = None,
        #  this is a special one
        low_temperature_generation_mwh: float | list[float] | np.ndarray | None = None,
    ) -> "epl.results.SimulationResult":
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
        print(f"{self.idata.low_temperature_generation_mwh=}")

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

            self.ivars.append([
                self.one_interval(self.optimizer, i, freq, flags=flags),
                self.boiler.one_interval(self.optimizer, i, freq),
                self.spill.one_interval(self.optimizer, i, freq),
                self.valve.one_interval(self.optimizer, i, freq),
            ])

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
            results,
            pathlib.Path(path),
            asset_name=self.cfg.name
        )
