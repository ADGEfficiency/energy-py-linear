"""Battery asset for optimizing battery dispatch for price or carbon arbitrage."""
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


class BatteryConfig(pydantic.BaseModel):
    """Battery asset configuration."""

    name: str
    power_mw: float
    capacity_mwh: float
    efficiency_pct: float
    initial_charge_mwh: float = 0
    final_charge_mwh: float = 0

    @pydantic.validator("name")
    def check_name(cls, name: str) -> str:
        """Check that name includes battery."""
        assert "battery" in name
        return name


class BatteryOneInterval(AssetOneInterval):
    """Battery asset data for a single interval"""

    cfg: BatteryConfig
    charge_mwh: pulp.LpVariable
    charge_binary: typing.Union[pulp.LpVariable, int]
    discharge_mwh: pulp.LpVariable
    discharge_binary: typing.Union[pulp.LpVariable, int]
    losses_mwh: pulp.LpVariable
    initial_charge_mwh: pulp.LpVariable
    final_charge_mwh: pulp.LpVariable
    efficiency_pct: float


def constrain_only_charge_or_discharge(
    optimizer: Optimizer,
    vars: collections.defaultdict,
    flags: Flags,
) -> None:
    """Constrain battery to only charge or discharge.

    Usually flagged off - slows things down a lot (~2x as slow).
    """
    if flags.include_charge_discharge_binary_variables:
        batteries = epl.utils.filter_assets(vars, "battery")
        for battery in batteries:
            optimizer.constrain_max(
                battery.charge_mwh, battery.charge_binary, battery.cfg.capacity_mwh
            )
            optimizer.constrain_max(
                battery.discharge_mwh,
                battery.discharge_binary,
                battery.cfg.capacity_mwh,
            )
            optimizer.constrain(battery.charge_binary + battery.discharge_binary <= 1)


def constrain_battery_electricity_balance(
    optimizer: Optimizer, vars: collections.defaultdict
) -> None:
    """Constrain energy balance in a single interval - also calculates losses."""

    assets = vars["assets"][-1]
    batteries = [a for a in assets if isinstance(a, epl.battery.BatteryOneInterval)]

    for battery in batteries:
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
    """Constrain battery dispatch between two adjacent intervals."""
    batteries = epl.utils.filter_all_assets(vars, "battery")

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
) -> None:
    """Constrain the battery state of charge at the start and end of the simulation."""

    assets = vars["assets"][0]
    first = [a for a in assets if isinstance(a, epl.battery.BatteryOneInterval)]
    for battery in first:
        optimizer.constrain(
            battery.initial_charge_mwh == battery.cfg.initial_charge_mwh
        )

    assets = vars["assets"][-1]
    last = [a for a in assets if isinstance(a, epl.battery.BatteryOneInterval)]
    for battery in last:
        optimizer.constrain(battery.final_charge_mwh == battery.cfg.final_charge_mwh)


class Battery:
    """Battery asset - handles optimization and plotting of results over many intervals.

    Args:
        power_mw - the maximum power output of the battery in mega-watts, used for both charge and discharge.
        capacity_mwh - battery capacity in mega-watt hours.
        efficiency - round-trip efficiency of the battery, with a default value of 90% efficient.
        battery_name parameter represents the name of the battery, with a default value of "battery".
    """

    def __init__(
        self,
        power_mw: float = 2.0,
        capacity_mwh: float = 4.0,
        efficiency: float = 0.9,
        name: str = "battery",
    ):
        """Initialize a Battery asset model."""
        self.cfg = BatteryConfig(
            name=name,
            power_mw=power_mw,
            capacity_mwh=capacity_mwh,
            efficiency_pct=efficiency,
        )

    def setup_initial_final_charge(
        self, initial_charge_mwh: float, final_charge_mwh: typing.Optional[float]
    ) -> None:
        """Processes the options for initial and final charge."""
        self.cfg.initial_charge_mwh = min(initial_charge_mwh, self.cfg.capacity_mwh)
        self.cfg.final_charge_mwh = (
            self.cfg.initial_charge_mwh
            if final_charge_mwh is None
            else min(final_charge_mwh, self.cfg.capacity_mwh)
        )

    def one_interval(
        self, optimizer: Optimizer, i: int, freq: Freq, flags: Flags
    ) -> BatteryOneInterval:
        """Create Battery asset data for a single interval."""
        return BatteryOneInterval(
            cfg=self.cfg,
            charge_mwh=optimizer.continuous(
                f"{self.cfg.name}-charge_mwh-{i}", up=freq.mw_to_mwh(self.cfg.power_mw)
            ),
            discharge_mwh=optimizer.continuous(
                f"{self.cfg.name}-discharge_mwh-{i}",
                up=freq.mw_to_mwh(self.cfg.power_mw),
            ),
            charge_binary=optimizer.binary(f"{self.cfg.name}-charge_binary-{i}")
            if flags.include_charge_discharge_binary_variables
            else 0,
            discharge_binary=optimizer.binary(f"{self.cfg.name}-discharge_binary-{i}")
            if flags.include_charge_discharge_binary_variables
            else 0,
            losses_mwh=optimizer.continuous(f"{self.cfg.name}-losses_mwh-{i}"),
            initial_charge_mwh=optimizer.continuous(
                f"{self.cfg.name}-initial_charge_mwh-{i}",
                low=0,
                up=self.cfg.capacity_mwh,
            ),
            final_charge_mwh=optimizer.continuous(
                f"{self.cfg.name}-final_charge_mwh-{i}", low=0, up=self.cfg.capacity_mwh
            ),
            efficiency_pct=self.cfg.efficiency_pct,
        )

    def constrain_within_interval(
        self,
        optimizer: Optimizer,
        vars: collections.defaultdict,
        flags: Flags,
        #  for freq - maybe better to have the freq here - idk...
        **kwargs: typing.Any,
    ) -> None:
        """Constrain battery dispatch within a single interval"""
        constrain_only_charge_or_discharge(optimizer, vars, flags)
        constrain_battery_electricity_balance(optimizer, vars)
        constrain_connection_batteries_between_intervals(optimizer, vars)

    def constrain_after_intervals(
        self,
        optimizer: Optimizer,
        vars: collections.defaultdict,
    ) -> None:
        """Constrain battery dispatch after all interval asset models are created."""
        constrain_initial_final_charge(optimizer, vars)

    def optimize(
        self,
        electricity_prices: np.ndarray,
        gas_prices: typing.Union[None, np.ndarray] = None,
        electricity_carbon_intensities: typing.Union[None, np.ndarray] = None,
        freq_mins: int = defaults.freq_mins,
        initial_charge_mwh: float = 0.0,
        final_charge_mwh: typing.Union[float, None] = None,
        objective: str = "price",
        flags: Flags = Flags(),
        verbose: int = 0,
    ) -> "epl.results.SimulationResult":
        """Optimize the battery's dispatch using a mixed-integer linear program.

        Args:
            electricity_prices: the price of electricity in each interval.
            gas_prices: the prices of natural gas, used in CHP and boilers in each interval.
            electricity_carbon_intensities: carbon intensity of electricity in each interval.
            high_temperature_load_mwh: high temperature load of the site in mega-watt hours.
            low_temperature_load_mwh: low temperature load of the site in mega-watt hours.
            freq_mins: the size of an interval in minutes.
            initial_charge_mwh: initial charge state of the battery in mega-watt hours.
            final_charge_mwh: final charge state of the battery in mega-watt hours.
            objective: the optimization objective - either "price" or "carbon".
            flags: boolean flags to change simulation and results behaviour.
            verbose: level of printing.

        Returns:
            epl.results.SimulationResult
        """
        self.optimizer = Optimizer()
        freq = Freq(freq_mins)
        interval_data = epl.interval_data.IntervalData(
            electricity_prices=electricity_prices,
            gas_prices=gas_prices,
            electricity_carbon_intensities=electricity_carbon_intensities,
        )
        self.site = epl.Site()
        self.spill = epl.spill.Spill()

        self.setup_initial_final_charge(initial_charge_mwh, final_charge_mwh)

        vars: collections.defaultdict[str, typing.Any] = collections.defaultdict(list)
        for i in interval_data.idx:
            vars["sites"].append(
                self.site.one_interval(self.optimizer, self.site.cfg, i, freq)
            )
            vars["spills"].append(self.spill.one_interval(self.optimizer, i, freq))
            vars["assets"].append([self.one_interval(self.optimizer, i, freq, flags)])

            self.site.constrain_within_interval(self.optimizer, vars, interval_data, i)
            self.constrain_within_interval(self.optimizer, vars, flags=flags)

        self.constrain_after_intervals(self.optimizer, vars)

        assert len(interval_data.idx) == len(vars["assets"]) == len(vars["sites"])

        objective_fn = epl.objectives[objective]
        self.optimizer.objective(objective_fn(self.optimizer, vars, interval_data))
        status = self.optimizer.solve(verbose=verbose)

        self.interval_data = interval_data
        return epl.results.extract_results(
            interval_data, vars, feasible=status.feasible
        )

    def plot(
        self,
        results: "epl.results.SimulationResult",
        path: typing.Union[pathlib.Path, str],
    ) -> None:
        """Plot simulation results."""
        return epl.plot.plot_battery(results, pathlib.Path(path))
