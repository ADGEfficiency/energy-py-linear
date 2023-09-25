"""Battery asset for optimizing battery dispatch for price or carbon arbitrage."""
import pathlib

import numpy as np
import pulp
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval
from energypylinear.defaults import defaults
from energypylinear.flags import Flags
from energypylinear.freq import Freq
from energypylinear.optimizer import Optimizer


def setup_initial_final_charge(
    initial_charge_mwh: float, final_charge_mwh: float | None, capacity_mwh: float
) -> tuple[float, float]:
    """Processes the battery config for initial and final charge."""
    initial_charge_mwh = min(initial_charge_mwh, capacity_mwh)
    final_charge_mwh = (
        initial_charge_mwh
        if final_charge_mwh is None
        else min(final_charge_mwh, capacity_mwh)
    )
    return initial_charge_mwh, final_charge_mwh


class BatteryConfig(pydantic.BaseModel):
    """Battery asset configuration."""

    name: str
    power_mw: float
    capacity_mwh: float
    efficiency_pct: float
    initial_charge_mwh: float = 0.0
    final_charge_mwh: float | None = 0.0
    freq_mins: int

    @pydantic.validator("name")
    def check_name(cls, name: str) -> str:
        """Ensure we can identify this asset correctly."""

        assert "battery" in name
        return name


class BatteryOneInterval(AssetOneInterval):
    """Battery asset data for a single interval."""

    cfg: BatteryConfig
    electric_charge_mwh: pulp.LpVariable
    electric_charge_binary: pulp.LpVariable | int
    electric_discharge_mwh: pulp.LpVariable
    electric_discharge_binary: pulp.LpVariable | int
    electric_loss_mwh: pulp.LpVariable
    initial_charge_mwh: pulp.LpVariable
    final_charge_mwh: pulp.LpVariable
    efficiency_pct: float


def constrain_only_charge_or_discharge(
    optimizer: Optimizer,
    battery: AssetOneInterval,
    flags: Flags,
) -> None:
    """Constrain battery to only charge or discharge.

    Usually flagged off - slows things down a lot (~2x as slow).  Instead of forcing only charge or
    discharge, the objective function just takes the difference to calculate net charge.
    """
    assert isinstance(battery, BatteryOneInterval)
    if flags.include_charge_discharge_binary_variables:
        optimizer.constrain_max(
            battery.electric_charge_mwh,
            battery.electric_charge_binary,
            battery.cfg.capacity_mwh,
        )
        optimizer.constrain_max(
            battery.electric_discharge_mwh,
            battery.electric_discharge_binary,
            battery.cfg.capacity_mwh,
        )
        optimizer.constrain(
            battery.electric_charge_binary + battery.electric_discharge_binary <= 1
        )


def constrain_battery_electricity_balance(
    optimizer: Optimizer, battery: AssetOneInterval
) -> None:
    """Constrain energy balance in a single interval - also calculates losses."""
    assert isinstance(battery, BatteryOneInterval)
    optimizer.constrain(
        battery.initial_charge_mwh
        + battery.electric_charge_mwh
        - battery.electric_discharge_mwh
        - battery.electric_loss_mwh
        == battery.final_charge_mwh
    )
    optimizer.constrain(
        battery.electric_loss_mwh
        == battery.electric_charge_mwh * (1 - battery.efficiency_pct)
    )


def constrain_connection_batteries_between_intervals(
    optimizer: Optimizer,
    batteries: list[list[AssetOneInterval]],
) -> None:
    """Constrain battery dispatch between two adjacent intervals."""

    #  if in first interval, do nothing
    #  could also do something based on `i` here...
    if len(batteries) < 2:
        return None

    else:
        old = batteries[-2]
        new = batteries[-1]
        for alt, neu in zip(old, new, strict=True):
            assert isinstance(alt, BatteryOneInterval)
            assert isinstance(neu, BatteryOneInterval)
            optimizer.constrain(alt.final_charge_mwh == neu.initial_charge_mwh)


def constrain_initial_final_charge(
    optimizer: Optimizer, initial: AssetOneInterval, final: AssetOneInterval
) -> None:
    """Constrain the battery state of charge at the start and end of the simulation."""
    assert isinstance(initial, BatteryOneInterval)
    assert isinstance(final, BatteryOneInterval)

    optimizer.constrain(initial.initial_charge_mwh == initial.cfg.initial_charge_mwh)
    optimizer.constrain(final.final_charge_mwh == final.cfg.final_charge_mwh)


class Battery:
    """Battery asset - handles optimization and plotting of results over many intervals.

    Args:
        power_mw: the maximum power output of the battery in mega-watts, used for both charge and discharge.
        capacity_mwh: battery capacity in mega-watt hours.
        efficiency_pct: round-trip efficiency of the battery, with a default value of 90% efficient.
        name: the asset name.
        electricity_prices: the price of electricity in each interval.
        electricity_carbon_intensities: carbon intensity of electricity in each interval.
        initial_charge_mwh: initial charge state of the battery in mega-watt hours.
        final_charge_mwh: final charge state of the battery in mega-watt hours.
    """

    def __init__(
        self,
        power_mw: float = 2.0,
        capacity_mwh: float = 4.0,
        efficiency_pct: float = 0.9,
        name: str = "battery",
        electricity_prices: float | list[float] | np.ndarray | None = None,
        electricity_carbon_intensities: float | list[float] | np.ndarray | None = None,
        initial_charge_mwh: float = 0.0,
        final_charge_mwh: float | None = None,
        freq_mins: int = defaults.freq_mins,
    ):
        """Initializes the asset."""

        initial_charge_mwh, final_charge_mwh = setup_initial_final_charge(
            initial_charge_mwh, final_charge_mwh, capacity_mwh
        )

        self.cfg = BatteryConfig(
            name=name,
            power_mw=power_mw,
            capacity_mwh=capacity_mwh,
            efficiency_pct=efficiency_pct,
            initial_charge_mwh=initial_charge_mwh,
            final_charge_mwh=final_charge_mwh,
            freq_mins=freq_mins,
        )

        if electricity_prices is not None or electricity_carbon_intensities is not None:
            assets = [self, epl.Spill()]
            self.site = epl.Site(
                assets=assets,
                electricity_prices=electricity_prices,
                electricity_carbon_intensities=electricity_carbon_intensities,
            )

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<energypylinear.Battery {self.cfg.power_mw=} {self.cfg.capacity_mwh=}>"

    def one_interval(
        self, optimizer: Optimizer, i: int, freq: Freq, flags: Flags
    ) -> BatteryOneInterval:
        """Generate linear program data for one interval."""
        return BatteryOneInterval(
            cfg=self.cfg,
            electric_charge_mwh=optimizer.continuous(
                f"{self.cfg.name}-charge_mwh-{i}", up=freq.mw_to_mwh(self.cfg.power_mw)
            ),
            electric_discharge_mwh=optimizer.continuous(
                f"{self.cfg.name}-discharge_mwh-{i}",
                up=freq.mw_to_mwh(self.cfg.power_mw),
            ),
            electric_charge_binary=optimizer.binary(
                f"{self.cfg.name}-charge_binary-{i}"
            )
            if flags.include_charge_discharge_binary_variables
            else 0,
            electric_discharge_binary=optimizer.binary(
                f"{self.cfg.name}-discharge_binary-{i}"
            )
            if flags.include_charge_discharge_binary_variables
            else 0,
            electric_loss_mwh=optimizer.continuous(
                f"{self.cfg.name}-electric_loss_mwh-{i}"
            ),
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
        ivars: "epl.interval_data.IntervalVars",
        i: int,
        freq: Freq,
        flags: Flags = Flags(),
    ) -> None:
        """Constrain asset within an interval."""
        battery = ivars.filter_objective_variables(
            BatteryOneInterval, i=-1, asset_name=self.cfg.name
        )[0][0]
        constrain_only_charge_or_discharge(optimizer, battery, flags)
        constrain_battery_electricity_balance(optimizer, battery)

        #  this is one battery asset, all intervals
        #  TODO maybe refactor into the after intervals?
        #  bit of a weird case really
        all_batteries = ivars.filter_objective_variables(
            BatteryOneInterval, i=None, asset_name=self.cfg.name
        )
        assert isinstance(all_batteries, list)
        constrain_connection_batteries_between_intervals(optimizer, all_batteries)

    def constrain_after_intervals(
        self,
        optimizer: Optimizer,
        ivars: "epl.interval_data.IntervalVars",
    ) -> None:
        """Constrain asset after all intervals."""
        initial = ivars.filter_objective_variables(
            BatteryOneInterval, i=0, asset_name=self.cfg.name
        )[0][0]
        final = ivars.filter_objective_variables(
            BatteryOneInterval, i=-1, asset_name=self.cfg.name
        )[0][0]
        constrain_initial_final_charge(optimizer, initial, final)

    def optimize(
        self,
        objective: str = "price",
        verbose: bool = True,
        flags: Flags = Flags(),
    ) -> "epl.SimulationResult":
        """Optimize the asset.

        Args:
            freq_mins: the size of an interval in minutes.
            objective: the optimization objective - either "price" or "carbon".
            flags: boolean flags to change simulation and results behaviour.
            verbose: level of printing.

        Returns:
            epl.results.SimulationResult
        """
        return self.site.optimize(
            freq_mins=self.cfg.freq_mins,
            objective=objective,
            flags=flags,
            verbose=verbose,
        )

    def plot(self, results: "epl.SimulationResult", path: pathlib.Path | str) -> None:
        """Plot simulation results."""
        return epl.plot.plot_battery(results, pathlib.Path(path))
