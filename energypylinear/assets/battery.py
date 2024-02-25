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
    """Determine initial and final battery state of charge, taking into account battery capacity.

    Args:
        initial_charge_mwh: state of charge at the start of the simulation in megawatt hours.
        final_charge_mwh: state of charge at the end of the simulation in megawatt hours. Defaults to the initial charge.
        capacity_mwh: battery capacity in megawatt hours.

    Returns:
        A tuple of initial and final charge, both in megawatt hours.
    """
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
    charge_power_mw: float
    discharge_power_mw: float
    capacity_mwh: float
    efficiency_pct: float
    initial_charge_mwh: float = 0.0
    final_charge_mwh: float | None = 0.0
    freq_mins: int

    @pydantic.field_validator("name")
    @classmethod
    def check_name(cls, name: str) -> str:
        """Ensure we can identify this asset correctly.

        Args:
            name: asset name.

        Returns:
            The asset name.
        """
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
    electric_initial_charge_mwh: pulp.LpVariable
    electric_final_charge_mwh: pulp.LpVariable
    efficiency_pct: float


def constrain_only_charge_or_discharge(
    optimizer: Optimizer,
    one_interval: AssetOneInterval,
) -> None:
    """Constrain battery to only charge or discharge.

    Args:
        optimizer: The linear program optimizer.
        one_interval: Linear program variables for a single interval.
    """
    assert isinstance(one_interval, BatteryOneInterval)
    optimizer.constrain_max(
        one_interval.electric_charge_mwh,
        one_interval.electric_charge_binary,
        one_interval.cfg.capacity_mwh,
    )
    optimizer.constrain_max(
        one_interval.electric_discharge_mwh,
        one_interval.electric_discharge_binary,
        one_interval.cfg.capacity_mwh,
    )
    optimizer.constrain(
        one_interval.electric_charge_binary + one_interval.electric_discharge_binary
        <= 1
    )


def constrain_battery_electricity_balance(
    optimizer: Optimizer, one_interval: AssetOneInterval
) -> None:
    """Constrain battery energy balance in a single interval.

    Calculates losses as a percentage of the battery charge.

    Args:
        optimizer: Linear program optimizer.
        one_interval: Linear program variables for a single interval.
    """
    assert isinstance(one_interval, BatteryOneInterval)
    optimizer.constrain(
        one_interval.electric_initial_charge_mwh
        + one_interval.electric_charge_mwh
        - one_interval.electric_discharge_mwh
        - one_interval.electric_loss_mwh
        == one_interval.electric_final_charge_mwh
    )
    optimizer.constrain(
        one_interval.electric_loss_mwh
        == one_interval.electric_charge_mwh * (1 - one_interval.efficiency_pct)
    )


def constrain_connection_batteries_between_intervals(
    optimizer: Optimizer,
    two_intervals: list[list[AssetOneInterval]],
) -> None:
    """Constrain battery between two adjacent intervals.

    Args:
        optimizer: Linear program optimizer.
        two_intervals: Linear program variables for two intervals.
    """
    #  if in first interval, do nothing, could also do something based on `i` here...
    if len(two_intervals) < 2:
        return

    old = two_intervals[-2]
    new = two_intervals[-1]
    for alt, neu in zip(old, new, strict=True):
        assert isinstance(alt, BatteryOneInterval)
        assert isinstance(neu, BatteryOneInterval)
        optimizer.constrain(
            alt.electric_final_charge_mwh == neu.electric_initial_charge_mwh
        )


def constrain_initial_final_charge(
    optimizer: Optimizer, initial: AssetOneInterval, final: AssetOneInterval
) -> None:
    """Constrain the battery state of charge at the start and end of the simulation.

    Args:
        optimizer: Linear program optimizer.
        initial: Linear program variables for the first interval.
        final: Linear program variables for the last interval.
    """
    assert isinstance(initial, BatteryOneInterval)
    assert isinstance(final, BatteryOneInterval)
    optimizer.constrain(
        initial.electric_initial_charge_mwh == initial.cfg.initial_charge_mwh
    )
    optimizer.constrain(final.electric_final_charge_mwh == final.cfg.final_charge_mwh)


class Battery(epl.Asset):
    """Electric battery asset, able to charge and discharge electricity."""

    def __init__(
        self,
        power_mw: float = 2.0,
        discharge_power_mw: float | None = None,
        capacity_mwh: float = 4.0,
        efficiency_pct: float = 0.9,
        name: str = "battery",
        electricity_prices: np.ndarray | list[float] | float | None = None,
        export_electricity_prices: np.ndarray | list[float] | float | None = None,
        electricity_carbon_intensities: np.ndarray | list[float] | float | None = None,
        initial_charge_mwh: float = 0.0,
        final_charge_mwh: float | None = None,
        freq_mins: int = defaults.freq_mins,
    ):
        """Initialize the asset.

        Args:
            power_mw: Maximum charge rate in megawatts. Will define both the charge and discharge rate if `discharge_power_mw` is None.
            discharge_power_mw: Maximum discharge rate in megawatts.
            capacity_mwh: Battery capacity in megawatt hours.
            efficiency_pct: Round-trip efficiency of the battery.
            name: The asset name.
            electricity_prices: The price of import electricity in each interval. Will define both import and export prices if `export_electricity_prices` is None.
            export_electricity_prices: The price of export electricity in each interval.
            electricity_carbon_intensities: Carbon intensity of electricity in each interval.
            initial_charge_mwh: Initial charge state of the battery in megawatt hours.
            final_charge_mwh: Final charge state of the battery in megawatt hours.
            freq_mins: length of the simulation intervals in minutes.
        """
        initial_charge_mwh, final_charge_mwh = setup_initial_final_charge(
            initial_charge_mwh, final_charge_mwh, capacity_mwh
        )

        if discharge_power_mw is None:
            discharge_power_mw = power_mw
        assert discharge_power_mw

        self.cfg = BatteryConfig(
            name=name,
            charge_power_mw=power_mw,
            discharge_power_mw=discharge_power_mw,
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
                export_electricity_prices=export_electricity_prices,
                electricity_carbon_intensities=electricity_carbon_intensities,
                freq_mins=self.cfg.freq_mins,
            )

    def __repr__(self) -> str:
        """Return a string representation of self.

        Returns:
            A string representation of self.
        """
        return f"<energypylinear.Battery {self.cfg.charge_power_mw=} {self.cfg.capacity_mwh=}>"

    def one_interval(
        self, optimizer: Optimizer, i: int, freq: Freq, flags: Flags = Flags()
    ) -> BatteryOneInterval:
        """Generate linear program data for one interval.

        Args:
            optimizer: Linear program optimizer.
            i: Integer index of the current interval.
            freq: Interval frequency.
            flags: Boolean flags to change simulation and results behaviour.

        Returns:
            Linear program variables for a single interval.
        """
        return BatteryOneInterval(
            cfg=self.cfg,
            electric_charge_mwh=optimizer.continuous(
                f"{self.cfg.name}-electric_charge_mwh-{i}",
                up=freq.mw_to_mwh(self.cfg.charge_power_mw),
            ),
            electric_discharge_mwh=optimizer.continuous(
                f"{self.cfg.name}-electric_discharge_mwh-{i}",
                up=freq.mw_to_mwh(self.cfg.discharge_power_mw),
            ),
            electric_charge_binary=optimizer.binary(
                f"{self.cfg.name}-electric_charge_binary-{i}"
            ),
            electric_discharge_binary=optimizer.binary(
                f"{self.cfg.name}-electric_discharge_binary-{i}"
            ),
            electric_loss_mwh=optimizer.continuous(
                f"{self.cfg.name}-electric_loss_mwh-{i}"
            ),
            electric_initial_charge_mwh=optimizer.continuous(
                f"{self.cfg.name}-electric_initial_charge_mwh-{i}",
                low=0,
                up=self.cfg.capacity_mwh,
            ),
            electric_final_charge_mwh=optimizer.continuous(
                f"{self.cfg.name}-electric_final_charge_mwh-{i}",
                low=0,
                up=self.cfg.capacity_mwh,
            ),
            efficiency_pct=self.cfg.efficiency_pct,
        )

    def constrain_within_interval(
        self,
        optimizer: Optimizer,
        ivars: "epl.IntervalVars",
        i: int,
        freq: Freq,
        flags: Flags = Flags(),
    ) -> None:
        """Constrain asset within an interval.

        Args:
            optimizer: Linear program optimizer.
            ivars: Linear program variables.
            i: Integer index of the current interval.
            freq: Interval frequency.
            flags: Boolean flags to change simulation and results behaviour.
        """
        battery = ivars.filter_objective_variables(
            instance_type=BatteryOneInterval, i=-1, asset_name=self.cfg.name
        )[0]
        constrain_only_charge_or_discharge(optimizer, battery)
        constrain_battery_electricity_balance(optimizer, battery)

        #  one battery asset, all intervals
        all_batteries = ivars.filter_objective_variables_all_intervals(
            instance_type=BatteryOneInterval, asset_name=self.cfg.name
        )
        assert isinstance(all_batteries, list)
        constrain_connection_batteries_between_intervals(optimizer, all_batteries)

    def constrain_after_intervals(
        self,
        optimizer: Optimizer,
        ivars: "epl.IntervalVars",
    ) -> None:
        """Constrain asset after all intervals.

        Args:
            optimizer: Linear program optimizer.
            ivars: Linear program variables.
        """
        initial = ivars.filter_objective_variables(
            instance_type=BatteryOneInterval, i=0, asset_name=self.cfg.name
        )[0]
        final = ivars.filter_objective_variables(
            instance_type=BatteryOneInterval, i=-1, asset_name=self.cfg.name
        )[0]
        assert isinstance(initial, BatteryOneInterval)
        assert isinstance(final, BatteryOneInterval)
        constrain_initial_final_charge(optimizer, initial, final)

    def optimize(
        self,
        objective: "str | dict | epl.objectives.CustomObjectiveFunction" = "price",
        verbose: int | bool = 2,
        flags: Flags = Flags(),
        optimizer_config: "epl.OptimizerConfig | dict" = epl.optimizer.OptimizerConfig(),
    ) -> "epl.SimulationResult":
        """Optimize the asset.

        Args:
            objective: The optimization objective - either "price" or "carbon".
            verbose: Level of printing.
            flags: Boolean flags to change simulation and results behaviour.
            optimizer_config: Configuration options for the optimizer.

        Returns:
            The simulation results.
        """
        return self.site.optimize(
            objective=objective,
            flags=flags,
            verbose=verbose,
            optimizer_config=optimizer_config,
        )

    def plot(self, results: "epl.SimulationResult", path: pathlib.Path | str) -> None:
        """Plot simulation results.

        Args:
            results: The simulation results.
            path: Directory to save the plots to.

        Returns:
            None.
        """
        return epl.plot.plot_battery(results, pathlib.Path(path))
