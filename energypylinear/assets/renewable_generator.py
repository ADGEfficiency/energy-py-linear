"""
Renewable Generator asset.

Suitable for modelling either turndownable wind or solar."""

import numpy as np
import pydantic
from pydantic import ConfigDict

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval
from energypylinear.defaults import defaults
from energypylinear.flags import Flags


class RenewableGeneratorIntervalData(pydantic.BaseModel):
    """Renewable Generator interval data."""

    electric_generation_mwh: np.ndarray | list[float] | float
    idx: list[int] | np.ndarray = pydantic.Field(default_factory=list)

    @pydantic.validator("idx", always=True)
    def create_idx(cls, _: list, values: dict) -> np.ndarray:
        """Creates an integer index."""
        return np.arange(len(values["electric_generation_mwh"]))

    @pydantic.validator("electric_generation_mwh", pre=True, always=True)
    def handle_single_float(cls, value: float | list | np.ndarray) -> list | np.ndarray:
        """Handles case where we want a single value broadcast to the length
        of the interval data.
        """
        if isinstance(value, float):
            return [value]
        return np.array(value)

    @pydantic.validator("electric_generation_mwh", always=True)
    def validate_greater_zero(cls, value: np.ndarray | list) -> np.ndarray | list:
        """Handles case where we want a single value broadcast to the length
        of the interval data.
        """
        assert np.array(value).min() >= 0.0
        return value

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RenewableGeneratorConfig(pydantic.BaseModel):
    """Renewable Generator asset configuration."""

    name: str
    electric_generation_lower_bound_pct: float = pydantic.Field(..., ge=0, le=1)
    interval_data: RenewableGeneratorIntervalData
    freq_mins: int


class RenewableGeneratorOneInterval(AssetOneInterval):
    """Contains linear program data for a single interval."""

    cfg: RenewableGeneratorConfig
    electric_generation_mwh: epl.LpVariable


class RenewableGenerator(epl.Asset):
    """Renewable Generator asset.

    Handles optimization and plotting of results over many intervals.

    This asset is suitable for modelling either wind or solar generatation.

    The electricity generation can be controlled with relative or absolute
    upper and lower bounds on the available generation in an interval.

    The upper bound for the electricity generation is limited by the
    `electric_generation_mwh` interval data input in
    `RenewableGenerator.optimize`.  This input is the amount of generation
    available from the wind or sun.

    The `electric_generation_mwh` input interval is the generation available
    to the site from the renewable resource - this should be after any limits or
    losses, such as inverter sizing and efficiency for solar or generator losses
    for wind.
    """

    def __init__(
        self,
        electric_generation_mwh: np.ndarray | list[float] | float,
        electricity_prices: np.ndarray | list[float] | np.ndarray | None = None,
        export_electricity_prices: np.ndarray | list[float] | np.ndarray | None = None,
        electricity_carbon_intensities: np.ndarray
        | list[float]
        | np.ndarray
        | None = None,
        electric_load_mwh: np.ndarray | list[float] | np.ndarray | None = None,
        electric_generation_lower_bound_pct: float = 1.0,
        name: str = "renewable-generator",
        freq_mins: int = defaults.freq_mins,
    ) -> None:
        """Initializes the asset."""
        self.cfg = RenewableGeneratorConfig(
            name=name,
            electric_generation_lower_bound_pct=electric_generation_lower_bound_pct,
            interval_data=RenewableGeneratorIntervalData(
                electric_generation_mwh=electric_generation_mwh,
            ),
            freq_mins=freq_mins,
        )

        if electricity_prices is not None or electricity_carbon_intensities is not None:
            assets = [self]

            self.site = epl.Site(
                assets=assets,
                electric_load_mwh=electric_load_mwh,
                electricity_prices=electricity_prices,
                export_electricity_prices=export_electricity_prices,
                electricity_carbon_intensities=electricity_carbon_intensities,
                freq_mins=self.cfg.freq_mins,
            )

    def __repr__(self) -> str:
        """A string representation of self."""
        return "<energypylinear.RenewableGenerator>"

    def one_interval(
        self, optimizer: "epl.Optimizer", i: int, freq: "epl.Freq", flags: "epl.Flags"
    ) -> RenewableGeneratorOneInterval:
        """Create asset data for a single interval."""
        name = f"i:{i},asset:{self.cfg.name}"
        assert isinstance(self.cfg.interval_data.electric_generation_mwh, np.ndarray)
        return RenewableGeneratorOneInterval(
            cfg=self.cfg,
            electric_generation_mwh=optimizer.continuous(
                f"electric_generation_mwh,{name}",
                low=self.cfg.interval_data.electric_generation_mwh[i]
                * self.cfg.electric_generation_lower_bound_pct,
                up=self.cfg.interval_data.electric_generation_mwh[i],
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
        """Constrain optimization within a single interval."""
        pass

    def constrain_after_intervals(
        self,
        optimizer: "epl.Optimizer",
        ivars: "epl.IntervalVars",
    ) -> None:
        """Constrain asset after all intervals."""
        pass

    def optimize(
        self,
        objective: str = "price",
        verbose: bool = True,
        flags: Flags = Flags(),
        optimizer_config: "epl.OptimizerConfig" = epl.optimizer.OptimizerConfig(),
    ) -> "epl.SimulationResult":
        """Optimize the asset.

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
