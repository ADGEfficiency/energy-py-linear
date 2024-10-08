"""
Renewable Generator asset.

Suitable for modelling either turndownable wind or solar.
"""

import typing

import numpy as np
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval
from energypylinear.defaults import defaults
from energypylinear.flags import Flags


class RenewableGeneratorIntervalData(pydantic.BaseModel):
    """Renewable Generator interval data."""

    electric_generation_mwh: np.ndarray | list[float] | float
    idx: list[int] | np.ndarray = pydantic.Field(default_factory=list)
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @pydantic.field_validator("electric_generation_mwh", mode="after")
    @classmethod
    def handle_single_float(cls, value: float | list | np.ndarray) -> list | np.ndarray:
        """
        Handle case where we want a single value broadcast to the length of the interval data.

        Args:
            value: The value to broadcast.

        Returns:
            The value as a list or np.ndarray.
        """
        if isinstance(value, float):
            return [value]
        return np.array(value)

    @pydantic.field_validator("electric_generation_mwh", mode="after")
    @classmethod
    def validate_greater_zero(cls, value: np.ndarray | list) -> np.ndarray | list:
        """
        Handle case where we want a single value broadcast to the length of the interval data.

        Args:
            value: The value to broadcast.

        Returns:
            The value as a list or np.ndarray.
        """
        assert np.array(value).min() >= 0.0
        return value

    @pydantic.model_validator(mode="after")
    def create_idx(self) -> "RenewableGeneratorIntervalData":
        """
        Create an integer index.

        Returns:
            The instance with an index.
        """
        assert isinstance(self.electric_generation_mwh, (np.ndarray, list))
        self.idx = np.arange(len(self.electric_generation_mwh))
        return self


class RenewableGeneratorConfig(pydantic.BaseModel):
    """Renewable Generator asset configuration."""

    name: str
    electric_generation_lower_bound_pct: float = pydantic.Field(..., ge=0, le=1)
    interval_data: RenewableGeneratorIntervalData
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
        assert ("wind" in name) or ("solar" in name) or ("renewable-generator" in name)
        return name


class RenewableGeneratorOneInterval(AssetOneInterval):
    """Contains linear program data for a single interval."""

    cfg: RenewableGeneratorConfig
    electric_generation_mwh: epl.LpVariable


class RenewableGenerator(epl.OptimizableAsset):
    """
    The Renewable Generator asset can generate electricity based on an available amount of electricity.

    This asset is suitable for modelling either wind or solar generatation.

    The electricity generation can be controlled with relative or absolute
    upper and lower bounds on the available generation in an interval.

    The upper bound for the electricity generation is limited by the `electric_generation_mwh` interval data
    input in `RenewableGenerator.optimize`. This input is the amount of generation available from the wind or sun.

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
        constraints: "list[epl.Constraint] | list[dict] | None" = None,
        include_spill: bool = False,
        **kwargs: typing.Any,
    ) -> None:
        """
        Initialize a Renewable Generator asset.

        Args:
            electric_generation_mwh: Available electricity generation from the renewable source.
            electricity_prices: The price of import electricity in each interval.
                Will define both import and export prices if `export_electricity_prices` is None.
            export_electricity_prices: The price of export electricity in each interval.
            electricity_carbon_intensities: Carbon intensity of electricity in each interval.
            electric_load_mwh: Electricity demand consumed by the site.
            electric_generation_lower_bound_pct: Sets how much the generator can dump available
                electricity.
            name: The asset name.
            freq_mins: length of the simulation intervals in minutes.
            constraints: Additional custom constraints to apply to the linear program.
            include_spill: Whether to include a spill asset in the site.
            kwargs: Extra keyword arguments attempted to be used as custom interval data.
        """
        self.cfg = RenewableGeneratorConfig(
            name=name,
            electric_generation_lower_bound_pct=electric_generation_lower_bound_pct,
            interval_data=RenewableGeneratorIntervalData(
                electric_generation_mwh=electric_generation_mwh,
            ),
            freq_mins=freq_mins,
        )

        if electricity_prices is not None or electricity_carbon_intensities is not None:
            assets: list[epl.Asset] = [self]
            if include_spill:
                assets.append(epl.Spill())

            self.site = epl.Site(
                assets=assets,
                electric_load_mwh=electric_load_mwh,
                electricity_prices=electricity_prices,
                export_electricity_prices=export_electricity_prices,
                electricity_carbon_intensities=electricity_carbon_intensities,
                freq_mins=self.cfg.freq_mins,
                constraints=constraints,
                **kwargs,
            )

    def __repr__(self) -> str:
        """
        Create a string representation of self.

        Returns:
            A string representation of self.
        """
        return "<energypylinear.RenewableGenerator>"

    def one_interval(
        self, optimizer: "epl.Optimizer", i: int, freq: "epl.Freq", flags: "epl.Flags"
    ) -> RenewableGeneratorOneInterval:
        """
        Create asset data for a single interval.

        Args:
            optimizer: Linear program optimizer.
            i: Integer index of the current interval.
            freq: Interval frequency.
            flags: Boolean flags to change simulation and results behaviour.

        Returns:
            Linear program variables for a single interval.
        """
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
        """
        Constrain asset within an interval.

        Args:
            optimizer: Linear program optimizer.
            ivars: Linear program variables.
            i: Integer index of the current interval.
            freq: Interval frequency.
            flags: Boolean flags to change simulation and results behaviour.
        """
        pass

    def constrain_after_intervals(
        self,
        optimizer: "epl.Optimizer",
        ivars: "epl.IntervalVars",
    ) -> None:
        """
        Constrain asset after all intervals.

        Args:
            optimizer: Linear program optimizer.
            ivars: Linear program variables.
        """
        pass

    def optimize(
        self,
        objective: "str | dict | epl.objectives.CustomObjectiveFunction" = "price",
        verbose: int | bool = 2,
        flags: Flags = Flags(),
        optimizer_config: "epl.OptimizerConfig | dict" = epl.optimizer.OptimizerConfig(),
    ) -> "epl.SimulationResult":
        """
        Optimize the asset.

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
