"""Boiler asset for optimizing dispatch of gas fired boilers"""

import pulp
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval
from energypylinear.defaults import defaults


class BoilerConfig(pydantic.BaseModel):
    """Boiler configuration."""

    name: str
    high_temperature_generation_max_mw: float = 0
    high_temperature_generation_min_mw: float = 0
    high_temperature_efficiency_pct: float = pydantic.Field(gt=0.0, default=0.8, le=1.0)

    @pydantic.field_validator("name")
    @classmethod
    def check_name(cls, name: str) -> str:
        """Ensure we can identify this asset correctly."""
        assert "boiler" in name
        return name


class BoilerOneInterval(AssetOneInterval):
    """Boiler data for a single interval."""

    high_temperature_generation_mwh: pulp.LpVariable
    gas_consumption_mwh: pulp.LpVariable
    binary: pulp.LpVariable
    cfg: BoilerConfig


class Boiler(epl.Asset):
    """Boiler asset - generates high temperature heat from natural gas."""

    def __init__(
        self,
        name: str = "boiler",
        high_temperature_generation_max_mw: float = defaults.boiler_high_temperature_generation_max_mw,
        high_temperature_generation_min_mw: float = 0,
        high_temperature_efficiency_pct: float = defaults.boiler_efficiency_pct,
    ):
        """Initialize the asset model."""
        self.cfg = BoilerConfig(
            name=name,
            high_temperature_generation_max_mw=high_temperature_generation_max_mw,
            high_temperature_generation_min_mw=high_temperature_generation_min_mw,
            high_temperature_efficiency_pct=high_temperature_efficiency_pct,
        )

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<energypylinear.Boiler {self.cfg.high_temperature_generation_max_mw=}>"

    def one_interval(
        self, optimizer: "epl.Optimizer", i: int, freq: "epl.Freq", flags: "epl.Flags"
    ) -> BoilerOneInterval:
        """Create asset data for a single interval."""
        return BoilerOneInterval(
            high_temperature_generation_mwh=optimizer.continuous(
                f"{self.cfg.name}-high_temperature_generation_mwh-{i}",
                low=freq.mw_to_mwh(self.cfg.high_temperature_generation_min_mw),
                up=freq.mw_to_mwh(self.cfg.high_temperature_generation_max_mw),
            ),
            binary=optimizer.binary(
                f"{self.cfg.name}-binary_mwh-{i}",
            ),
            gas_consumption_mwh=optimizer.continuous(
                f"{self.cfg.name}-gas_consumption_mwh-{i}"
            ),
            cfg=self.cfg,
        )

    def constrain_within_interval(
        self,
        optimizer: "epl.Optimizer",
        ivars: "epl.IntervalVars",
        i: int,
        freq: "epl.Freq",
        flags: "epl.Flags",
    ) -> None:
        """Constrain boiler for generation of high temperature heat."""
        boiler = ivars.filter_objective_variables(
            instance_type=BoilerOneInterval, i=-1, asset_name=self.cfg.name
        )[0]
        assert isinstance(boiler, BoilerOneInterval)
        optimizer.constrain(
            boiler.gas_consumption_mwh
            == boiler.high_temperature_generation_mwh
            * (1 / boiler.cfg.high_temperature_efficiency_pct)
        )
        optimizer.constrain_max(
            boiler.high_temperature_generation_mwh,
            boiler.binary,
            freq.mw_to_mwh(boiler.cfg.high_temperature_generation_max_mw),
        )
        optimizer.constrain_min(
            boiler.high_temperature_generation_mwh,
            boiler.binary,
            freq.mw_to_mwh(boiler.cfg.high_temperature_generation_min_mw),
        )

    def constrain_after_intervals(
        self, optimizer: "epl.Optimizer", ivars: "epl.IntervalVars"
    ) -> None:
        """Constrain the asset after all intervals."""
        return
