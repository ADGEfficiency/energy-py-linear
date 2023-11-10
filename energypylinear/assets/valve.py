"""Valve asset for allowing heat to flow from high to low temperature.

This allows high temperature heat generated by either gas boilers or
CHP generators to be used for low temperature heat consumption.
"""
import pulp
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval


class ValveConfig(pydantic.BaseModel):
    """Valve configuration."""

    name: str

    @pydantic.field_validator("name")
    @classmethod
    def check_name(cls, name: str) -> str:
        """Ensure we can identify this asset correctly."""
        assert "valve" in name
        return name


class ValveOneInterval(AssetOneInterval):
    """Valve asset data for a single interval."""

    cfg: ValveConfig
    high_temperature_load_mwh: pulp.LpVariable
    low_temperature_generation_mwh: pulp.LpVariable


class Valve(epl.Asset):
    """Valve asset - allows heat to flow from high to low temperature."""

    def __init__(self, name: str = "valve"):
        """Initialize the asset model."""
        self.cfg = ValveConfig(name=name)

    def __repr__(self) -> str:
        """A string representation of self."""
        return "<energypylinear.Valve>"

    def one_interval(
        self, optimizer: "epl.Optimizer", i: int, freq: "epl.Freq", flags: "epl.Flags"
    ) -> ValveOneInterval:
        """Create asset data for a single interval."""
        return ValveOneInterval(
            cfg=self.cfg,
            high_temperature_load_mwh=optimizer.continuous(
                f"{self.cfg.name}-high_temperature_load_mwh-{i}", low=0
            ),
            low_temperature_generation_mwh=optimizer.continuous(
                f"{self.cfg.name}-low_temperature_generation_mwh-{i}", low=0
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
        """Constrain thermal balance across the valve."""
        valve = ivars.filter_objective_variables(
            ValveOneInterval, i=-1, asset_name=self.cfg.name
        )[0]
        assert isinstance(valve, ValveOneInterval)
        optimizer.constrain(
            valve.high_temperature_load_mwh == valve.low_temperature_generation_mwh
        )

    def constrain_after_intervals(
        self, optimizer: "epl.Optimizer", ivars: "epl.IntervalVars"
    ) -> None:
        """Constrain the asset after all intervals."""
        return
