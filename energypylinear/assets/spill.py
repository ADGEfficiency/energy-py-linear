"""Spill asset for allowing addition electric or thermal generation or consumption.

This allows infeasible simulations to become feasible. If a spill asset is used, then a warning is raised.
"""
import typing

import pulp
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval


class SpillConfig(AssetOneInterval):
    """Spill configuration."""

    name: str = "spill"

    @pydantic.validator("name")
    def check_name(cls, name: str) -> str:
        """Ensure we can identify this asset correctly."""
        assert "spill" in name
        return name


class SpillOneInterval(AssetOneInterval):
    """Spill asset data for a single interval."""

    cfg: SpillConfig = SpillConfig()
    electric_generation_mwh: pulp.LpVariable
    electric_load_mwh: pulp.LpVariable
    high_temperature_generation_mwh: pulp.LpVariable
    low_temperature_load_mwh: pulp.LpVariable

    electric_charge_mwh: float = 0.0
    electric_discharge_mwh: float = 0.0


class Spill(epl.Asset):
    """Spill asset - allows excess or insufficient balances to be filled in."""

    def __init__(self, name: str = "spill"):
        """Initializes the asset."""
        self.cfg = SpillConfig(name=name)

    def __repr__(self) -> str:
        """A string representation of self."""
        return "<energypylinear.Spill>"

    def one_interval(
        self,
        optimizer: epl.optimizer.Optimizer,
        i: int,
        freq: epl.freq.Freq,
        flags: epl.flags.Flags = epl.flags.Flags(),
    ) -> SpillOneInterval:
        """Generate linear program data for one interval."""
        return SpillOneInterval(
            cfg=self.cfg,
            electric_generation_mwh=optimizer.continuous(
                f"{self.cfg.name}-electric_generation_mwh-{i}", low=0
            ),
            high_temperature_generation_mwh=optimizer.continuous(
                f"{self.cfg.name}-high_temperature_generation_mwh-{i}", low=0
            ),
            electric_load_mwh=optimizer.continuous(
                f"{self.cfg.name}-electric_load_mwh-{i}", low=0
            ),
            low_temperature_load_mwh=optimizer.continuous(
                f"{self.cfg.name}-low_temperature_load_mwh-{i}", low=0
            ),
        )

    def constrain_within_interval(
        self, *args: typing.Any, **kwargs: typing.Any
    ) -> None:
        """Constrain asset within a single interval"""
        return

    def constrain_after_intervals(
        self, *args: typing.Any, **kwargs: typing.Any
    ) -> None:
        """Constrain asset after all intervals."""
        return
