"""Spill asset for allowing addition electric or thermal generation or consumption.

This allows infeasible simulations to become feasible. If a spill asset is used, then a warning is raised.
"""
import pulp
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval


class SpillConfig(AssetOneInterval):
    """Spill configuration."""

    name: str = "spill-default"

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


def spill_one_interval(
    optimizer: epl.optimizer.Optimizer, cfg: SpillConfig, i: int, freq: epl.freq.Freq
) -> SpillOneInterval:
    """Create Spill asset data for a single interval."""
    return SpillOneInterval(
        cfg=cfg,
        electric_generation_mwh=optimizer.continuous(
            f"{cfg.name}-electric_generation_mwh-{i}", low=0
        ),
        high_temperature_generation_mwh=optimizer.continuous(
            f"{cfg.name}-high_temperature_generation_mwh-{i}", low=0
        ),
        electric_load_mwh=optimizer.continuous(
            f"{cfg.name}-electric_load_mwh-{i}", low=0
        ),
        low_temperature_load_mwh=optimizer.continuous(
            f"{cfg.name}-low_temperature_load_mwh-{i}", low=0
        ),
    )
