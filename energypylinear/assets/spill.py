import pulp
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval


class SpillConfig(AssetOneInterval):
    name: str = "spill-default"

    @pydantic.validator("name")
    def ensure_spill_in_name(cls, name):
        assert "spill" in name
        return name


class SpillOneInterval(AssetOneInterval):
    cfg: SpillConfig = SpillConfig()
    electric_generation_mwh: pulp.LpVariable
    electric_load_mwh: pulp.LpVariable
    high_temperature_generation_mwh: pulp.LpVariable
    low_temperature_load_mwh: pulp.LpVariable


def spill_one_interval(
    optimizer: epl.optimizer.Optimizer, cfg: SpillConfig, i: int, freq: epl.freq.Freq
) -> SpillOneInterval:
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
