import pulp

import energypylinear as epl
from energypylinear.assets.asset import Asset


class SpillConfig(Asset):
    name: str = "spill-alpha"


class SpillOneInterval(Asset):
    cfg: SpillConfig
    electric_generation_mwh: pulp.LpVariable
    high_temperature_generation_mwh: pulp.LpVariable

    electric_load_mwh: pulp.LpVariable
    high_temperature_load_mwh: pulp.LpVariable


def spill_one_interval(
    optimizer: epl.optimizer.Pulp, cfg: SpillConfig, i: int, freq: epl.freq.Freq
) -> SpillOneInterval:
    return SpillOneInterval(
        cfg=cfg,
        electric_generation_mwh=optimizer.continuous(
            f"{cfg.name}-electric_generation_mwh-{i}",
        ),
        high_temperature_generation_mwh=optimizer.continuous(
            f"{cfg.name}-high_temperature_generation_mwh-{i}",
        ),
        electric_load_mwh=optimizer.continuous(
            f"{cfg.name}-electric_load_mwh-{i}",
        ),
        high_temperature_load_mwh=optimizer.continuous(
            f"{cfg.name}-high_temperature_load_mwh-{i}",
        ),
    )
