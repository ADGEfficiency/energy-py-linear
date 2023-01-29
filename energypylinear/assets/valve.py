import pulp
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval


class ValveConfig(pydantic.BaseModel):
    name: str


class ValveOneInterval(AssetOneInterval):
    cfg: ValveConfig
    high_temperature_load_mwh: pulp.LpVariable
    low_temperature_generation_mwh: pulp.LpVariable


def valve_one_interval(
    optimizer: "epl.optimizer.Optimizer",
    cfg: ValveConfig,
    i: int,
    freq: "epl.freq.Freq",
) -> ValveOneInterval:
    return ValveOneInterval(
        cfg=cfg,
        high_temperature_load_mwh=optimizer.continuous(
            f"{cfg.name}-high_temperature_load_mwh-{i}", low=0
        ),
        low_temperature_generation_mwh=optimizer.continuous(
            f"{cfg.name}-low_temperature_generation_mwh-{i}", low=0
        ),
    )


def constrain_within_interval_valve(
    optimizer: "epl.optimizer.Optimizer", vars: dict
) -> None:
    valve = vars["valves"][-1]
    optimizer.constrain(
        valve.high_temperature_load_mwh == valve.low_temperature_generation_mwh
    )
