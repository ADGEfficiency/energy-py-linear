"""Boiler asset for optimizing dispatch of gas fired boilers"""
import typing

import pulp
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval
from energypylinear.flags import Flags
from energypylinear.freq import Freq
from energypylinear.optimizer import Optimizer


class BoilerConfig(AssetOneInterval):
    """Gas boiler configuration."""

    name: str
    high_temperature_generation_max_mw: float = 0
    high_temperature_generation_min_mw: float = 0
    high_temperature_efficiency_pct: float = pydantic.Field(gt=0.0, default=0.8, le=1.0)


class BoilerOneInterval(AssetOneInterval):
    """Boiler data for a single interval."""

    cfg: BoilerConfig
    high_temperature_generation_mwh: pulp.LpVariable
    gas_consumption_mwh: pulp.LpVariable
    binary: pulp.LpVariable


class Boiler:
    """Battery asset - generates high temperature heat."""

    def __init__(
        self,
        name: str = "boiler",
        high_temperature_generation_max_mw: float = 0,
        high_temperature_generation_min_mw: float = 0,
        high_temperature_efficiency_pct: float = 0.8,
    ):
        """Initialize a Boiler asset model."""
        self.cfg = BoilerConfig(
            name=name,
            high_temperature_generation_max_mw=high_temperature_generation_max_mw,
            high_temperature_generation_min_mw=high_temperature_generation_min_mw,
            high_temperature_efficiency_pct=high_temperature_efficiency_pct,
        )

    def __repr__(self) -> str:
        return "<energypylinear.Boiler>"

    def one_interval(
        self, optimizer: Optimizer, i: int, freq: Freq, flags: Flags = Flags()
    ) -> BoilerOneInterval:
        """Create Boiler asset data for a single interval."""
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
        optimizer: Optimizer,
        ivars: "epl.interval_data.IntervalVars",
        interval_data: "epl.IntervalData",
        i: int,
        freq: Freq,
        flags: Flags = Flags(),
    ) -> None:
        """Constrain boiler upper and lower bounds for generating high & low temperature heat."""
        boilers = ivars.filter_objective_variables(
            BoilerOneInterval, i=-1, asset_name=self.cfg.name
        )
        boiler = boilers[0][0]
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
        self, *args: typing.Any, **kwargs: typing.Any
    ) -> None:
        """Constrain asset after all interval asset models are created."""
        return
