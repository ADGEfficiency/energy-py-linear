"""
Renewable Generator asset.

Suitable for modelling either turndownable wind or solar."""

import numpy as np
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval
from energypylinear.defaults import defaults
from energypylinear.flags import Flags


def repeat_to_match_length(a, b):
    #  TODO unit test and move to utils
    quotient, remainder = divmod(len(b), len(a))
    return np.concatenate([np.tile(a, quotient), a[:remainder]])


def validate_interval_data(assets, site, repeat_interval_data: bool = True):

    if not repeat_interval_data:
        for asset in assets:
            assert len(asset.cfg.interval_data.idx) == len(site.cfg.interval_data.idx)

    else:
        for asset in assets:
            if len(asset.cfg.interval_data.idx) != len(site.cfg.interval_data.idx):

                idata = asset.cfg.interval_data.dict(exclude={"idx"})
                for name, data in idata.items():
                    setattr(
                        asset.cfg.interval_data,
                        name,
                        repeat_to_match_length(
                            data,
                            site.cfg.interval_data.idx,
                        ),
                    )


class RenewableGeneratorIntervalData(pydantic.BaseModel):
    electric_generation_mwh: float | list[float] | np.ndarray
    idx: list[int] | np.ndarray = []

    @pydantic.validator("idx", always=True)
    def create_idx(cls, _: list, values: dict) -> np.ndarray:
        """Creates an integer index."""
        return np.arange(len(values["electric_generation_mwh"]))

    @pydantic.validator("electric_generation_mwh", pre=True, always=True)
    def handle_single_float(cls, value) -> np.ndarray:
        if isinstance(value, float):
            return [value]
        return np.array(value)

    class Config:
        arbitrary_types_allowed: bool = True


class RenewableGeneratorConfig(pydantic.BaseModel):
    """Renewable Generator asset configuration."""

    name: str
    electric_generation_lower_bound_pct: float = pydantic.Field(..., ge=0, le=1)
    interval_data: RenewableGeneratorIntervalData


class RenewableGeneratorOneInterval(AssetOneInterval):
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
        electric_generation_mwh: float | list[float] | np.ndarray,
        electricity_prices: float | list[float] | np.ndarray | None = None,
        electricity_carbon_intensities: float | list[float] | np.ndarray | None = None,
        electric_load_mwh: float | list[float] | np.ndarray | None = None,
        electric_generation_lower_bound_pct: float = 1.0,
        name="str",
        freq_mins: int = defaults.freq_mins,
    ):
        self.cfg = RenewableGeneratorConfig(
            name=name,
            electric_generation_lower_bound_pct=electric_generation_lower_bound_pct,
            interval_data=RenewableGeneratorIntervalData(
                electric_generation_mwh=electric_generation_mwh,
            ),
            freq_mins=freq_mins,
        )

        assets = [self]

        self.site = epl.Site(
            assets=assets,
            electric_load_mwh=electric_load_mwh,
            electricity_prices=electricity_prices,
            electricity_carbon_intensities=electricity_carbon_intensities,
        )

        validate_interval_data(assets, self.site)

    def __repr__(self) -> str:
        """A string representation of self."""
        raise NotImplementedError()
        # return (
        #     f"<energypylinear.HeatPump {self.cfg.electric_power_mw=}, {self.cfg.cop=}>"
        # )

    def one_interval(
        self, optimizer: "epl.Optimizer", i: int, freq: "epl.Freq", flags: "epl.Flags"
    ):
        """Create asset data for a single interval."""
        name = f"i:{i},asset:{self.cfg.name}"
        assert isinstance(self.cfg.interval_data.electric_generation_mwh, np.ndarray)
        return RenewableGeneratorOneInterval(
            cfg=self.cfg,
            electric_generation_mwh=optimizer.continuous(
                f"electric_generation_mwh,{name}",
                low=freq.mw_to_mwh(self.cfg.interval_data.electric_generation_mwh[i])
                * self.cfg.electric_generation_lower_bound_pct,
                up=freq.mw_to_mwh(self.cfg.interval_data.electric_generation_mwh[i]),
            ),
        )

    def constrain_within_interval(
        self,
        optimizer: "epl.Optimizer",
        ivars: "epl.IntervalVars",
        i: int,
        freq: "epl.Freq",
        flags: "epl.Flags",
    ):
        """Constrain optimization within a single interval."""
        pass

    def constrain_after_intervals(
        self,
        optimizer: "epl.Optimizer",
        ivars: "epl.IntervalVars",
    ):
        pass

    def optimize(
        self,
        objective: str = "price",
        verbose: bool = True,
        flags: Flags = Flags(),
    ) -> "epl.results.extract.SimulationResult":
        return self.site.optimize(
            freq_mins=self.cfg.freq_mins,
            objective=objective,
            flags=flags,
            verbose=verbose,
        )
