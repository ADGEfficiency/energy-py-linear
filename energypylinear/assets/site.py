"""Site asset for optimizing dispatch of combined heat and power (CHP) generators."""

import numpy as np
import pulp
import pydantic

import energypylinear as epl
from energypylinear.defaults import defaults
from energypylinear.flags import Flags
from energypylinear.freq import Freq
from energypylinear.optimizer import Optimizer
from energypylinear.utils import repeat_to_match_length


def validate_interval_data(
    assets: list, site: "epl.Site", repeat_interval_data: bool = True
) -> None:
    """Validates asset interval data against the site."""
    if not repeat_interval_data:
        for asset in assets:
            if hasattr(asset.cfg, "interval_data"):
                assert len(asset.cfg.interval_data.idx) == len(
                    site.cfg.interval_data.idx
                )

    else:
        for asset in assets:
            if hasattr(asset.cfg, "interval_data"):
                if len(asset.cfg.interval_data.idx) != len(site.cfg.interval_data.idx):

                    idata = asset.cfg.interval_data.dict(exclude={"idx"})
                    for name, data in idata.items():
                        assert isinstance(site.cfg.interval_data.idx, np.ndarray)
                        setattr(
                            asset.cfg.interval_data,
                            name,
                            repeat_to_match_length(
                                data,
                                site.cfg.interval_data.idx,
                            ),
                        )


class SiteIntervalData(pydantic.BaseModel):
    """Site interval data."""

    electricity_prices: np.ndarray | list[float] | float | None = None
    export_electricity_prices: np.ndarray | list[float] | float | None = None
    electricity_carbon_intensities: np.ndarray | list[float] | float | None = None
    gas_prices: np.ndarray | list[float] | float | None = None

    electric_load_mwh: np.ndarray | list[float] | float | None = None
    high_temperature_load_mwh: np.ndarray | list[float] | float | None = None
    low_temperature_load_mwh: np.ndarray | list[float] | float | None = None
    low_temperature_generation_mwh: np.ndarray | list[float] | float | None = None

    idx: list[int] | np.ndarray = []
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @pydantic.root_validator(pre=True)
    def validate_all_things(cls, values: dict) -> dict:
        """Validates site interval data."""

        fields = list(cls.__fields__.keys())
        fields.remove("idx")
        if values.get("electricity_prices") is not None:
            values["idx"] = np.arange(len(values["electricity_prices"]))
            values["electricity_prices"] = np.atleast_1d(
                np.array(values["electricity_prices"])
            )
            fields.remove("electricity_prices")

            if values.get("export_electricity_prices") is None:
                values["export_electricity_prices"] = values["electricity_prices"]
                fields.remove("export_electricity_prices")

        elif values.get("electricity_carbon_intensities") is not None:
            values["idx"] = np.arange(len(values["electricity_carbon_intensities"]))
            values["electricity_carbon_intensities"] = np.atleast_1d(
                np.array(values["electricity_carbon_intensities"])
            )
            fields.remove("electricity_carbon_intensities")

        else:
            raise ValueError(
                "One of electricity_prices or carbon_intensities should be specified."
            )

        idx = values["idx"]
        for field in fields:
            value = values.get(field)
            if isinstance(value, (float, int)):
                values[field] = np.array([value] * len(idx))

            elif value is None:
                values[field] = np.array([getattr(defaults, field)] * len(idx))

            else:
                assert len(value) == len(
                    idx
                ), f"{field} has len {len(value)}, index has {len(idx)}"
                values[field] = np.array(value)

            assert values[field] is not None
            assert isinstance(values[field], np.ndarray)
            assert np.isnan(values[field]).sum() == 0
            values[field] = np.atleast_1d(values[field])

        return values


class SiteConfig(pydantic.BaseModel):
    """Site configuration."""

    name: str
    interval_data: SiteIntervalData
    freq_mins: int

    import_limit_mw: float
    export_limit_mw: float


class SiteOneInterval(pydantic.BaseModel):
    """Site data for a single interval."""

    cfg: SiteConfig

    import_power_mwh: pulp.LpVariable
    export_power_mwh: pulp.LpVariable
    import_power_bin: pulp.LpVariable
    export_power_bin: pulp.LpVariable

    import_limit_mwh: float
    export_limit_mwh: float
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


def constrain_site_electricity_balance(
    optimizer: Optimizer,
    cfg: SiteConfig,
    ivars: "epl.interval_data.IntervalVars",
    interval_data: SiteIntervalData,
    i: int,
) -> None:
    """Constrain site electricity balance.

    in = out + accumulation
    import + generation = (export + load) + (charge - discharge)
    import + generation - (export + load) - (charge - discharge) = 0
    """
    assets = ivars.objective_variables[-1]
    site = ivars.filter_site(i=-1, site_name=cfg.name)
    spills = ivars.filter_objective_variables(epl.assets.spill.SpillOneInterval, i=-1)[
        0
    ]

    assert interval_data.electric_load_mwh is not None
    assert isinstance(interval_data.electric_load_mwh, np.ndarray)
    optimizer.constrain(
        (
            site.import_power_mwh
            - site.export_power_mwh
            - interval_data.electric_load_mwh[i]
            + optimizer.sum([a.electric_generation_mwh for a in assets])
            - optimizer.sum([a.electric_load_mwh for a in assets])
            - optimizer.sum([a.electric_charge_mwh for a in assets])
            + optimizer.sum([a.electric_discharge_mwh for a in assets])
            + (spills[-1].electric_generation_mwh if spills else 0)
            - (spills[-1].electric_load_mwh if spills else 0)
        )
        == 0
    )


def constrain_site_import_export(
    optimizer: Optimizer,
    cfg: SiteConfig,
    ivars: "epl.interval_data.IntervalVars",
) -> None:
    """Constrain to only do one of import and export electricity in an interval."""
    site = ivars.filter_site(i=-1, site_name=cfg.name)
    optimizer.constrain(
        site.import_power_mwh - site.import_limit_mwh * site.import_power_bin <= 0
    )
    optimizer.constrain(
        site.export_power_mwh - site.export_limit_mwh * site.export_power_bin <= 0
    )
    optimizer.constrain(site.import_power_bin + site.export_power_bin == 1)


def constrain_site_high_temperature_heat_balance(
    optimizer: Optimizer,
    cfg: SiteConfig,
    ivars: "epl.interval_data.IntervalVars",
    interval_data: SiteIntervalData,
    i: int,
) -> None:
    """Constrain high temperature energy balance.

    in = out + accumulation
    generation = load
    generation - load = 0
    """
    assets = ivars.objective_variables[-1]
    assert isinstance(interval_data.high_temperature_load_mwh, np.ndarray)
    optimizer.constrain(
        optimizer.sum([a.high_temperature_generation_mwh for a in assets])
        - optimizer.sum([a.high_temperature_load_mwh for a in assets])
        - interval_data.high_temperature_load_mwh[i]
        == 0
    )


def constrain_site_low_temperature_heat_balance(
    optimizer: Optimizer,
    cfg: SiteConfig,
    ivars: "epl.interval_data.IntervalVars",
    interval_data: SiteIntervalData,
    i: int,
) -> None:
    """Constrain low temperature energy balance.

    in = out + accumulation
    generation = load
    generation - load = 0
    """
    assets = ivars.objective_variables[-1]
    assert isinstance(interval_data.low_temperature_load_mwh, np.ndarray)
    assert isinstance(interval_data.low_temperature_generation_mwh, np.ndarray)
    optimizer.constrain(
        optimizer.sum([a.low_temperature_generation_mwh for a in assets])
        - optimizer.sum([a.low_temperature_load_mwh for a in assets])
        - interval_data.low_temperature_load_mwh[i]
        + interval_data.low_temperature_generation_mwh[i]
        == 0
    )


class Site:
    """Site asset - handles optimization and plotting of many assets over many intervals.

    All assets are connected to the same site electricity, high and low temperature networks.

    All assets are optimized as a single linear program.

    Args:
        assets: list[Asset] - a list of energypylinear assets to optimize together.
        cfg: SiteConfig - configuration for the site.
    """

    def __init__(
        self,
        assets: list,
        electricity_prices: np.ndarray | list[float] | float | None = None,
        export_electricity_prices: np.ndarray | list[float] | float | None = None,
        electricity_carbon_intensities: np.ndarray | list[float] | float | None = None,
        electric_load_mwh: np.ndarray | list[float] | float | None = None,
        gas_prices: np.ndarray | list[float] | float | None = None,
        high_temperature_load_mwh: np.ndarray | list[float] | float | None = None,
        low_temperature_load_mwh: np.ndarray | list[float] | float | None = None,
        low_temperature_generation_mwh: np.ndarray | list[float] | float | None = None,
        name: str = "site",
        freq_mins: int = defaults.freq_mins,
        import_limit_mw: float = 10000,
        export_limit_mw: float = 10000,
    ):
        """Initialize a Site asset model."""
        self.assets = assets

        self.cfg = SiteConfig(
            name=name,
            interval_data=SiteIntervalData(
                electricity_prices=electricity_prices,
                export_electricity_prices=export_electricity_prices,
                electricity_carbon_intensities=electricity_carbon_intensities,
                gas_prices=gas_prices,
                electric_load_mwh=electric_load_mwh,
                high_temperature_load_mwh=high_temperature_load_mwh,
                low_temperature_load_mwh=low_temperature_load_mwh,
                low_temperature_generation_mwh=low_temperature_generation_mwh,
            ),
            import_limit_mw=import_limit_mw,
            export_limit_mw=export_limit_mw,
            freq_mins=freq_mins,
        )

        validate_interval_data(assets, self)

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<energypylinear.Site assets: {len(self.assets)}>"

    def one_interval(
        self, optimizer: Optimizer, site: SiteConfig, i: int, freq: Freq
    ) -> SiteOneInterval:
        """Create Site asset data for a single interval."""
        return SiteOneInterval(
            cfg=site,
            import_power_mwh=optimizer.continuous(
                f"import_power_mw-{i}", up=freq.mw_to_mwh(self.cfg.import_limit_mw)
            ),
            export_power_mwh=optimizer.continuous(
                f"export_power_mw-{i}", up=freq.mw_to_mwh(self.cfg.export_limit_mw)
            ),
            import_power_bin=optimizer.binary(f"import_power_bin-{i}"),
            export_power_bin=optimizer.binary(f"export_power_bin-{i}"),
            import_limit_mwh=freq.mw_to_mwh(site.import_limit_mw),
            export_limit_mwh=freq.mw_to_mwh(site.export_limit_mw),
        )

    def constrain_within_interval(
        self,
        optimizer: Optimizer,
        ivars: "epl.interval_data.IntervalVars",
        interval_data: SiteIntervalData,
        i: int,
    ) -> None:
        """Constrain site within a single interval."""
        constrain_site_electricity_balance(optimizer, self.cfg, ivars, interval_data, i)
        constrain_site_import_export(optimizer, self.cfg, ivars)
        constrain_site_high_temperature_heat_balance(
            optimizer, self.cfg, ivars, interval_data, i
        )
        constrain_site_low_temperature_heat_balance(
            optimizer, self.cfg, ivars, interval_data, i
        )

    def optimize(
        self,
        objective: str = "price",
        flags: Flags = Flags(),
        verbose: bool = True,
        optimizer_config: "epl.OptimizerConfig" = epl.optimizer.OptimizerConfig(),
    ) -> "epl.SimulationResult":
        """Optimize sites dispatch using a mixed-integer linear program.

        Args:
            objective: the optimization objective - either "price" or "carbon".
            flags: boolean flags to change simulation and results behaviour.
            verbose: level of printing.
            optimizer_config: configuration options for the optimizer.

        Returns:
            epl.results.SimulationResult
        """

        self.optimizer = Optimizer(optimizer_config)
        freq = Freq(self.cfg.freq_mins)

        #  TODO this is repeated in `validate_interval_data`
        #  should reuse that here - TODO when I do the site again
        names = [asset.cfg.name for asset in self.assets]
        assert len(names) == len(
            set(names)
        ), f"Asset names must be unique, your assets are called {names}"

        #  TODO warn about sites without boilers?  warn sites without valve / spill?

        ivars = epl.IntervalVars()
        for i in self.cfg.interval_data.idx:
            ivars.append(self.one_interval(self.optimizer, self.cfg, i, freq))

            assets = []
            for asset in self.assets:
                neu_assets = asset.one_interval(self.optimizer, i, freq, flags)
                #  tech debt TODO
                #  EV is special because it returns many one interval blocks per step
                if isinstance(asset, epl.EVs):
                    evs, evs_array, spill_evs, spill_evs_array = neu_assets
                    assets.extend(evs)
                    assets.extend(spill_evs)
                    #  ivars has special logic for append
                    #  the EVsArrayOneInterval deals with them separately
                    ivars.append(evs_array)
                    ivars.append(spill_evs_array)
                else:
                    assets.append(neu_assets)

            ivars.append(assets)

            self.constrain_within_interval(
                self.optimizer, ivars, self.cfg.interval_data, i
            )
            for asset in self.assets:
                asset.constrain_within_interval(
                    self.optimizer, ivars, i, flags=flags, freq=freq
                )

        for asset in self.assets:
            asset.constrain_after_intervals(
                self.optimizer,
                ivars,
            )

        objective_fn = epl.objectives[objective]
        self.optimizer.objective(
            objective_fn(
                self.optimizer,
                ivars,
                self.cfg.interval_data,
            )
        )

        status = self.optimizer.solve(
            verbose=verbose, allow_infeasible=flags.allow_infeasible
        )

        return epl.extract_results(
            self,
            self.assets,
            ivars,
            feasible=status.feasible,
            verbose=verbose,
            flags=flags,
        )
