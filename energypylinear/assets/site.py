"""Site asset for optimizing dispatch of combined heat and power (CHP) generators."""

import typing

import numpy as np
import pulp
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval
from energypylinear.constraints import add_custom_constraint
from energypylinear.defaults import defaults
from energypylinear.flags import Flags
from energypylinear.freq import Freq
from energypylinear.logger import logger, set_logging_level
from energypylinear.optimizer import Optimizer
from energypylinear.utils import repeat_to_match_length


def get_custom_interval_data(kwargs: dict | None) -> list | None:
    """
    Attempts to extract custom interval data from keyword arguments.

    Args:
        kwargs: Dictionary of potential custom interal data.

    Returns:
        A list of custom interval data, or `None` if no kwargs.
    """
    if (kwargs is None) or (kwargs == {}):
        return None

    custom_interval_data = []
    for key, data in kwargs.items():
        # check if data is a list, nparry, tuple - sequence like
        # could I check with the `typing.Sequence` type?
        if data is not None and isinstance(data, (list, np.ndarray, tuple)):
            custom_interval_data.append({"name": key, "data": data})
    return custom_interval_data


def validate_interval_data(
    assets: list,
    site: "epl.Site",
    custom_interval_data: dict | None = None,
    repeat_interval_data: bool = True,
) -> None:
    """Validates asset interval data against the site."""
    cid = get_custom_interval_data(custom_interval_data)

    # sets the interval data of each asset to the same length as the site interval data
    for asset in assets:
        if hasattr(asset.cfg, "interval_data"):
            if len(asset.cfg.interval_data.idx) != len(site.cfg.interval_data.idx):
                idata = asset.cfg.interval_data.model_dump(exclude={"idx"})

                for name, data in idata.items():
                    assert isinstance(site.cfg.interval_data.idx, np.ndarray)
                    setattr(
                        asset.cfg.interval_data,
                        name,
                        repeat_to_match_length(
                            data,
                            site.cfg.interval_data.idx,
                        )
                        if repeat_interval_data
                        else data,
                    )
                if repeat_interval_data:
                    setattr(asset.cfg.interval_data, "idx", site.cfg.interval_data.idx)

    if cid is not None:
        for custom in cid:
            if len(np.array(custom["data"]).shape) == 1:
                setattr(
                    site.cfg.interval_data,
                    custom["name"],
                    repeat_to_match_length(
                        custom["data"],
                        np.array(site.cfg.interval_data.idx),
                    )
                    if repeat_interval_data
                    else custom["data"],
                )

    # here really should check over all the interval data, not just the idx
    for asset in assets:
        if hasattr(asset.cfg, "interval_data"):
            assert len(asset.cfg.interval_data.idx) == len(site.cfg.interval_data.idx)


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
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @pydantic.model_validator(mode="after")
    def validate_all_things(self) -> "SiteIntervalData":
        """Validates site interval data."""

        fields = list(self.model_fields.keys())
        fields.remove("idx")
        if self.electricity_prices is not None:
            assert isinstance(self.electricity_prices, (np.ndarray, list))
            self.idx = np.arange(len(self.electricity_prices))
            self.electricity_prices = np.atleast_1d(np.array(self.electricity_prices))
            fields.remove("electricity_prices")

            if self.export_electricity_prices is None:
                self.export_electricity_prices = self.electricity_prices
                fields.remove("export_electricity_prices")

        elif self.electricity_carbon_intensities is not None:
            assert isinstance(self.electricity_carbon_intensities, (np.ndarray, list))
            self.idx = np.arange(len(self.electricity_carbon_intensities))
            self.electricity_carbon_intensities = np.atleast_1d(
                np.array(self.electricity_carbon_intensities)
            )
            fields.remove("electricity_carbon_intensities")

        else:
            raise ValueError(
                "One of electricity_prices or carbon_intensities should be specified."
            )

        idx = self.idx
        for field in fields:
            value = getattr(self, field, None)
            if isinstance(value, (float, int)):
                setattr(self, field, np.array([value] * len(idx)))

            elif value is None:
                setattr(self, field, np.array([getattr(defaults, field)] * len(idx)))

            else:
                assert len(value) == len(
                    idx
                ), f"{field} has len {len(value)}, index has {len(idx)}"
                setattr(self, field, np.array(value))

            assert getattr(self, field) is not None
            assert isinstance(getattr(self, field), np.ndarray)

            assert np.isnan(getattr(self, field)).sum() == 0

            setattr(self, field, np.atleast_1d(getattr(self, field)))

        return self


class SiteConfig(pydantic.BaseModel):
    """Site configuration."""

    name: typing.Literal["site"] = "site"
    interval_data: SiteIntervalData
    freq_mins: int
    import_limit_mw: float
    export_limit_mw: float

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<SiteConfig {self.name=}>"

    def __str__(self) -> str:
        """A string representation of self."""
        return f"<SiteConfig name={self.name}, freq_mins={self.freq_mins}, import_limit_mw={self.import_limit_mw}, export_limit_mw={self.export_limit_mw}>"

    @pydantic.field_validator("name")
    @classmethod
    def check_name(cls, name: str) -> str:
        """Ensure we can identify this asset correctly.

        Args:
            name: asset name.

        Returns:
            The asset name.
        """
        assert name == "site"
        return name


class SiteOneInterval(AssetOneInterval):
    """Site data for a single interval."""

    cfg: SiteConfig
    import_power_mwh: pulp.LpVariable
    export_power_mwh: pulp.LpVariable
    import_power_bin: pulp.LpVariable
    export_power_bin: pulp.LpVariable
    import_limit_mwh: float
    export_limit_mwh: float
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    """
    TODO

    this matters in (at least) two different places

    /Users/adam/energy-py-linear/energypylinear/constraints.py _resolve_constraint_term

            (getattr(v, term.variable) if getattr(v, term.variable) is not None else 0)

    /Users/adam/energy-py-linear/energypylinear/objectives.py
            variables = []
            for ass in assets:
                for a in ass:
                    if (v := getattr(a, term.variables.variable)) is not None:
                        variables.append(v)
    """
    electric_generation_mwh: None = None
    high_temperature_generation_mwh: None = None
    low_temperature_generation_mwh: None = None
    electric_charge_mwh: None = None
    electric_discharge_mwh: None = None


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
    site = ivars.filter_objective_variables(
        instance_type=epl.assets.site.SiteOneInterval, i=-1, asset_name=cfg.name
    )[0]
    assert isinstance(site, epl.assets.site.SiteOneInterval)
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
        )
        == 0,
        name=f"site_electricity_balance,i:{i}",
    )


def constrain_site_import_export(
    optimizer: Optimizer,
    cfg: SiteConfig,
    ivars: "epl.interval_data.IntervalVars",
) -> None:
    """Constrain to only do one of import and export electricity in an interval."""
    site = ivars.filter_objective_variables(
        instance_type=epl.assets.site.SiteOneInterval, i=-1, asset_name=cfg.name
    )[0]
    assert isinstance(site, epl.assets.site.SiteOneInterval)
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


# --8<-- [start:site]
class Site:
    """
    The Site asset can optimize many assets together in a single linear program.

    All assets are connected to the same site electricity, high and low temperature networks.
    """

    def __init__(
        self,
        assets: list,
        electricity_prices: np.ndarray | list[float] | float | None = None,
        export_electricity_prices: np.ndarray | list[float] | float | None = None,
        electricity_carbon_intensities: np.ndarray | list[float] | float | None = None,
        gas_prices: np.ndarray | list[float] | float | None = None,
        electric_load_mwh: np.ndarray | list[float] | float | None = None,
        high_temperature_load_mwh: np.ndarray | list[float] | float | None = None,
        low_temperature_load_mwh: np.ndarray | list[float] | float | None = None,
        low_temperature_generation_mwh: np.ndarray | list[float] | float | None = None,
        name: typing.Literal["site"] = "site",
        freq_mins: int = defaults.freq_mins,
        import_limit_mw: float = 10000,
        export_limit_mw: float = 10000,
        constraints: "list[epl.Constraint] | list[dict] | None" = None,
        **kwargs: typing.Any,
    ):
        """
        Initialize a Site.

        Args:
            assets: Assets to optimize together.
            electricity_prices: The price of import electricity in each interval.
                Will define both import and export prices if `export_electricity_prices` is None.
            export_electricity_prices: The price of export electricity in each interval.
            electricity_carbon_intensities: Carbon intensity of electricity in each interval.
            electric_load_mwh: Electricity demand consumed by the site.
            gas_prices: Price of natural gas, used in CHP and boilers in each interval.
            high_temperature_load_mwh: High temperature load of the site.
            low_temperature_load_mwh: Low temperature load of the site.
            name: The site name.
            freq_mins: Size of an interval in minutes.
            import_limit_mw: Maximum import power of the site.
            export_limit_mw: Minimum import power of the site.
            constraints: Additional custom constraints to apply to the linear program.
            kwargs: Keyword arguments attempted to be used as extra interval data.
        """
        # --8<-- [end:site]
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

        validate_interval_data(assets, self, custom_interval_data=kwargs)

        # TODO - should raise warning/error if kwargs get through - if there is a extra with that isn't made into interval data
        # could check if attr of interval data, if not, raise warning

        # TODO - these could go into the optimizer
        self.custom_constraints = constraints

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
        objective: "str | dict | epl.objectives.CustomObjectiveFunction" = "price",
        flags: Flags = Flags(),
        verbose: int | bool = 2,
        optimizer_config: "epl.OptimizerConfig | dict" = epl.optimizer.OptimizerConfig(),
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

        set_logging_level(logger, verbose)
        logger.info(f"assets.site.optimize: cfg={self.cfg}")
        logger.info(f"assets.site.optimize: assets={names}")

        #  TODO warn about sites without boilers?  warn sites without valve / spill?
        ivars = epl.IntervalVars()
        for i in self.cfg.interval_data.idx:
            one_interval = []
            one_interval.append(self.one_interval(self.optimizer, self.cfg, i, freq))
            for asset in self.assets:
                neu_assets = asset.one_interval(self.optimizer, i, freq, flags)
                if isinstance(neu_assets, list):
                    one_interval.extend(neu_assets)
                else:
                    one_interval.append(neu_assets)

            ivars.append(one_interval)

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

        # custom constraints are only placed after intervals
        # if we need access to the interval index again, do the loop when setting up the constraint

        if self.custom_constraints is not None:
            for constraint in self.custom_constraints:
                add_custom_constraint(
                    optimizer=self.optimizer,
                    constraint=constraint,
                    ivars=ivars,
                    interval_data=self.cfg.interval_data,
                )

        self.optimizer.objective(
            epl.get_objective(objective, self.optimizer, ivars, self.cfg.interval_data)
        )
        self.objective = objective

        status = self.optimizer.solve(
            verbose=verbose, allow_infeasible=flags.allow_infeasible
        )

        return epl.extract_results(
            self,
            self.assets,
            ivars,
            feasible=status.feasible,
            status=status,
            verbose=verbose,
            flags=flags,
        )
