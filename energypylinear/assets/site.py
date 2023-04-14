"""Site asset for optimizing dispatch of combined heat and power (CHP) generators."""
import collections
import typing

import numpy as np
import pulp
import pydantic

import energypylinear as epl
from energypylinear.defaults import defaults
from energypylinear.flags import Flags
from energypylinear.freq import Freq
from energypylinear.optimizer import Optimizer


class SiteConfig(pydantic.BaseModel):
    """Site configuration."""

    import_limit_mw: float = 10000
    export_limit_mw: float = 10000


class SiteOneInterval(pydantic.BaseModel):
    """Site data for a single interval."""

    import_power_mwh: pulp.LpVariable
    export_power_mwh: pulp.LpVariable
    import_power_bin: pulp.LpVariable
    export_power_bin: pulp.LpVariable

    import_limit_mwh: float
    export_limit_mwh: float

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed: bool = True


def constrain_site_electricity_balance(
    optimizer: Optimizer,
    vars: dict,
    interval_data: "epl.interval_data.IntervalData",
    i: int,
) -> None:
    """Constrain site electricity balance.

    in = out + accumulation
    import + generation = (export + load) + (charge - discharge)
    import + generation - (export + load) - (charge - discharge) = 0
    """
    assets = vars["assets"][-1]
    site = vars["sites"][-1]
    spills = epl.utils.filter_assets(vars, "spill")

    assert interval_data.electricity_load_mwh is not None
    optimizer.constrain(
        (
            site.import_power_mwh
            - site.export_power_mwh
            - interval_data.electricity_load_mwh[i]
            + optimizer.sum([a.electric_generation_mwh for a in assets])
            - optimizer.sum([a.electric_load_mwh for a in assets])
            - optimizer.sum([a.charge_mwh for a in assets])
            + optimizer.sum([a.discharge_mwh for a in assets])
            + (spills[-1].electric_generation_mwh if spills else 0)
            - (spills[-1].electric_load_mwh if spills else 0)
        )
        == 0
    )


def constrain_site_import_export(optimizer: Optimizer, vars: dict) -> None:
    """Constrain to only do one of import and export electricity in an interval."""
    site = vars["sites"][-1]
    optimizer.constrain(
        site.import_power_mwh - site.import_limit_mwh * site.import_power_bin <= 0
    )
    optimizer.constrain(
        site.export_power_mwh - site.export_limit_mwh * site.export_power_bin <= 0
    )
    optimizer.constrain(site.import_power_bin + site.export_power_bin == 1)


def constrain_site_high_temperature_heat_balance(
    optimizer: Optimizer,
    vars: dict,
    interval_data: "epl.interval_data.IntervalData",
    i: int,
) -> None:
    """Constrain high temperature energy balance.

    in = out + accumulation
    generation = load
    generation - load = 0
    """
    assets = vars["assets"][-1]
    assert interval_data.high_temperature_load_mwh is not None
    optimizer.constrain(
        optimizer.sum([a.high_temperature_generation_mwh for a in assets])
        - optimizer.sum([a.high_temperature_load_mwh for a in assets])
        - interval_data.high_temperature_load_mwh[i]
        == 0
    )


def constrain_site_low_temperature_heat_balance(
    optimizer: Optimizer,
    vars: dict,
    interval_data: "epl.interval_data.IntervalData",
    i: int,
) -> None:
    """Constrain low temperature energy balance.

    in = out + accumulation
    generation = load
    generation - load = 0
    """
    assets = vars["assets"][-1]
    assert interval_data.low_temperature_load_mwh is not None
    optimizer.constrain(
        optimizer.sum([a.low_temperature_generation_mwh for a in assets])
        - optimizer.sum([a.low_temperature_load_mwh for a in assets])
        - interval_data.low_temperature_load_mwh[i]
        == 0
    )


class Site:
    def __init__(
        self,
        assets: typing.Optional[list] = None,
        cfg: SiteConfig = SiteConfig(),
    ):
        if assets is None:
            assets = []
        self.assets = assets
        self.cfg = cfg

    def __repr__(self) -> str:
        return f"<energypylinear.Site assets: {len(self.assets)}>"

    def one_interval(
        self, optimizer: Optimizer, site: SiteConfig, i: int, freq: Freq
    ) -> SiteOneInterval:
        """Create Site asset data for a single interval."""
        return SiteOneInterval(
            import_power_mwh=optimizer.continuous(
                f"import_power_mw-{i}", up=freq.mw_to_mwh(self.cfg.import_limit_mw)
            ),
            export_power_mwh=optimizer.continuous(
                f"export_power_mw-{i}", up=freq.mw_to_mwh(self.cfg.import_limit_mw)
            ),
            import_power_bin=optimizer.binary(f"import_power_bin-{i}"),
            export_power_bin=optimizer.binary(f"export_power_bin-{i}"),
            import_limit_mwh=freq.mw_to_mwh(site.import_limit_mw),
            export_limit_mwh=freq.mw_to_mwh(site.export_limit_mw),
        )

    def constrain_within_interval(
        self,
        optimizer: Optimizer,
        vars: dict,
        interval_data: "epl.interval_data.IntervalData",
        i: int,
    ) -> None:
        """Constrain site within a single interval."""
        constrain_site_electricity_balance(optimizer, vars, interval_data, i)
        constrain_site_import_export(optimizer, vars)
        constrain_site_high_temperature_heat_balance(optimizer, vars, interval_data, i)
        constrain_site_low_temperature_heat_balance(optimizer, vars, interval_data, i)

    def optimize(
        self,
        electricity_prices: typing.Union[float, typing.Iterable[float]],
        gas_prices: typing.Optional[typing.Union[float, typing.Iterable[float]]] = None,
        electricity_carbon_intensities: typing.Optional[
            typing.Union[float, typing.Iterable[float]]
        ] = None,
        high_temperature_load_mwh: typing.Optional[
            typing.Union[float, typing.Iterable[float]]
        ] = None,
        low_temperature_load_mwh: typing.Optional[
            typing.Union[float, typing.Iterable[float]]
        ] = None,
        freq_mins: int = defaults.freq_mins,
        initial_charge_mwh: float = 0.0,
        final_charge_mwh: typing.Union[float, None] = None,
        objective: str = "price",
        flags: Flags = Flags(),
        verbose: int = 0,
    ) -> "epl.results.SimulationResult":
        for asset in self.assets:
            if isinstance(asset, epl.Battery):
                asset.setup_initial_final_charge(initial_charge_mwh, final_charge_mwh)

        self.optimizer = Optimizer()
        freq = Freq(freq_mins)

        interval_data = epl.interval_data.IntervalData(
            electricity_prices=electricity_prices,
            gas_prices=gas_prices,
            electricity_carbon_intensities=electricity_carbon_intensities,
            high_temperature_load_mwh=high_temperature_load_mwh,
            low_temperature_load_mwh=low_temperature_load_mwh,
        )

        self.spill = epl.spill.Spill()
        self.valve = epl.valve.Valve()

        self.assets.append(self.spill)
        self.assets.append(self.valve)

        names = [asset.cfg.name for asset in self.assets]
        assert len(names) == len(
            set(names)
        ), f"Asset names must be unique, your assets are called {names}"

        #  warn about sites without boilers?  warn sites without valve / spill?

        vars: collections.defaultdict[str, typing.Any] = collections.defaultdict(list)
        for i in interval_data.idx:

            vars["sites"].append(self.one_interval(self.optimizer, self.cfg, i, freq))

            assets = []
            for asset in self.assets:
                assets.extend([asset.one_interval(self.optimizer, i, freq, flags)])
            vars["assets"].append(assets)

            self.constrain_within_interval(self.optimizer, vars, interval_data, i)
            for asset in self.assets:
                asset.constrain_within_interval(
                    self.optimizer, vars, flags=flags, freq=freq
                )

        for asset in self.assets:
            asset.constrain_after_intervals(self.optimizer, vars)

        assert len(interval_data.idx) == len(vars["assets"]) == len(vars["sites"])

        objective_fn = epl.objectives[objective]
        self.optimizer.objective(
            objective_fn(
                self.optimizer,
                vars,
                interval_data,
            )
        )

        status = self.optimizer.solve(verbose=verbose)
        self.interval_data = interval_data
        return epl.results.extract_results(
            interval_data, vars, feasible=status.feasible
        )
