"""Site asset for optimizing dispatch of combined heat and power (CHP) generators."""
import typing

import pulp
import pydantic

import energypylinear as epl
from energypylinear.defaults import defaults
from energypylinear.flags import Flags
from energypylinear.freq import Freq
from energypylinear.optimizer import Optimizer


class SiteConfig(pydantic.BaseModel):
    """Site configuration."""

    name: str = "site"

    import_limit_mw: float = 10000
    export_limit_mw: float = 10000


class SiteOneInterval(pydantic.BaseModel):
    """Site data for a single interval."""

    cfg: SiteConfig

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
    cfg: SiteConfig,
    ivars: "epl.interval_data.IntervalVars",
    interval_data: "epl.interval_data.IntervalData",
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

    assert interval_data.electricity_load_mwh is not None
    optimizer.constrain(
        (
            site.import_power_mwh
            - site.export_power_mwh
            - interval_data.electricity_load_mwh[i]
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
    interval_data: "epl.interval_data.IntervalData",
    i: int,
) -> None:
    """Constrain high temperature energy balance.

    in = out + accumulation
    generation = load
    generation - load = 0
    """
    assets = ivars.objective_variables[-1]
    assert interval_data.high_temperature_load_mwh is not None
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
    interval_data: "epl.interval_data.IntervalData",
    i: int,
) -> None:
    """Constrain low temperature energy balance.

    in = out + accumulation
    generation = load
    generation - load = 0
    """
    assets = ivars.objective_variables[-1]
    assert interval_data.low_temperature_load_mwh is not None
    optimizer.constrain(
        optimizer.sum([a.low_temperature_generation_mwh for a in assets])
        - optimizer.sum([a.low_temperature_load_mwh for a in assets])
        - interval_data.low_temperature_load_mwh[i]
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
        assets: list | None = None,
        cfg: SiteConfig = SiteConfig(),
    ):
        """Initialize a Site asset model."""
        if assets is None:
            assets = []
        self.assets = assets
        self.cfg = cfg

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
        ivars: "epl.interval_data.IntervalVars",
        interval_data: "epl.interval_data.IntervalData",
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
        charge_events: typing.Union[list[list[int]], typing.Iterable[int], None] = None,
        freq_mins: int = defaults.freq_mins,
        initial_charge_mwh: float = 0.0,
        final_charge_mwh: typing.Union[float, None] = None,
        objective: str = "price",
        flags: Flags = Flags(),
        verbose: bool = True,
    ) -> "epl.results.SimulationResult":
        """Optimize sites dispatch using a mixed-integer linear program.

        Args:
            electricity_prices: the price of electricity in each interval.
            gas_prices: the prices of natural gas, used in CHP and boilers in each interval.
            electricity_carbon_intensities: carbon intensity of electricity in each interval.
            high_temperature_load_mwh: high temperature load of the site in mega-watt hours.
            low_temperature_load_mwh: low temperature load of the site in mega-watt hours.
            charge_events: 2D matrix representing when a charge event is active.
                Shape is (n_charge_events, n_timesteps).
                A charge events matrix for 4 charge events over 5 intervals.
                ```
                charge_events = [
                    [1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1],
                    [0, 1, 0, 0, 0],
                ]
                ```

            freq_mins: the size of an interval in minutes.
            initial_charge_mwh: initial charge state of the battery in mega-watt hours.
            final_charge_mwh: final charge state of the battery in mega-watt hours.
            objective: the optimization objective - either "price" or "carbon".
            flags: boolean flags to change simulation and results behaviour.
            verbose: level of printing.

        Returns:
            epl.results.SimulationResult
        """

        #  TODO could be `asset.before_intervals()`
        for asset in self.assets:
            if isinstance(asset, epl.Battery):
                asset.setup_initial_final_charge(initial_charge_mwh, final_charge_mwh)

            if isinstance(asset, epl.EVs):
                assert (
                    asset.charge_events is not None
                ), "EV asset should have charge events defined in it's __init__"

        self.optimizer = Optimizer()
        freq = Freq(freq_mins)

        #  not using the evs' interval data here
        #  instead that data structure is only made inside the ev assets
        interval_data = epl.interval_data.IntervalData(
            electricity_prices=electricity_prices,
            gas_prices=gas_prices,
            electricity_carbon_intensities=electricity_carbon_intensities,
            high_temperature_load_mwh=high_temperature_load_mwh,
            low_temperature_load_mwh=low_temperature_load_mwh,
            evs=None,
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

        #  this is needed for evs_one_interval
        # for asset in self.assets:
        #     asset.interval_data = interval_data

        ivars = epl.interval_data.IntervalVars()
        for i in interval_data.idx:
            ivars.append(self.one_interval(self.optimizer, self.cfg, i, freq))
            assets = []
            for asset in self.assets:

                neu_assets = asset.one_interval(self.optimizer, i, freq, flags)
                #  tech debt TODO
                #  EV is special beacuse it returns many blocks per step
                if isinstance(asset, epl.EVs):
                    evs, evs_array, spill_evs, spill_evs_array = neu_assets
                    assets.extend(evs)
                    assets.extend(spill_evs)
                    ivars.append(evs_array)
                    ivars.append(spill_evs_array)
                else:
                    assets.append(neu_assets)

            ivars.append(assets)

            self.constrain_within_interval(self.optimizer, ivars, interval_data, i)
            for asset in self.assets:
                asset.constrain_within_interval(
                    self.optimizer, ivars, interval_data, i, flags=flags, freq=freq
                )

        for asset in self.assets:
            asset.constrain_after_intervals(
                self.optimizer,
                ivars,
                interval_data,
            )

        assert len(interval_data.idx) == len(ivars.objective_variables)

        objective_fn = epl.objectives[objective]
        self.optimizer.objective(
            objective_fn(
                self.optimizer,
                ivars,
                interval_data,
            )
        )

        status = self.optimizer.solve(verbose=verbose)
        self.interval_data = interval_data
        return epl.results.extract_results(
            interval_data, ivars, feasible=status.feasible, verbose=verbose
        )
