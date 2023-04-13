"""
how to do multi-subsite

`top-level` sites attach to the grid - site import = grid

`sub-level` sites attach to other sites - site import = is coming from other site

also have this idea of generation / load - is that different from site import / export?  not really

want to rewrite it so that

    epl.Site(),
1. site import / export is handled by adding a grid connection as an asset to a site?

asset = [epl.GridConnection()]

what is a site? - site is list of assets - a site can be an asset
"""

import typing

import energypylinear as epl
from energypylinear.assets.site import SiteConfig


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
    spills = vars.get("spills")

    optimizer.constrain(
        (
            optimizer.sum([a.electric_generation_mwh for a in assets])
            - optimizer.sum([a.electric_load_mwh for a in assets])
            - optimizer.sum([a.charge_mwh for a in assets])
            + optimizer.sum([a.discharge_mwh for a in assets])
            + (spills[-1].electric_generation_mwh if spills else 0)
            - (spills[-1].electric_load_mwh if spills else 0)
            - interval_data.electricity_load_mwh[i]
        )
        == 0
    )


class GridConnectionConfig(pydantic.BaseModel):
    """Site configuration."""

    import_limit_mw: float = 10000
    export_limit_mw: float = 10000


class GridConnectionOneInterval(pydantic.BaseModel):
    """Site data for a single interval."""

    #  import power
    electric_generation_mwh: pulp.LpVariable
    #  export power
    electric_load_mwh: pulp.LpVariable

    import_power_bin: pulp.LpVariable
    export_power_bin: pulp.LpVariable

    import_limit_mwh: float
    export_limit_mwh: float

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed: bool = True


class GridConnection:
    def __init__(
        self,
        cfg: GridConnectionConfig = GridConnectionConfig(),
    ):
        self.assets = assets
        self.cfg = cfg

    def one_interval(
        self, optimizer: Optimizer, site: SiteConfig, i: int, freq: Freq
    ) -> SiteOneInterval:
        """Create Site asset data for a single interval."""
        return SiteOneInterval(
            import_power_mwh=optimizer.continuous(
                f"import_power_mw-{i}", up=freq.mw_to_mwh(site.import_limit_mw)
            ),
            export_power_mwh=optimizer.continuous(
                f"export_power_mw-{i}", up=freq.mw_to_mwh(site.import_limit_mw)
            ),
            import_power_bin=optimizer.binary(f"import_power_bin-{i}"),
            export_power_bin=optimizer.binary(f"export_power_bin-{i}"),
            import_limit_mwh=freq.mw_to_mwh(site.import_limit_mw),
            export_limit_mwh=freq.mw_to_mwh(site.export_limit_mw),
        )


class MultiSite:
    def __init__(
        self,
        assets: typing.Optional[list] = None,
        cfg: SiteConfig = SiteConfig(),
    ):
        self.assets = assets
        self.cfg = cfg

    def __repr__(self) -> str:
        return f"<energypylinear.Site assets: {len(self.assets)}>"

    def constrain_within_interval(
        self,
        optimizer: Optimizer,
        vars: dict,
        interval_data: "epl.interval_data.IntervalData",
        i: int,
    ) -> None:
        """Constrain site within a single interval."""
        constrain_site_electricity_balance(optimizer, vars, interval_data, i)
        # constrain_site_import_export(optimizer, vars)
        # constrain_site_high_temperature_heat_balance(optimizer, vars, interval_data, i)
        # constrain_site_low_temperature_heat_balance(optimizer, vars, interval_data, i)

    def optimize(self):
        for i in interval_data.idx:
            assets = []

            for asset in self.assets:
                assets.extend([asset.one_interval(self.optimizer, i, freq, flags)])

            vars["assets"].append(assets)
            self.constrain_within_interval(self.optimizer, vars, interval_data, i)


def test_multisite():
    site = epl.Site(assets=[GridConnection(), epl.Battery(), epl.Site(assets=[epl.Battery()])])
    breakpoint()  # fmt: skip
