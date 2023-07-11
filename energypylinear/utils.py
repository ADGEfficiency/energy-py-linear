"""Utility code for energypylinear."""
import energypylinear as epl

typs = {
    "boiler": epl.assets.boiler.BoilerOneInterval,
    "battery": epl.assets.battery.BatteryOneInterval,
    "generator": epl.assets.chp.GeneratorOneInterval,
    "valve": epl.assets.valve.ValveOneInterval,
    "spill": epl.assets.spill.SpillOneInterval,
    "spill-evs": epl.assets.spill.SpillOneInterval,
    "evs-array": epl.assets.evs.EVsArrayOneInterval,
}


def filter_assets(vars: dict, asset: str, i: int = -1, name: str | None = None) -> list:
    """Returns a single timestep of assets by type."""
    assets = vars["assets"][i]
    return [
        a
        for a in assets
        if isinstance(a, typs[asset])
        and (a.cfg.name == name if name is not None else True)
    ]


def filter_all_assets(vars: dict, asset: str, name: str | None = None) -> list:
    """Returns all timesteps of assets by type."""
    out = []
    for assets in vars["assets"]:
        out.append(
            [
                a
                for a in assets
                if isinstance(a, typs[asset])
                and (a.cfg.name == name if name is not None else True)
            ]
        )
    return out
