"""Utility code for energypylinear."""
import energypylinear as epl

typs = {
    "boiler": epl.assets.boiler.BoilerOneInterval,
    "battery": epl.assets.battery.BatteryOneInterval,
    "generator": epl.assets.chp.GeneratorOneInterval,
    "valve": epl.assets.valve.ValveOneInterval,
    "spill": epl.assets.spill.SpillOneInterval,
    "evs-array": epl.assets.evs.EVsArrayOneInterval,
}


def filter_assets(vars: dict, asset: str, i: int = -1) -> list:
    """Returns a single timestep of assets by type."""
    assets = vars["assets"][i]
    return [a for a in assets if isinstance(a, typs[asset])]


def filter_all_assets(vars: dict, asset: str) -> list:
    """Returns all timesteps of assets by type."""
    out = []
    for assets in vars["assets"]:
        out.append([a for a in assets if isinstance(a, typs[asset])])
    return out
