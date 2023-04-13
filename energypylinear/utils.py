import energypylinear as epl

typs = {
    "boiler": epl.assets.boiler.BoilerOneInterval,
    "battery": epl.assets.battery.BatteryOneInterval,
    "generator": epl.assets.chp.GeneratorOneInterval,
    "valve": epl.assets.valve.ValveOneInterval,
}


def filter_assets(vars: dict, asset: str, i: str = -1) -> list:
    assets = vars["assets"][i]
    return [a for a in assets if isinstance(a, typs[asset])]


def filter_all_assets(vars: dict, asset: str) -> list:
    out = []
    for assets in vars["assets"]:
        out.append([a for a in assets if isinstance(a, typs[asset])])
    return out
