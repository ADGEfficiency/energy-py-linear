import energypylinear as epl


def filter_assets(vars: dict, asset: str, i: str = -1) -> list:
    assets = vars["assets"][i]

    typs = {
        "boiler": epl.assets.boiler.BoilerOneInterval,
        "battery": epl.assets.battery.BatteryOneInterval,
        "generator": epl.assets.chp.GeneratorOneInterval,
    }
    return [a for a in assets if isinstance(a, typs[asset])]
