"""Common utilities for testing."""
import numpy as np

import energypylinear as epl

asset_names = ["battery", "evs", "chp", "heat-pump", "renewable"]


def get_assets(ds: dict, asset: str) -> list[epl.OptimizeableAsset]:
    """Helper function to get assets from a string."""
    assets: list = []
    library = {
        "battery": epl.Battery(
            power_mw=2,
            capacity_mwh=4,
            efficiency_pct=0.9,
            **ds,
        ),
        "evs": epl.EVs(**ds, charger_turndown=0.0, charge_event_efficiency=1.0),
        "chp": epl.CHP(
            electric_power_max_mw=100,
            electric_power_min_mw=50,
            electric_efficiency_pct=0.2,
            high_temperature_efficiency_pct=0.2,
            low_temperature_efficiency_pct=0.2,
            **ds,
        ),
        "heat-pump": epl.HeatPump(
            **ds,
        ),
        "renewable": epl.RenewableGenerator(
            electric_generation_mwh=np.random.uniform(
                0, 100, len(ds["electricity_prices"])
            ),
            **ds,
        ),
    }
    assets.append(library[asset])
    return assets
