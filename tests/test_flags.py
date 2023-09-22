"""Test use of boolean flags to change sim and results behaviour."""
import numpy as np

import energypylinear as epl
from energypylinear.flags import Flags


def test_flags() -> None:
    """Test that we can correctly use flags in our assets."""

    asset = epl.Battery(
        power_mw=2.0,
        capacity_mwh=4.0,
        electricity_prices=np.random.normal(10, 5, 100),
    )
    asset.optimize(
        flags=Flags(include_charge_discharge_binary_variables=True),
    )

    evs = epl.EVs(
        electricity_prices=np.random.normal(10, 5, 48),
    )
    evs.optimize(
        flags=Flags(limit_charge_variables_to_valid_events=True),
    )
    evs.optimize(
        flags=Flags(limit_charge_variables_to_valid_events=False),
    )
