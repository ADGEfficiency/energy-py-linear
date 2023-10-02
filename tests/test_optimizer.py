"""Tests the configuration of the optimizer."""


import numpy as np

import energypylinear as epl


def test_optimizer_config() -> None:
    """Test the use of the optimizer config

    TODO - pulp will still report Optimal status after the timeout, even if the
    relative gap isn't reached.

    Looks like the only way to sort this would be to capture the logs.
    """

    opt_cfg = epl.OptimizerConfig(timeout=2, relative_tolerance=0.0, msg=True)
    electricity_prices = np.clip(np.random.normal(100, 1000, 512), a_min=0, a_max=None)
    export_electricity_prices = 20.0
    asset = epl.Battery(
        electricity_prices=electricity_prices,
        export_electricity_prices=export_electricity_prices,
        optimizer_config=opt_cfg,
    )
    asset.optimize()
