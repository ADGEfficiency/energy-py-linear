"""Tests that we can calculate accounts for custom objectives.

These tests are similar to those in `test_complex_terms.py`"""
import numpy as np

import energypylinear as epl
from energypylinear.objectives import FunctionTermTwoVariables


def test_min_two_variables_export_threshold() -> None:
    """Test the use of the minimum of two variables function term.

    Similar to a test in `test_complex_terms.py`"""

    electricity_prices = np.random.uniform(-100, 100, 256)
    electric_load_mwh = np.random.uniform(0, 10, 256)
    gas_prices = 20
    export_threshold_mwh = 5
    export_charge = -500

    assets = [
        epl.CHP(
            electric_efficiency_pct=0.3,
            electric_power_max_mw=10.0,
        )
    ]
    site = epl.Site(
        assets=assets,
        gas_prices=gas_prices,
        electricity_prices=electricity_prices,
        electric_load_mwh=electric_load_mwh,
    )

    # start with a site with a simple objective function
    # test that the accounts balance against the objective function
    # value from the status
    terms = [
        {
            "asset_type": "site",
            "variable": "import_power_mwh",
            "interval_data": "electricity_prices",
        },
        {
            "asset_type": "site",
            "variable": "export_power_mwh",
            "interval_data": "electricity_prices",
            "coefficient": -1,
        },
        {
            "asset_type": "*",
            "variable": "gas_consumption_mwh",
            "interval_data": "gas_prices",
        },
    ]
    simulation = site.optimize(verbose=4, objective={"terms": terms})
    accounts = epl.get_accounts(simulation.results)
    np.testing.assert_allclose(simulation.status.objective, accounts.profit * -1)

    terms.append(
        FunctionTermTwoVariables(
            **{
                "type": "complex",
                "function": "min_two_variables",
                "a": {
                    "asset_type": "site",
                    "variable": "export_power_mwh",
                },
                "b": export_threshold_mwh,
                "coefficient": export_charge,
                "M": (
                    electric_load_mwh.max()
                    + assets[0].cfg.electric_power_max_mw
                    + export_threshold_mwh
                )
                * 1,
            }
        )
    )
    simulation = site.optimize(verbose=4, objective={"terms": terms})
    accounts = epl.get_accounts(simulation.results, custom_terms=terms[-1:])
    expected_charge = (
        simulation.results["site-export_power_mwh"]
        .clip(upper=export_threshold_mwh)
        .sum()
        * export_charge
    )
    np.testing.assert_allclose(expected_charge, accounts.custom.cost)
    np.testing.assert_allclose(simulation.status.objective, accounts.profit * -1)
