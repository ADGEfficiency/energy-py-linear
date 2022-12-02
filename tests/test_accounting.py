import pandas as pd

import energypylinear as epl


def test_accounting():
    results = pd.DataFrame(
        {
            "import_power_mwh": [100, 50, 0],
            "export_power_mwh": [0, 0, 20],
            "gas_consumption_mwh": [20, 30, 40],
        }
    )
    actuals = epl.data.IntervalData(
        electricity_prices=[100, 200, -300], gas_prices=15, carbon_intensities=0.5
    )
    account = epl.accounting.accounting(actuals, results, forecasts=None)
    assert account.electricity.actuals.import_cost == 100 * 100 + 200 * 50
    assert account.electricity.actuals.export_cost == -20 * -300
    assert account.electricity.actuals.cost == 100 * 100 + 50 * 200 - 20 * -300

    #  TODO gas + carbon
