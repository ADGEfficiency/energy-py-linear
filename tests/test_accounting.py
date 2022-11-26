import pandas as pd

import energypylinear as epl


def test_accounting():
    results = pd.DataFrame(
        {
            "import-power-mwh": [100, 50, 0],
            "export-power-mwh": [0, 0, 20],
            "total-gas_consumption_mwh": [20, 30, 40],
        }
    )
    actuals = epl.data.IntervalData(
        electricity_prices=[100, 200, -300], gas_prices=15, carbon_intensities=0.5
    )
    account = epl.accounting.accounting(actuals, results, forecasts=None)
    """
    what is account?
    - dataframe?
    - pydantic type?
        - lists
        - totals / sums

    can I make account nested - probably not?

    account.electricity.total_import_cost
    account.gas.total_cost
    account.carbon.total_net_carbon_emissions
    """
    assert account.total_import_electricity_cost = 100 * 100 + 200 * 50
    assert account.total_export_electricity_revenue = 20 * 40
    assert account.total_net_electricity_cost =  100 * 100 + 200 * 50 - 20 * 40

    breakpoint()  # fmt: skip
