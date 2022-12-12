import pandas as pd

import energypylinear as epl
from energypylinear.defaults import defaults


def test_accounting_actuals():
    results = pd.DataFrame(
        {
            "import_power_mwh": [100, 50, 0],
            "export_power_mwh": [0, 0, 20],
            "gas_consumption_mwh": [20, 30, 40],
        }
    )
    actuals = epl.data.IntervalData(
        electricity_prices=[100, 200, -300],
        gas_prices=15,
        electricity_carbon_intensities=0.5,
    )
    account = epl.accounting.get_accounts(actuals, results, forecasts=None)

    assert account.electricity.actuals.import_cost == 100 * 100 + 200 * 50
    assert account.electricity.actuals.export_cost == -20 * -300
    assert account.electricity.actuals.cost == 100 * 100 + 50 * 200 - 20 * -300

    assert account.electricity.actuals.import_emissions == 0.5 * (100 + 50)
    assert account.electricity.actuals.export_emissions == -0.5 * (20)
    assert account.electricity.actuals.emissions == 0.5 * (100 + 50 - 20)

    assert account.gas.actuals.cost == 15 * (20 + 30 + 40)
    assert account.gas.actuals.emissions == defaults.gas_carbon_intensity * (
        20 + 30 + 40
    )

    assert account.actuals.emissions == 0.5 * (
        100 + 50 - 20
    ) + defaults.gas_carbon_intensity * (20 + 30 + 40)
    assert account.actuals.cost == 100 * 100 + 50 * 200 - 20 * -300 + 15 * (
        20 + 30 + 40
    )

    #  made accounts with no forecasts
    #  so the variance should be zero
    variance = account.actuals - account.forecasts
    assert variance.cost == 0
    assert variance.emissions == 0


def test_accounting_forecasts():
    results = pd.DataFrame(
        {
            "import_power_mwh": [100, 50, 0],
            "export_power_mwh": [0, 0, 20],
            "gas_consumption_mwh": [20, 30, 40],
        }
    )
    actuals = epl.data.IntervalData(
        electricity_prices=[100, 200, -300],
        gas_prices=15,
        electricity_carbon_intensities=0.5,
    )
    forecasts = epl.data.IntervalData(
        electricity_prices=[200, -100, 100],
        gas_prices=10,
        electricity_carbon_intensities=0.4,
    )
    account = epl.accounting.get_accounts(actuals, results, forecasts=forecasts)

    assert account.electricity.forecasts.import_cost == 200 * 100 + -100 * 50
    assert account.electricity.forecasts.export_cost == -20 * 100
    assert account.electricity.forecasts.cost == 200 * 100 + -100 * 50 - 20 * 100

    assert account.electricity.forecasts.import_emissions == 0.4 * (100 + 50)
    assert account.electricity.forecasts.export_emissions == -0.4 * (20)
    assert account.electricity.forecasts.emissions == 0.4 * (100 + 50 - 20)

    assert account.gas.forecasts.cost == 10 * (20 + 30 + 40)
    assert account.gas.forecasts.emissions == defaults.gas_carbon_intensity * (
        20 + 30 + 40
    )

    assert account.forecasts.emissions == 0.4 * (
        100 + 50 - 20
    ) + defaults.gas_carbon_intensity * (20 + 30 + 40)
    assert account.forecasts.cost == 200 * 100 + -100 * 50 - 20 * 100 + 10 * (
        20 + 30 + 40
    )

    variance = account.actuals - account.forecasts
    assert variance.cost == account.actuals.cost - account.forecasts.cost
    assert variance.emissions == account.actuals.emissions - account.forecasts.emissions
