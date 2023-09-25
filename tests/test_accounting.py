"""Tests for `epl.accounting`."""
import pandas as pd
import pytest

import energypylinear as epl
from energypylinear.defaults import defaults


def test_accounting_actuals() -> None:
    """Check calculation of electricity and gas costs and emissions."""

    results = pd.DataFrame(
        {
            "site-import_power_mwh": [100, 50, 0],
            "site-export_power_mwh": [0, 0, 20],
            "total-gas_consumption_mwh": [20, 30, 40],
            "total-electric_generation_mwh": [20, 30, 40],
            "total-electric_load_mwh": [120, 80, 20],
            "total-high_temperature_generation_mwh": [0, 0, 0],
            "total-high_temperature_load_mwh": [0, 0, 0],
            "total-low_temperature_generation_mwh": [0, 0, 0],
            "total-low_temperature_load_mwh": [0, 0, 0],
            "total-electric_charge_mwh": [0, 0, 0],
            "total-electric_discharge_mwh": [0, 0, 0],
            "total-electric_loss_mwh": [0, 0, 0],
            "total-spills_mwh": [0, 0, 0],
            "load-high_temperature_load_mwh": [0, 0, 0],
            "load-low_temperature_load_mwh": [0, 0, 0],
            "load-low_temperature_generation_mwh": [0, 0, 0],
            "site-electricity_prices": [100, 200, -300],
            "site-gas_prices": 15,
            "site-electricity_carbon_intensities": 0.5,
        }
    )
    actuals = epl.accounting.get_accounts(results, validate=True)

    assert actuals.electricity.import_cost == 100 * 100 + 200 * 50
    assert actuals.electricity.export_cost == -20 * -300
    assert actuals.electricity.cost == 100 * 100 + 50 * 200 - 20 * -300

    assert actuals.electricity.import_emissions == 0.5 * (100 + 50)
    assert actuals.electricity.export_emissions == -0.5 * (20)
    assert actuals.electricity.emissions == 0.5 * (100 + 50 - 20)

    assert actuals.gas.cost == 15 * (20 + 30 + 40)
    assert actuals.gas.emissions == defaults.gas_carbon_intensity * (20 + 30 + 40)

    assert actuals.emissions == 0.5 * (
        100 + 50 - 20
    ) + defaults.gas_carbon_intensity * (20 + 30 + 40)
    assert actuals.cost == 100 * 100 + 50 * 200 - 20 * -300 + 15 * (20 + 30 + 40)

    variance = actuals - actuals
    assert variance.cost == 0
    assert variance.emissions == 0

    #  randomly thrown in here for coverage
    with pytest.raises(NotImplementedError):
        actuals - float(32)


def test_accounting_forecasts() -> None:
    """Check calculation of forecast electricity and gas costs and emissions."""
    results = pd.DataFrame(
        {
            "site-import_power_mwh": [100, 50, 0],
            "site-export_power_mwh": [0, 0, 20],
            "total-gas_consumption_mwh": [20, 30, 40],
            "total-electric_generation_mwh": [20, 30, 40],
            "total-electric_load_mwh": [20, 30, 40],
        }
    )
    price_results_actuals = pd.DataFrame(
        {
            "site-electricity_prices": [100, 200, -300],
            "site-gas_prices": 15,
            "site-electricity_carbon_intensities": 0.5,
        }
    )
    price_results_forecasts = pd.DataFrame(
        {
            "site-electricity_prices": [200, -100, 100],
            "site-gas_prices": 10,
            "site-electricity_carbon_intensities": 0.4,
        }
    )
    actuals = epl.accounting.get_accounts(
        results, price_results_actuals, validate=False
    )
    forecasts = epl.accounting.get_accounts(
        results, price_results_forecasts, validate=False
    )

    assert forecasts.electricity.import_cost == 200 * 100 + -100 * 50
    assert forecasts.electricity.export_cost == -20 * 100
    assert forecasts.electricity.cost == 200 * 100 + -100 * 50 - 20 * 100

    assert forecasts.electricity.import_emissions == 0.4 * (100 + 50)
    assert forecasts.electricity.export_emissions == -0.4 * (20)
    assert forecasts.electricity.emissions == 0.4 * (100 + 50 - 20)

    assert forecasts.gas.cost == 10 * (20 + 30 + 40)
    assert forecasts.gas.emissions == defaults.gas_carbon_intensity * (20 + 30 + 40)

    assert forecasts.emissions == 0.4 * (
        100 + 50 - 20
    ) + defaults.gas_carbon_intensity * (20 + 30 + 40)
    assert forecasts.cost == 200 * 100 + -100 * 50 - 20 * 100 + 10 * (20 + 30 + 40)

    variance = actuals - forecasts
    assert variance.cost == actuals.cost - forecasts.cost
    assert variance.emissions == actuals.emissions - forecasts.emissions
