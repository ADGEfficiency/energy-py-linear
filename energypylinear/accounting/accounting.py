import typing

import pandas as pd
import pydantic

import energypylinear as epl
from energypylinear.defaults import defaults


class OneElectricityAccount(pydantic.BaseModel):
    import_cost: float
    export_cost: float
    cost: float

    import_emissions: float
    export_emissions: float
    emissions: float


class OneGasAccount(pydantic.BaseModel):
    cost: float
    emissions: float


class GasAccounts(pydantic.BaseModel):
    actuals: OneGasAccount
    forecasts: typing.Optional[OneGasAccount] = None


class ElectricityAccounts(pydantic.BaseModel):
    actuals: OneElectricityAccount
    forecasts: typing.Optional[OneElectricityAccount] = None


class OneAccount(pydantic.BaseModel):
    cost: float
    emissions: float


class Accounts(pydantic.BaseModel):
    electricity: ElectricityAccounts
    gas: GasAccounts
    actuals: OneAccount
    forecasts: OneAccount


def get_one_gas_account(
    interval_data: "epl.data.IntervalData",
    results: pd.DataFrame,
):
    return OneGasAccount(
        cost=(interval_data.gas_prices * results["gas_consumption_mwh"]).sum(),
        emissions=(
            defaults.gas_carbon_intensity * results["gas_consumption_mwh"]
        ).sum(),
    )


def get_one_electricity_account(
    interval_data: "epl.data.IntervalData",
    results: pd.DataFrame,
):
    import_cost = (interval_data.electricity_prices * results["import_power_mwh"]).sum()

    export_cost = -(
        interval_data.electricity_prices * results["export_power_mwh"]
    ).sum()

    import_emissions = (
        interval_data.electricity_carbon_intensities * results["import_power_mwh"]
    ).sum()
    export_emissions = -(
        interval_data.electricity_carbon_intensities * results["export_power_mwh"]
    ).sum()

    return OneElectricityAccount(
        import_cost=import_cost,
        export_cost=export_cost,
        cost=import_cost + export_cost,
        import_emissions=import_emissions,
        export_emissions=export_emissions,
        emissions=import_emissions + export_emissions,
    )


def get_electricity_accounts(
    actuals: "epl.data.IntervalData",
    results: pd.DataFrame,
    forecasts: "epl.data.IntervalData",
) -> ElectricityAccounts:
    actuals_account = get_one_electricity_account(actuals, results)
    forecasts_account = get_one_electricity_account(forecasts, results)
    return ElectricityAccounts(actuals=actuals_account, forecasts=forecasts_account)


def get_gas_accounts(
    actuals: "epl.data.IntervalData",
    results: pd.DataFrame,
    forecasts: "epl.data.IntervalData" = None,
) -> GasAccounts:
    actuals_account = get_one_gas_account(actuals, results)
    forecasts_account = get_one_gas_account(forecasts, results)
    return GasAccounts(actuals=actuals_account, forecasts=forecasts_account)


def get_accounts(
    actuals: "epl.data.IntervalData",
    results: pd.DataFrame,
    forecasts: typing.Optional["epl.data.IntervalData"] = None,
):
    epl.data.validate_results(results)
    if forecasts is None:
        forecasts = actuals

    electricity = get_electricity_accounts(actuals, results, forecasts)
    gas = get_gas_accounts(actuals, results, forecasts)
    return Accounts(
        electricity=electricity,
        gas=gas,
        actuals=OneAccount(
            cost=electricity.actuals.cost + gas.actuals.cost,
            emissions=electricity.actuals.emissions + gas.actuals.emissions,
        ),
        forecasts=OneAccount(
            cost=electricity.forecasts.cost + gas.forecasts.cost,
            emissions=electricity.forecasts.emissions + gas.forecasts.emissions,
        ),
    )
