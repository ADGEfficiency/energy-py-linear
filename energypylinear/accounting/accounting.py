import typing

import pandas as pd
import pydantic

import energypylinear as epl
from energypylinear.defaults import defaults


class ElectricityAccount(pydantic.BaseModel):
    import_cost: float
    export_cost: float
    cost: float

    import_emissions: float
    export_emissions: float
    emissions: float


class GasAccount(pydantic.BaseModel):
    cost: float
    emissions: float


class GasAccounts(pydantic.BaseModel):
    actuals: GasAccount
    forecasts: GasAccount


class ElectricityAccounts(pydantic.BaseModel):
    actuals: ElectricityAccount
    forecasts: ElectricityAccount


class Account(pydantic.BaseModel):
    cost: float
    emissions: float

    def __sub__(self, other: object) -> "Account":
        """
        self == other -> True, else False
        """
        if not isinstance(other, Account):
            raise NotImplementedError("Cannot compare {other} to Account")
        return Account(
            cost=self.cost - other.cost,
            emissions=self.emissions - other.emissions,
        )


class Accounts(pydantic.BaseModel):
    electricity: ElectricityAccounts
    gas: GasAccounts
    actuals: Account
    forecasts: Account


def get_one_gas_account(
    interval_data: "epl.data.IntervalData",
    results: pd.DataFrame,
):
    return GasAccount(
        cost=(interval_data.gas_prices * results["gas_consumption_mwh"]).sum(),
        emissions=(
            defaults.gas_carbon_intensity * results["gas_consumption_mwh"]
        ).sum(),
    )


def get_gas_accounts(
    actuals: "epl.data.IntervalData",
    results_actuals: pd.DataFrame,
    results_forecasts: pd.DataFrame,
    forecasts: "epl.data.IntervalData",
) -> GasAccounts:
    actuals_account = get_one_gas_account(actuals, results_actuals)
    forecasts_account = get_one_gas_account(forecasts, results_forecasts)
    return GasAccounts(actuals=actuals_account, forecasts=forecasts_account)


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

    return ElectricityAccount(
        import_cost=import_cost,
        export_cost=export_cost,
        cost=import_cost + export_cost,
        import_emissions=import_emissions,
        export_emissions=export_emissions,
        emissions=import_emissions + export_emissions,
    )


def get_electricity_accounts(
    actuals: "epl.data.IntervalData",
    results_actuals: pd.DataFrame,
    results_forecasts: pd.DataFrame,
    forecasts: "epl.data.IntervalData",
) -> ElectricityAccounts:
    actuals_account = get_one_electricity_account(actuals, results_actuals)
    forecasts_account = get_one_electricity_account(forecasts, results_forecasts)
    return ElectricityAccounts(actuals=actuals_account, forecasts=forecasts_account)


def get_accounts(
    actuals: "epl.data.IntervalData",
    results_actuals: pd.DataFrame,
    results_forecasts: typing.Optional[pd.DataFrame] = None,
    forecasts: typing.Optional["epl.data.IntervalData"] = None,
):
    if results_forecasts is None:
        results_forecasts = results_actuals
    assert results_forecasts is not None

    if forecasts is None:
        forecasts = actuals

    epl.data.validate_results(results_actuals)
    epl.data.validate_results(results_forecasts)

    electricity = get_electricity_accounts(
        actuals, results_actuals, results_forecasts, forecasts
    )
    gas = get_gas_accounts(actuals, results_actuals, results_forecasts, forecasts)

    return Accounts(
        electricity=electricity,
        gas=gas,
        actuals=Account(
            cost=electricity.actuals.cost + gas.actuals.cost,
            emissions=electricity.actuals.emissions + gas.actuals.emissions,
        ),
        forecasts=Account(
            cost=electricity.forecasts.cost + gas.forecasts.cost,
            emissions=electricity.forecasts.emissions + gas.forecasts.emissions,
        ),
    )