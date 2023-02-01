"""Functions to calculate electricity and gas accounts."""

import typing

import pandas as pd
import pydantic

import energypylinear as epl
from energypylinear.defaults import defaults


class Account(pydantic.BaseModel):
    """An account containing cost and emissions."""

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


class ElectricityAccount(Account):
    """An electricity account, containing import and export costs and emissions."""

    import_cost: float
    export_cost: float
    cost: float

    import_emissions: float
    export_emissions: float
    emissions: float


class GasAccount(Account):
    """A natural gas account, containing cost and emissions"""

    pass


class GasAccounts(pydantic.BaseModel):
    """A set of gas accounts, containing actuals and forecasts."""

    actuals: GasAccount
    forecasts: GasAccount


class ElectricityAccounts(pydantic.BaseModel):
    """A set of electricity accounts, containing actuals and forecasts."""

    actuals: ElectricityAccount
    forecasts: ElectricityAccount


class Accounts(pydantic.BaseModel):
    electricity: ElectricityAccounts
    gas: GasAccounts
    actuals: Account
    forecasts: Account


def get_one_gas_account(
    interval_data: "epl.interval_data.IntervalData",
    results: pd.DataFrame,
):
    """Calculate a single gas account from interval data and results."""
    return GasAccount(
        cost=(interval_data.gas_prices * results["gas_consumption_mwh"]).sum(),
        emissions=(
            defaults.gas_carbon_intensity * results["gas_consumption_mwh"]
        ).sum(),
    )


def get_gas_accounts(
    actuals: "epl.interval_data.IntervalData",
    results_actuals: pd.DataFrame,
    results_forecasts: pd.DataFrame,
    forecasts: "epl.interval_data.IntervalData",
) -> GasAccounts:
    """Calculate gas accounts from actuals and forecasts interval data and results."""
    actuals_account = get_one_gas_account(actuals, results_actuals)
    forecasts_account = get_one_gas_account(forecasts, results_forecasts)
    return GasAccounts(actuals=actuals_account, forecasts=forecasts_account)


def get_one_electricity_account(
    interval_data: "epl.interval_data.IntervalData",
    results: pd.DataFrame,
):
    """Calculate a single electricity account from interval data and results."""
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
    actuals: "epl.interval_data.IntervalData",
    results_actuals: pd.DataFrame,
    results_forecasts: pd.DataFrame,
    forecasts: "epl.interval_data.IntervalData",
) -> ElectricityAccounts:
    """Calculate electricity accounts from actuals and forecasts interval data and results."""
    actuals_account = get_one_electricity_account(actuals, results_actuals)
    forecasts_account = get_one_electricity_account(forecasts, results_forecasts)
    return ElectricityAccounts(actuals=actuals_account, forecasts=forecasts_account)


def get_accounts(
    actuals: "epl.interval_data.IntervalData",
    results_actuals: pd.DataFrame,
    results_forecasts: typing.Optional[pd.DataFrame] = None,
    forecasts: typing.Optional["epl.interval_data.IntervalData"] = None,
):
    """Calculate electricity and gas accounts from actuals and forecasts interval data and results."""
    if results_forecasts is None:
        results_forecasts = results_actuals
    assert results_forecasts is not None

    if forecasts is None:
        forecasts = actuals

    epl.results.validate_results(actuals, results_actuals)
    epl.results.validate_results(forecasts, results_forecasts)

    electricity = get_electricity_accounts(
        actuals, results_actuals, results_forecasts, forecasts
    )
    gas = get_gas_accounts(actuals, results_actuals, results_forecasts, forecasts)

    return Accounts(
        electricity=electricity,
        gas=gas,
        actuals=Account(
            cost=electricity.actuals.cost + gas.actuals.cost,
            profit=-(electricity.actuals.cost + gas.actuals.cost),
            emissions=electricity.actuals.emissions + gas.actuals.emissions,
        ),
        forecasts=Account(
            cost=electricity.forecasts.cost + gas.forecasts.cost,
            profit=-(electricity.forecasts.cost + gas.forecasts.cost),
            emissions=electricity.forecasts.emissions + gas.forecasts.emissions,
        ),
    )
