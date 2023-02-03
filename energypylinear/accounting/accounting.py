"""Functions to calculate electricity and gas accounts."""

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


class Accounts(Account):
    electricity: ElectricityAccount = pydantic.Field(..., repr=False)
    gas: GasAccount = pydantic.Field(..., repr=False)

    cost: float
    profit: float
    emissions: float


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


def get_accounts(
    interval_data: "epl.interval_data.IntervalData",
    simulation: pd.DataFrame,
):
    epl.results.validate_results(interval_data, simulation)
    electricity = get_one_electricity_account(interval_data, simulation)
    gas = get_one_gas_account(interval_data, simulation)

    return Accounts(
        electricity=electricity,
        gas=gas,
        cost=electricity.cost + gas.cost,
        profit=-(electricity.cost + gas.cost),
        emissions=electricity.emissions + gas.emissions,
    )
