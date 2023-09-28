"""Account for the use, cost and carbon emissions of electricity and gas."""

import numpy as np
import pandas as pd
import pydantic

import energypylinear as epl
from energypylinear.defaults import defaults


class Account(pydantic.BaseModel):
    """An account containing economic cost and carbon emissions."""

    cost: float
    emissions: float

    def __str__(self) -> str:
        """A string representation of self."""
        return f"<Account profit={-1 * self.cost:.2f} emissions={self.emissions:.4f}>"

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<Account profit={-1 * self.cost:.2f} emissions={self.emissions:.4f}>"

    def __sub__(self, other: object) -> "Account":
        """Subtracts two Account objects and returns a new Account.

        Args:
            other (object): The Account object to subtract from self.

        Returns:
            Account: New Account object with updated cost and emissions.

        Raises:
            NotImplementedError: If 'other' is not an instance of Account.
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

    @pydantic.validator("import_emissions", "export_emissions")
    def no_infinite(cls, v: float) -> float:
        """Check that our value is finite."""
        assert np.isfinite(v)
        return v


class GasAccount(Account):
    """A natural gas account, containing cost and emissions."""

    pass


class Accounts(Account):
    """A set of accounts for electricity and gas."""

    electricity: ElectricityAccount = pydantic.Field(..., repr=False)
    gas: GasAccount = pydantic.Field(..., repr=False)

    cost: float
    profit: float
    emissions: float

    def __str__(self) -> str:
        """A string representation of self."""
        return f"<Accounts profit={self.profit:.2f} emissions={self.emissions:.4f}>"

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<Accounts profit={self.profit:.2f} emissions={self.emissions:.4f}>"


def get_one_gas_account(
    results: pd.DataFrame,
    price_results: pd.DataFrame,
) -> GasAccount:
    """Calculate a single gas account."""
    return GasAccount(
        cost=(
            price_results["site-gas_prices"] * results["total-gas_consumption_mwh"]
        ).sum(),
        emissions=(
            defaults.gas_carbon_intensity * results["total-gas_consumption_mwh"]
        ).sum(),
    )


def get_one_electricity_account(
    results: pd.DataFrame,
    price_results: pd.DataFrame,
) -> ElectricityAccount:
    """Calculate a single electricity account."""
    import_cost = (
        price_results["site-electricity_prices"] * results["site-import_power_mwh"]
    ).sum()
    export_cost = -(
        price_results["site-electricity_prices"] * results["site-export_power_mwh"]
    ).sum()

    import_emissions = (
        price_results["site-electricity_carbon_intensities"]
        * results["site-import_power_mwh"]
    ).sum()
    export_emissions = -(
        price_results["site-electricity_carbon_intensities"]
        * results["site-export_power_mwh"]
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
    results: pd.DataFrame,
    price_results: pd.DataFrame | None = None,
    validate: bool = True,
    verbose: bool = True,
) -> Accounts:
    """Create a pair of gas and electricity accounts.

    Args:
        results: (pd.DataFrame)
            Simulation results used for accounts.
        price_results: (pd.DataFrame | None)
            Optional source of electricity & gas prices or carbon intensities.
    """
    if price_results is None:
        price_results = results

    if validate:
        epl.check_results(results, verbose=verbose)
        epl.check_results(price_results, verbose=verbose)

    electricity = get_one_electricity_account(results, price_results)
    gas = get_one_gas_account(results, price_results)

    return Accounts(
        electricity=electricity,
        gas=gas,
        cost=electricity.cost + gas.cost,
        profit=-(electricity.cost + gas.cost),
        emissions=electricity.emissions + gas.emissions,
    )
