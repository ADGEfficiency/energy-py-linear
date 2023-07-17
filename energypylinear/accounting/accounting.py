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
        return f"<Account profit={-1 * self.cost:.2f} emissions={self.emissions:.4f}>"

    def __repr__(self) -> str:
        #  TODO
        return f"<Account profit={-1 * self.cost:.2f} emissions={self.emissions:.4f}>"

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
        return f"<Accounts profit={self.profit:.2f} emissions={self.emissions:.4f}>"

    def __repr__(self) -> str:
        #  TODO
        return f"<Accounts profit={self.profit:.2f} emissions={self.emissions:.4f}>"


def get_one_gas_account(
    interval_data: "epl.interval_data.IntervalData",
    results: pd.DataFrame,
) -> GasAccount:
    """Calculate a single gas account from interval data and results."""
    return GasAccount(
        cost=(interval_data.gas_prices * results["total-gas_consumption_mwh"]).sum(),
        emissions=(
            defaults.gas_carbon_intensity * results["total-gas_consumption_mwh"]
        ).sum(),
    )


def get_one_electricity_account(
    interval_data: "epl.interval_data.IntervalData",
    results: pd.DataFrame,
) -> ElectricityAccount:
    """Calculate a single electricity account from interval data and results."""
    import_cost = (
        interval_data.electricity_prices * results["site-import_power_mwh"]
    ).sum()
    export_cost = -(
        interval_data.electricity_prices * results["site-export_power_mwh"]
    ).sum()

    import_emissions = (
        interval_data.electricity_carbon_intensities * results["site-import_power_mwh"]
    ).sum()
    export_emissions = -(
        interval_data.electricity_carbon_intensities * results["site-export_power_mwh"]
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
    validate: bool = True,
    verbose: bool = True,
) -> Accounts:
    """
    Create one pair of gas and electricity accounts.
    for given of interval data and simulation results.

    `interval_data` gives the prices ($/MWh) and carbon intensities (tC/MWh) used
    in the calculation of cost and carbon emissions.

    `simulation` gives the energy quantities (MWh) used in the
    calculation of cost and carbon emissions.

    Args:
        interval_data: holds prices and carbon intensities.
        simulation: simulation results.
    """
    if validate:
        epl.results.validate_results(interval_data, simulation, verbose=verbose)
    electricity = get_one_electricity_account(interval_data, simulation)
    gas = get_one_gas_account(interval_data, simulation)

    return Accounts(
        electricity=electricity,
        gas=gas,
        cost=electricity.cost + gas.cost,
        profit=-(electricity.cost + gas.cost),
        emissions=electricity.emissions + gas.emissions,
    )
