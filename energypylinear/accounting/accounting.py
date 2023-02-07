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


def get_one_gas_account(
    interval_data: "epl.interval_data.IntervalData",
    results: pd.DataFrame,
) -> GasAccount:
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
) -> ElectricityAccount:
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

    # id = interval_data.electricity_carbon_intensities
    # pr = results["import_power_mwh"].values
    # breakpoint()  # fmt: skip
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
