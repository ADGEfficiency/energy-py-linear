"""Account for the use, cost and carbon emissions of electricity and gas."""

import numpy as np
import pandas as pd
import pydantic

import energypylinear as epl
from energypylinear.defaults import defaults
from energypylinear.objectives import (
    FunctionTermManyVariables,
    FunctionTermTwoVariables,
    OneTerm,
    Term,
    term_factory,
)


class Account(pydantic.BaseModel):
    """An account containing economic cost and carbon emissions."""

    cost: float = 0.0
    emissions: float = 0.0

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

    @pydantic.field_validator("import_emissions", "export_emissions")
    @classmethod
    def no_infinite(cls, v: float) -> float:
        """Check that our value is finite."""
        assert np.isfinite(v)
        return v


class GasAccount(Account):
    """A natural gas account, containing cost and emissions."""

    pass


class CustomAccount(Account):
    """An account for custom objective function terms."""

    pass


class Accounts(Account):
    """A set of accounts for electricity and gas."""

    electricity: ElectricityAccount = pydantic.Field(..., repr=False)
    gas: GasAccount = pydantic.Field(..., repr=False)
    custom: CustomAccount = pydantic.Field(..., repr=False)

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


def add_simple_terms() -> None:
    """TODO"""
    return None


def find_values_for_term(t_or_f: Term | float, results: pd.DataFrame) -> np.ndarray:
    """Get result values for a single term."""

    if isinstance(t_or_f, float):
        return np.full_like(results.index, t_or_f)
    else:
        assert isinstance(t_or_f, Term)
        term = t_or_f
        # do not use the interval data from the .a Term
        # TODO - here we are relying on the asset being called `site`
        # and that we forbid the use of any asset type except `site`
        assert term.asset_type == "site"
        assert term.asset_type is None or term.asset_type == "site"
        assert term.asset_name is None or term.asset_name == "site"
        assert term.asset_type != "*"
        vals = results[f"site-{term.variable}"].values

    return np.array(vals)


def add_two_variable_terms(results: pd.DataFrame, terms: list[OneTerm]) -> float:
    """Add a two variable term as a custom cost to the accounts.

    Works by finding the revelant variable values from the results and applying
    an aggregation function on them.
    """
    function_factory = {
        "max_two_variables": lambda x: np.max(x, axis=0),
        "min_two_variables": lambda x: np.min(x, axis=0),
    }
    costs = 0.0
    for term in terms:
        if term.type == "complex" and term.function in function_factory:
            assert isinstance(term, FunctionTermTwoVariables)

            a_vals = find_values_for_term(term.a, results)
            b_vals = find_values_for_term(term.b, results)
            assert a_vals.shape[0] == b_vals.shape[0]

            vals = function_factory[term.function]([a_vals, b_vals])
            assert vals.shape[0] == a_vals.shape[0]

            costs += (
                vals
                * (term.interval_data if term.interval_data else 1.0)
                * term.coefficient
            ).sum()

    return costs


def add_many_variable_terms(
    results: pd.DataFrame, terms: list[OneTerm], assets: list | None = None
) -> float:
    """Add a many variable term as a custom cost to the accounts.

    Works by finding the revelant variable values from the results and applying
    an aggregation function on them.
    """
    function_factory = {
        "max_many_variables": lambda x: np.max(x),
        "min_many_variables": lambda x: np.min(x),
    }
    costs = 0.0
    for term in terms:
        if term.type == "complex" and term.function in function_factory:
            assert isinstance(term, FunctionTermManyVariables)
            if term.variables.asset_type == "*":
                # get all assets with this variable
                vars = results[
                    [
                        col
                        for col in results.columns
                        if col.endswith(term.variables.variable)
                    ]
                ]

            elif (
                term.variables.asset_type is not None
                and term.variables.asset_name is None
            ):
                # get this asset type by variable

                # TODO - complex because of the renewable generator name thing
                # want ot keep `wind` and `solar` as ok names at the momnet
                # will go away when we change results col names

                type_ids = [term.variables.asset_type]
                if term.variables.asset_type == "renewable-generator":
                    type_ids = ["solar", "wind", "renewable-generator"]

                vars = results[
                    [
                        col
                        for col in results.columns
                        if (col.endswith(term.variables.variable))
                        # TODO this is a bit hairy
                        # we are relying on the asset type being part of the name
                        # will change this in the future when the asset type is separate from the name
                        # as part of the results
                        # `name--type--variable`
                        and (
                            len(
                                set(type_ids).intersection(
                                    set(["-".join(col.split("-")[:-1])])
                                )
                            )
                            == 1
                        )
                    ]
                ]

            else:
                # get this asset type and asset name
                vars = results[
                    [
                        col
                        for col in results.columns
                        if (col.endswith(term.variables.variable))
                        and (col.startswith(f"{term.variables.asset_name}-"))
                    ]
                ]

            costs += (
                function_factory[term.function](
                    np.hstack(
                        [
                            vars,
                            np.full_like(results.index, term.constant).reshape(-1, 1),
                        ]
                    )
                )
                * term.coefficient
            )

    return costs


def get_accounts(
    results: pd.DataFrame,
    price_results: pd.DataFrame | None = None,
    custom_terms: list[dict] | list[OneTerm] | None = None,
    assets: list | None = None,
    validate: bool = True,
    verbose: bool = True,
) -> Accounts:
    """Create a pair of gas and electricity accounts.

    Args:
        results: Simulation results used for accounts.
        price_results: Source of electricity & gas prices or carbon intensities.
            If not supplied, then `results` are used.
    """
    if price_results is None:
        price_results = results

    if validate:
        epl.check_results(results, verbose=verbose)
        epl.check_results(price_results, verbose=verbose)

    electricity = get_one_electricity_account(results, price_results)
    gas = get_one_gas_account(results, price_results)

    custom_costs = 0.0
    if custom_terms is not None:
        custom_terms = [term_factory(t) for t in custom_terms]
        custom_costs += add_simple_terms(results, custom_terms)
        custom_costs += add_two_variable_terms(results, custom_terms)
        custom_costs += add_many_variable_terms(results, custom_terms, assets)

    custom = CustomAccount(cost=custom_costs, emissions=0)

    return Accounts(
        electricity=electricity,
        gas=gas,
        custom=custom,
        cost=electricity.cost + gas.cost + custom.cost,
        profit=-(electricity.cost + gas.cost + custom.cost),
        emissions=electricity.emissions + gas.emissions + custom.emissions,
    )
