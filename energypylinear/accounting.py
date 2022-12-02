import typing

import pandas as pd

import energypylinear as epl


def validate_results(results: pd.DataFrame) -> None:
    """
    validations TODO
    - don't import / export in same interval
    - column names (types?)
    - validate against interval data - lengths
    """

    cols = [
        "import_power_mwh",
        "export_power_mwh",
    ]
    for c in cols:
        assert c in results.columns


import pydantic


class OneElectricityAccount(pydantic.BaseModel):
    import_cost: float
    export_cost: float
    cost: float


class ElectricityAccount(pydantic.BaseModel):
    actuals: OneElectricityAccount
    forecasts: typing.Optional[OneElectricityAccount] = None


class Account(pydantic.BaseModel):
    electricity: ElectricityAccount


def get_one_electricity_account(
    actuals: "epl.data.IntervalData",
    results: pd.DataFrame,
):
    import_cost = (actuals.electricity_prices * results["import_power_mwh"]).sum()
    export_cost = -(actuals.electricity_prices * results["export_power_mwh"]).sum()

    return OneElectricityAccount(
        import_cost=import_cost, export_cost=export_cost, cost=import_cost + export_cost
    )


def get_electricity_account(
    actuals: "epl.data.IntervalData",
    results: pd.DataFrame,
    forecasts: "epl.data.IntervalData" = None,
):
    actuals_account = get_one_electricity_account(actuals, results)

    return ElectricityAccount(actuals=actuals_account, forecasts=forecasts)


def accounting(
    actuals: "epl.data.IntervalData",
    results: pd.DataFrame,
    forecasts: typing.Optional["epl.data.IntervalData"] = None,
):
    validate_results(results)
    return Account(
        electricity=get_electricity_account(actuals, results, forecasts),
    )
