import pandas as pd

import energypylinear as epl


def validate_results(results: pd.DataFrame) -> None:
    """
    check
    - don't import / export in same interval
    - column names (types?)

    validate against actuals / forecasts - lengths
    """


def accounting(
    actuals: "epl.data.IntervalData",
    results: pd.DataFrame,
    forecasts: "epl.data.IntervalData" = None,
):
    validate_results(results)

    return None
