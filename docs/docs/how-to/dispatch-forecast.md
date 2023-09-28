`energypylinear` has the ability to optimize for both actuals & forecasts.

An asset (or site) can be used to model the variance between optimizing for actual & forecast prices.

## Setup Interval Data

<!--phmdoctest-share-names-->
```python
electricity_prices = [100, 50, 200, -100, 0, 200, 100, -100]
forecasts = [-100, 0, 200, 100, -100, 100, 50, 200]
```

## Optimize with Perfect Foresight

<!--phmdoctest-share-names-->
```python
import energypylinear as epl

asset = epl.Battery(
    power_mw=2,
    capacity_mwh=4,
    efficiency_pct=0.9,
    electricity_prices=electricity_prices
)
actual = asset.optimize(verbose=False)
perfect_foresight = epl.get_accounts(actual.results, verbose=False)
print(f"{perfect_foresight=}")
```

```
perfect_foresight=<Accounts profit=1057.78 emissions=0.0822>
```

## Optimize to a Forecast

<!--phmdoctest-share-names-->
```python
import energypylinear as epl

asset = epl.Battery(
    power_mw=2,
    capacity_mwh=4,
    efficiency_pct=0.9,
    electricity_prices=forecasts
)
forecast = asset.optimize(verbose=False)
forecast_account = epl.get_accounts(
    forecast.results,
    price_results=actual.results,
    verbose=False
)
print(f"{forecast_account=}")
```

```
forecast_account=<Accounts profit=-140.00 emissions=0.0800>
```

## Calculate Variance Between Accounts

<!--phmdoctest-share-names-->
```python
variance = perfect_foresight - forecast_account
print(f"{variance=}")
```

```
variance=<Account profit=1197.78 emissions=0.0022>
```

## Full Example

```python
import io

import pandas as pd

import energypylinear as epl

#  price and forecast csv data
raw = """
Timestamp,Trading Price [$/MWh],Predispatch Forecast [$/MWh]
2018-07-01 17:00:00,177.11,97.58039000000001
2018-07-01 17:30:00,135.31,133.10307
2018-07-01 18:00:00,143.21,138.59978999999998
2018-07-01 18:30:00,116.25,128.09559
2018-07-01 19:00:00,99.97,113.29413000000001
2018-07-01 19:30:00,99.71,113.95063
2018-07-01 20:00:00,97.81,105.5491
2018-07-01 20:30:00,96.1,102.99768
2018-07-01 21:00:00,98.55,106.34366000000001
2018-07-01 21:30:00,95.78,91.82700000000001
2018-07-01 22:00:00,98.46,87.45
2018-07-01 22:30:00,91.88,85.65775
2018-07-01 23:00:00,91.69,85.0
2018-07-01 23:30:00,101.2,85.0
2018-07-02 00:00:00,139.55,80.99999
2018-07-02 00:30:00,102.9,75.85762
2018-07-02 01:00:00,83.86,67.86758
2018-07-02 01:30:00,71.1,70.21946
2018-07-02 02:00:00,60.35,62.151
2018-07-02 02:30:00,56.01,62.271919999999994
2018-07-02 03:00:00,51.22,56.79063000000001
2018-07-02 03:30:00,48.55,53.8532
2018-07-02 04:00:00,55.17,53.52591999999999
2018-07-02 04:30:00,56.21,49.57504
2018-07-02 05:00:00,56.32,48.42244
2018-07-02 05:30:00,58.79,54.15495
2018-07-02 06:00:00,73.32,58.01054
2018-07-02 06:30:00,80.89,68.31508000000001
2018-07-02 07:00:00,88.43,85.0
2018-07-02 07:30:00,201.43,119.73926999999999
2018-07-02 08:00:00,120.33,308.88984
2018-07-02 08:30:00,113.26,162.32117
"""
data = pd.read_csv(io.StringIO(raw))

#  optimize for actuals
asset = epl.Battery(
    power_mw=2,
    capacity_mwh=4,
    efficiency_pct=0.9,
    electricity_prices=data["Trading Price [$/MWh]"].values,
    freq_mins=30,
)
actuals = asset.optimize(verbose=False)

#  optimize for forecasts
asset = epl.Battery(
    power_mw=2,
    capacity_mwh=4,
    efficiency_pct=0.9,
    electricity_prices=data["Predispatch Forecast [$/MWh]"].values,
    freq_mins=30,
)
forecasts = asset.optimize(verbose=False)

#  calculate the variance between accounts
actual_account = epl.get_accounts(
    actuals.results, verbose=False

)
forecast_account = epl.get_accounts(
    forecasts.results,
    price_results=actuals.results,
    verbose=False
)
variance = actual_account - forecast_account

print(f"actuals: {actual_account}")
print(f"forecasts: {forecast_account}")
print(f"variance: {variance}")
print(
    f"\nforecast error: $ {-1 * variance.cost:2.2f} pct: {100 * variance.cost / actual_account.cost:2.1f} %"
)
```

```
actuals: <Accounts profit=325.82 emissions=0.0644>
forecasts: <Accounts profit=232.85 emissions=0.0544>
variance: <Account profit=92.97 emissions=0.0100>

forecast error: $ 92.97 pct: 28.5 %
```
