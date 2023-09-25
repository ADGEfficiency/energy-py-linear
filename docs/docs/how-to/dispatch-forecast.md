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

See [examples/forecast-accuracy.py](https://github.com/ADGEfficiency/energy-py-linear/blob/main/examples/forecast-accuracy.py) for a complete example.
