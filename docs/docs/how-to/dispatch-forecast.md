`energypylinear` has the ability to optimize for both actuals & forecasts.

An asset (or site) can be used to model the variance between optimizing for actual & forecast prices.

## Setup Battery Asset

<!--phmdoctest-share-names-->
```python
import energypylinear as epl

#  interval data
electricity_prices = [100, 50, 200, -100, 0, 200, 100, -100]
forecasts = [-100, 0, 200, 100, -100, 100, 50, 200]

#  battery asset
asset = epl.battery.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)
```

## Optimize with Perfect Foresight

<!--phmdoctest-share-names-->
```python
#  optimize with perfect foresight
actual = asset.optimize(electricity_prices=electricity_prices, verbose=False)

# create accounts for the two scenarios 
perfect_foresight = epl.get_accounts(actual.interval_data, actual.simulation, verbose=False)
print(f"{perfect_foresight=}")
```

```
perfect_foresight=<Accounts profit=1057.78 emissions=0.0822>
```

## Optimize to a Forecast

<!--phmdoctest-share-names-->
```python
#  optimize to the forecast
forecast = asset.optimize(electricity_prices=forecasts, verbose=False)

# in the forecast we use the actual interval_data, not the forecast interval_data
forecast_account = epl.get_accounts(actual.interval_data, forecast.simulation, verbose=False)
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
