### Dispatch for Actuals vs. Dispatch for Forecasts

`energypylinear` has the ability to optimize for both actuals and forecasts.

The same primitives can be used to model the variance in performance of an asset optimized for actual prices versus forecast prices:

```python
import energypylinear as epl

#  interval data
electricity_prices = [100, 50, 200, -100, 0, 200, 100, -100]
forecasts = [-100, 0, 200, 100, -100, 100, 50, 200]

#  battery asset
asset = epl.battery.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)

#  optimize with perfect foresight
actual = asset.optimize(electricity_prices=electricity_prices)

#  optimize to the forecast
forecast = asset.optimize(electricity_prices=forecasts)

# create accounts for the two scenarios 
perfect_foresight = epl.get_accounts(actual.interval_data, actual.simulation)

# in the forecast we use the actual interval_data, not the forecast interval_data
forecast = epl.get_accounts(actual.interval_data, forecast.simulation)

variance = perfect_foresight - forecast
# cost=-1197.777778 emissions=0.002222221999999996
```

See also [examples/forecast-accuracy.py](https://github.com/ADGEfficiency/energy-py-linear/blob/main/examples/forecast-accuracy.py).

