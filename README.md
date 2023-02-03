# energy-py-linear

A Python library for optimizing the dispatch of energy assets with mixed-integer linear programming.

## Setup

Requires Python 3.10:

```shell
$ pip install energypylinear
```

## Battery

Dispatch an electric battery operating in wholesale price arbitrage:

```python
import energypylinear as epl

#  2.0 MW, 4.0 MWh battery
asset = epl.battery.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)
results = asset.optimize(
  electricity_prices=[100.0, 50, 200, -100, 0, 200, 100, -100],
  freq_mins=60,
  initial_charge_mwh=1,
  final_charge_mwh=3
)
print(results.simulation.head(3))
```

## CHP

Dispatch a Combined Heat & Power unit to generate high price electricity from natural gas.

The `epl.chp.Generator` model can be setup with an electric, high and low temperature thermal efficiencies - this allows modelling both gas engines and gas turbines:

```python
import energypylinear as epl

#  100 MWe gas engine
asset = epl.chp.Generator(
    electric_power_max_mw=100,
    electric_power_min_mw=30,
    electric_efficiency_pct=0.4,
    high_temperature_efficiency_pct=0.2,
    low_temperature_efficiency_pct=0.2,
)

#  100 MWe gas turbine
asset = epl.chp.Generator(
    electric_power_max_mw=100,
    electric_power_min_mw=50,
    electric_efficiency_pct=0.3,
    high_temperature_efficiency_pct=0.5,
)
```

When optimizing, we can input data for both the high and low temperature loads - both will be met by gas boilers if the CHP chooses not to generate.


```python
import energypylinear as epl

#  100 MWe gas engine
asset = epl.chp.Generator(
    electric_power_max_mw=100,
    electric_power_min_mw=10,
    electric_efficiency_pct=0.4,
    high_temperature_efficiency_pct=0.2,
    low_temperature_efficiency_pct=0.2,
)
results = asset.optimize(
  electricity_prices=[100, 50, 200, -100, 0, 200, 100, -100],
  high_temperature_load_mwh=[100, 50, 200, 40, 0, 200, 100, 100],
  low_temperature_load_mwh=20
)
print(results.simulation.head(3))
```

The `epl.chp.Generator` is allowed to dump both high temperature and low temperature heat.

## EVs

Control a number of EV chargers to charge a number of charge events.  Each charge event has a required amount of electricity `charge_event_mwh`:

```python
import energypylinear as epl

#  2 100 MW EV chargers
asset = epl.evs.EVs(charger_mws=[100, 100], charger_turndown=0.1)

electricity_prices = [-100, 50, 30, 50, 40]
charge_events = [
    [1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0],
]
charge_event_mwh = [50, 100, 30, 40]

results = asset.optimize(
    electricity_prices=electricity_prices,
    charge_events=charge_events,
    charge_event_mwh=charge_event_mwh,
)
print(results.simulation.head(3))
```

## Price versus Carbon Optimization

A key feature of `energypylinear` is the ability to optimize for both price and carbon as a first class feature of the library.

We can dispatch a battery for carbon by passing in `objective='carbon'`:

```python
import energypylinear as epl

asset = epl.battery.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)
results = asset.optimize(
  electricity_prices=[100, 50, 200, -100, 0, 200, 100, -100],
  electricity_carbon_intensities = [0.1, 0.2, 0.1, 0.15, 0.01, 0.7, 0.5, 0.01],
  objective='carbon'
)
print(results.simulation)
```

We can compare the results above with a simulation that optimizes for price.

We can use an `energypylinear.accounting.Account` to compare both simulations.  The accounting API is in it's first iteration - expect it to change in the future.

```python
import energypylinear as epl

#  interval data
electricity_prices = [100, 50, 200, -100, 0, 200, 100, -100]
electricity_carbon_intensities = [0.1, 0.2, 0.1, 0.15, 0.01, 0.7, 0.5, 0.01]

#  battery asset
asset = epl.battery.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)

#  optimize for money
price = asset.optimize(electricity_prices=electricity_prices)

#  optimize for the planet
carbon = asset.optimize(
  electricity_prices=electricity_prices,
  electricity_carbon_intensities=electricity_carbon_intensities,
  objective='carbon'
)

#  get an account representing the difference between the two
price = epl.accounting.get_accounts(
  price.interval_data,
  price.simulation,
)
carbon = epl.accounting.get_accounts(
  carbon.interval_data,
  carbon.simulation,
)
print(price)
# cost=-1057.777778 emissions=0.08222222199999996 profit=1057.777778

print(carbon)
# cost=-134.44444399999998 emissions=-2.2733333339999997 profit=134.44444399999998

variance = price - carbon
print(variance)
# cost=-923.3333339999999 emissions=2.3555555559999997

print(-variance.cost / variance.emissions)
# 391.9811322845319
```

Our optimization for price has a high negative cost.  The optimization for carbon has lower emissions, but at a higher cost.

## Actuals versus Forecasts

The same primitives can be used to model the variance in performance of an asset optimized for actual prices versus forecast prices.

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

# get accounts - note in the forecast we use the actual interval_data
perfect_foresight = epl.get_accounts(actual.interval_data, actual.simulation)
forecast = epl.get_accounts(actual.interval_data, forecast.simulation)

variance = perfect_foresight - forecast
print(variance)
# cost=-1197.777778 emissions=0.002222221999999996
```




## Test

```shell
$ make test
```
