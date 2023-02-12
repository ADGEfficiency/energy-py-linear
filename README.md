# energy-py-linear

<img src="./static/coverage.svg"> [![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

A Python library for optimizing the dispatch of energy assets with mixed-integer linear programming:

- electric batteries,
- combined heat & power (CHP) generators,
- electric vehicle smart charging.

Models can be optimized to either maximize profit or minimize carbon emissions.

## Setup

Requires Python 3.10+:

```shell
$ pip install energypylinear
```

## Assets

### Battery

Dispatch an electric battery operating in wholesale price arbitrage using `epl.Battery`:

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
```

### CHP

Dispatch a CHP unit to generate electricity and heat from natural gas.

The `epl.Generator` model can be configured with electric, high and low temperature thermal efficiencies. 

This allows modelling both gas engines and gas turbines:

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

When optimizing, we can use interval data for the high and low temperature loads.  These thermal loads will be met by gas boilers if the CHP chooses not to generate.  The `epl.chp.Generator` is allowed to dump both high temperature and low temperature heat.

To dispatch a CHP generator:

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
```

### EVs

Control a number of EV chargers to charge a number of charge events.  

A `charge_event` is a period of time where an EV can be charged.  This is given as a boolean 2D array, with one binary digit for each pair of charge events and intervals.

Each charge event has a required amount of electricity `charge_event_mwh`, that can be delivered anytime the `charge_event` is 1.

Chargers are configured by `charger_mws`.  

Optimize two 100 MWe chargers for 4 charge events over 5 intervals:

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
```

## Examples & Use Cases

### Assets

Further examples for each asset exist in `./examples`:

```shell
$ ls ./examples
./examples
├── battery.py
├── chp.py
└── evs.py
```

### Price vs. Carbon Optimization

A key feature of `energypylinear` is the ability to optimize for both price and carbon.

We can dispatch a battery to minimize carbon emissions by passing in `objective='carbon'`:

```python
import energypylinear as epl

asset = epl.battery.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)

results = asset.optimize(
  electricity_prices=[100, 50, 200, -100, 0, 200, 100, -100],
  electricity_carbon_intensities = [0.1, 0.2, 0.1, 0.15, 0.01, 0.7, 0.5, 0.01],
  objective='carbon'
)
```

We can compare these results above with a simulation that optimizes for price, using a `energypylinear.accounting.Account` to compare both simulations.  The accounting API is in it's first iteration - expect it to change in the future.

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
price = epl.get_accounts(
  price.interval_data,
  price.simulation,
)
carbon = epl.get_accounts(
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

### Dispatch for Actuals vs. Dispatch for Forecasts

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
print(variance)
# cost=-1197.777778 emissions=0.002222221999999996
```

## Test

```shell
$ make test
```
