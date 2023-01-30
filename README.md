# energy-py-linear

A Python library for optimizing the dispatch of energy assets with mixed-integer linear programming.

# Use

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
  electricity_prices=[100, 50, 200, -100, 0, 200, 100, -100],
  freq_mins=60,
  initial_charge_mwh=1,
  final_charge_mwh=3
)
```

## CHP

Dispatch a Combined Heat & Power unit to generate high price electricity:

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
results = asset.optimize(electricity_prices=[100, 50, 200, -100, 0, 200, 100, -100])
```

## EVs

Control a number of EV chargers to charge each charge event with a required amount of electricity:

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

## Price versus Carbon Optimization

A key feature of `energypylinear` is the ability to optimize for both price and carbon side by side.

We can dispatch a battery for carbon by passing in `objective='carbon'`:

```python
import energypylinear as epl

asset = epl.battery.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)
results = asset.optimize(
  electricity_prices=[100, 50, 200, -100, 0, 200, 100, -100],
  electricity_carbon_intensities = [0.1, 0.2, 0.1, 0.15, 0.01, 0.7, 0.5, 0.01],
  objective='carbon'
)
```

We can compare the results above with a simulation that optimizes for price.

We can use an `energypylinear.accounting.Account` to compare both simulations.

This accounting API is in it's first iteration - expect it to change in the future:

```python
import energypylinear as epl

electricity_prices = [100, 50, 200, -100, 0, 200, 100, -100]
electricity_carbon_intensities = [0.1, 0.2, 0.1, 0.15, 0.01, 0.7, 0.5, 0.01]
asset = epl.battery.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)
price = asset.optimize(electricity_prices=electricity_prices)
carbon = asset.optimize(
  electricity_prices=electricity_prices,
  electricity_carbon_intensities=electricity_carbon_intensities,
  objective='carbon'
)

account = epl.accounting.get_accounts(
  price.interval_data,
  price.simulation,
  carbon.simulation,
)
```
Actuals are our optimization for price - we have a high negative cost:

```python
print(account.actuals)
Account(cost=-1057.777778, emissions=0.08222222199999996)
```

Forecast here represents our alternative scenario where we optimize for emissions reduction 

We have lower (better) emissions when we optimize for carbon, but at a higher cost (we make less money):

```python
print(account.forecast)
Account(cost=-134.44444399999998, emissions=0.06444444400000005)
```

# Test

```shell
$ make test
```
