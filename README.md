# energy-py-linear

A Python library for optimizing the dispatch of energy assets with mixed-integer linear programming.

# Use

## Setup

Requires Python 3.10:

```shell
$ pip install energypylinear
```

## Battery

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

```python
import energypylinear as epl

#  2 100 MW EV chargers
asset = epl.evs.EVs(charger_mws=[100, 100])

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

# Price versus Carbon Optimization

Dispatch a battery for price:

```python
import energypylinear as epl

asset = epl.battery.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)
results = asset.optimize(electricity_prices=[100, 50, 200, -100, 0, 200, 100, -100])
```

Dispatch a battery for carbon:

```python
import energypylinear as epl

asset = epl.battery.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)
results = asset.optimize(
  electricity_prices=[100, 50, 200, -100, 0, 200, 100, -100],
  electricity_carbon_intensities = [0.1, 0.2, 0.1, 0.15, 0.01, 0.7, 0.5, 0.01],
  objective='carbon'
)
```

# Test

```shell
$ make test
```
