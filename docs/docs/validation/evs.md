A natural response when you get access to something someone else build is to wonder - **does this work correctly?**

This section will give you confidence in the implementation of the EV implementation.

## Fully Constrained EV Charging

```python
import energypylinear as epl

asset = epl.EVs(
    chargers_power_mw=[100, 100],
    charge_events_capacity_mwh=[50, 100, 30],
    charger_turndown=0.0,
    charge_event_efficiency=1.0
)
results = asset.optimize(
    electricity_prices=[-100, 50, 30, 10, 40],
    charge_events=[
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ]
)
asset.plot(results, path="./docs/docs/static/ev-validation-1.png")
```

Note third charger - this is the spill charger

![](../static/ev-validation-1.png)

## Expanding a Charge Event Window

```python
import energypylinear as epl

asset = epl.EVs(
    chargers_power_mw=[100, 100],
    charge_events_capacity_mwh=[50, 100, 30],
    charger_turndown=0.0,
    charge_event_efficiency=1.0
)
results = asset.optimize(
    electricity_prices=[-100, 50, 300, 10, 40],
    charge_events=[
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
    ]
)
asset.plot(results, path="./docs/docs/static/ev-validation-2.png")
```

Now we see that the charge has happened in interval 3, this is because electricity prices are lowest in this interval.

![](../static/ev-validation-2.png)

## Overlapping Charge Events

```python
import energypylinear as epl

asset = epl.EVs(
    chargers_power_mw=[100, 100],
    charge_events_capacity_mwh=[50, 100, 30],
    charger_turndown=0.0,
    charge_event_efficiency=1.0
)
results = asset.optimize(
    electricity_prices=[-100, 50, 300, 10, 40],
    charge_events=[
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
    ]
)
asset.plot(results, path="./docs/docs/static/ev-validation-3.png")
```

![](../static/ev-validation-3.png)

we have moved our 100 MWh charging into the third interval

our first charge event is still at the negative price because splitting our 100 MWh load would be worse

do the math here just as a fyi

## Adding V2G

```python
import energypylinear as epl

asset = epl.EVs(
    chargers_power_mw=[100, 100],
    charge_events_capacity_mwh=[50, 100, 30],
    charger_turndown=0.0,
    charge_event_efficiency=1.0
)
results = asset.optimize(
    electricity_prices=[-100, 50, 300, 10, 40],
    charge_events=[
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
    ],
    flags=epl.Flags(allow_evs_discharge=True)
)
asset.plot(results, path="./docs/docs/static/ev-validation-4.png")
```

![](../static/ev-validation-4.png)

## Charge Event Final Charges
