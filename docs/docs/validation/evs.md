A natural response when you get access to something someone else built is to wonder - **does this work correctly?**

This section will give you confidence in the implementation of the EV asset.

## Fully Constrained EV Charging

```python
import energypylinear as epl

asset = epl.EVs(
    chargers_power_mw=[100, 100],
    charge_events_capacity_mwh=[50, 100, 30],
    charger_turndown=0.0,
    charge_event_efficiency=1.0,
    electricity_prices=[-100, 50, 30, 10, 40],
    charge_events=[
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ]
)
simulation = asset.optimize()
asset.plot(simulation, path="./docs/docs/static/ev-validation-1.png")
```

The third charger is the spill charger.

![](../static/ev-validation-1.png)

## Expanding a Charge Event Window

Let's expand out the charge event window to the last three intervals for the last charge event:

```python
import energypylinear as epl

asset = epl.EVs(
    chargers_power_mw=[100, 100],
    charge_events_capacity_mwh=[50, 100, 30],
    charger_turndown=0.0,
    charge_event_efficiency=1.0,
    electricity_prices=[-100, 50, 300, 10, 40],
    charge_events=[
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
    ]
)
simulation = asset.optimize()
asset.plot(simulation, path="./docs/docs/static/ev-validation-2.png")
```

Now we see that the charge has happened in interval 3, this is because electricity prices are lowest in this interval.

![](../static/ev-validation-2.png)

## Overlapping Charge Events

When charge events overlap at low prices, both (but only two) chargers are used:

```python
import energypylinear as epl

asset = epl.EVs(
    chargers_power_mw=[100, 100],
    charge_events_capacity_mwh=[50, 100, 30],
    charger_turndown=0.0,
    charge_event_efficiency=1.0,
    electricity_prices=[-100, 50, 300, 10, 40],
    charge_events=[
        [1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
    ]
)
simulation = asset.optimize()
asset.plot(simulation, path="./docs/docs/static/ev-validation-3.png")
```

![](../static/ev-validation-3.png)

## Adding V2G

```python
import energypylinear as epl

asset = epl.EVs(
    chargers_power_mw=[100, 100],
    charge_events_capacity_mwh=[50, 100, 30],
    charger_turndown=0.0,
    charge_event_efficiency=1.0,
    electricity_prices=[-100, 50, 300, 10, 40],
    charge_events=[
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
    ],
)
simulation = asset.optimize(
    flags=epl.Flags(allow_evs_discharge=True)
)
asset.plot(simulation, path="./docs/docs/static/ev-validation-4.png")
```

![](../static/ev-validation-4.png)

The key takeaway here is that we discharge during interval 2.  All our charge events still end up at the correct state of charge at the end of the program.

## Spill Chargers

```python
import energypylinear as epl

asset = epl.EVs(
    chargers_power_mw=[100, 100],
    charge_events_capacity_mwh=[50, 100, 30, 500],
    charger_turndown=0.0,
    charge_event_efficiency=1.0,
    electricity_prices=[-100, 50, 300, 10, 40],
    charge_events=[
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0],
    ],
)
simulation = asset.optimize(
    flags=epl.Flags(allow_evs_discharge=True)
)
asset.plot(simulation, path="./docs/docs/static/ev-validation-5.png")
```

Key takeaway here is the use of the spill charger - we have a 500 MWh charge event, but only 200 MWh of capacity.  We meet the remaining demand from a spill charger.

This allows the linear program to be feasible, while communicating directly which intervals or charge events are causing the mismatch between charge event demand and spill charger capacity.

![](../static/ev-validation-5.png)
