The `epl.EVs` asset is suitable for modelling electric vehicle charging. One asset can operate many electric vehicle chargers that supply electricity to many electric vehicle charge events.

The electric vehicle asset will optimize the dispatch of the chargers to supply electricity to the charge events. The asset can do both grid to vehicle and vehicle to grid electricity flow with a `allows_evs_discharge` flag.

## Assumptions

Electric vehicle chargers are configured by the charger power output, given as `charger_mws`.

A `charge_event` is a time interval where an electric vehicle can be charged.  This is given as a boolean 2D array, with one binary digit for each charge event, interval pairs.

Each charge event has a required amount of electricity `charge_event_mwh`, that can be delivered when the `charge_event` is 1.  The model is constrained so that each charge event receives all of it's `charge_event_mwh`.

## Use

Optimize two 100 MWe chargers for 4 charge events over 5 intervals:

```python
import energypylinear as epl

electricity_prices = [-100, 50, 30, 50, 40]
charge_events = [
    [1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0],
]

#  2 100 MW EV chargers
asset = epl.EVs(
    chargers_power_mw=[100, 100],
    charge_events_capacity_mwh=[50, 100, 30, 40],
    charger_turndown=0.1,
    electricity_prices=electricity_prices,
    charge_events=charge_events,
)

simulation = asset.optimize()

assert all(
    simulation.results.columns
    == [
        "site-import_power_mwh",
        "site-export_power_mwh",
        "site-electricity_prices",
        "site-electricity_carbon_intensities",
        "site-high_temperature_load_mwh",
        "site-low_temperature_load_mwh",
        "site-low_temperature_generation_mwh",
        "site-gas_prices",
        "site-electric_load_mwh",
        "spill-electric_generation_mwh",
        "spill-electric_load_mwh",
        "spill-high_temperature_generation_mwh",
        "spill-low_temperature_generation_mwh",
        "spill-high_temperature_load_mwh",
        "spill-low_temperature_load_mwh",
        "spill-gas_consumption_mwh",
        "evs-charger-0-electric_charge_mwh",
        "evs-charger-0-electric_charge_binary",
        "evs-charger-0-electric_discharge_mwh",
        "evs-charger-0-electric_discharge_binary",
        "evs-charger-0-electric_loss_mwh",
        "evs-charger-1-electric_charge_mwh",
        "evs-charger-1-electric_charge_binary",
        "evs-charger-1-electric_discharge_mwh",
        "evs-charger-1-electric_discharge_binary",
        "evs-charger-1-electric_loss_mwh",
        "evs-charge-event-0-electric_charge_mwh",
        "evs-charge-event-0-electric_discharge_mwh",
        "evs-charge-event-0-electric_loss_mwh",
        "evs-charge-event-1-electric_charge_mwh",
        "evs-charge-event-1-electric_discharge_mwh",
        "evs-charge-event-1-electric_loss_mwh",
        "evs-charge-event-2-electric_charge_mwh",
        "evs-charge-event-2-electric_discharge_mwh",
        "evs-charge-event-2-electric_loss_mwh",
        "evs-charge-event-3-electric_charge_mwh",
        "evs-charge-event-3-electric_discharge_mwh",
        "evs-charge-event-3-electric_loss_mwh",
        "evs-charge-event-0-initial_soc_mwh",
        "evs-charge-event-1-initial_soc_mwh",
        "evs-charge-event-2-initial_soc_mwh",
        "evs-charge-event-3-initial_soc_mwh",
        "evs-charge-event-0-final_soc_mwh",
        "evs-charge-event-1-final_soc_mwh",
        "evs-charge-event-2-final_soc_mwh",
        "evs-charge-event-3-final_soc_mwh",
        "evs-charger-spill-evs-electric_charge_mwh",
        "evs-charger-spill-evs-electric_charge_binary",
        "evs-charger-spill-evs-electric_discharge_mwh",
        "evs-charger-spill-evs-electric_discharge_binary",
        "evs-charger-spill-evs-electric_loss_mwh",
        "total-electric_generation_mwh",
        "total-electric_load_mwh",
        "total-high_temperature_generation_mwh",
        "total-low_temperature_generation_mwh",
        "total-high_temperature_load_mwh",
        "total-low_temperature_load_mwh",
        "total-gas_consumption_mwh",
        "total-electric_charge_mwh",
        "total-electric_discharge_mwh",
        "total-spills_mwh",
        "total-electric_loss_mwh",
        "site-electricity_balance_mwh",
    ]
)
```

## Validation

A natural response when you get access to something someone else built is to wonder - **does this work correctly?**

This section will give you confidence in the implementation of the EV asset.

### Fully Constrained EV Charging

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

### Expanding a Charge Event Window

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

### Overlapping Charge Events

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

### Adding V2G

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

### Spill Chargers

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
