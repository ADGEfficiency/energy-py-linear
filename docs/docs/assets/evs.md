# Electric Vehicle Charging

Control a number of EV chargers to charge a number of charge events.

Chargers are configured by their size given in `charger_mws`.

A `charge_event` is a time interval where an EV can be charged.  This is given as a boolean 2D array, with one binary digit for each charge events, interval pairs.

Each charge event has a required amount of electricity `charge_event_mwh`, that can be delivered when the `charge_event` is 1.  The model is constrained so that each charge event receives all of it's `charge_event_mwh`.

To optimize two 100 MWe chargers for 4 charge events over 5 intervals:

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
        "evs-charger-1-electric_charge_mwh",
        "evs-charger-1-electric_charge_binary",
        "evs-charger-1-electric_discharge_mwh",
        "evs-charger-1-electric_discharge_binary",
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
