# Optimizing an Asset with the Asset API

The asset API allows optimizing a single asset at once.  Internally the assets are using the `epl.Site`, but this is hidden when using the asset API.

[You can find full examples for each asset here](https://github.com/ADGEfficiency/energy-py-linear/tree/main/examples).

## Battery

Dispatch an electric battery operating in wholesale price arbitrage using `epl.Battery`:

```python
import energypylinear as epl

#  2.0 MW, 4.0 MWh battery
asset = epl.battery.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)

results = asset.optimize(
  electricity_prices=[100.0, 50, 200, -100, 0, 200, 100, -100],
  freq_mins=60,
  initial_charge_mwh=1,
  final_charge_mwh=3,
  objective="price"
)

assert all(
    results.simulation.columns
    == [
        'site-import_power_mwh',
        'site-export_power_mwh',
        'spill-electric_generation_mwh',
        'spill-electric_load_mwh',
        'spill-high_temperature_generation_mwh',
        'spill-low_temperature_generation_mwh',
        'spill-high_temperature_load_mwh',
        'spill-low_temperature_load_mwh',
        'spill-gas_consumption_mwh',
        'battery-electric_charge_mwh',
        'battery-electric_charge_binary',
        'battery-electric_discharge_mwh',
        'battery-electric_discharge_binary',
        'battery-electric_loss_mwh',
        'battery-initial_charge_mwh',
        'battery-final_charge_mwh',
        'total-electric_generation_mwh',
        'total-electric_load_mwh',
        'total-high_temperature_generation_mwh',
        'total-low_temperature_generation_mwh',
        'total-high_temperature_load_mwh',
        'total-low_temperature_load_mwh',
        'total-gas_consumption_mwh',
        'total-electric_charge_mwh',
        'total-electric_discharge_mwh',
        'total-spills_mwh',
        'total-electric_loss_mwh',
        'site-electricity_balance_mwh',
        'electricity_prices',
        'electricity_carbon_intensities',
        'load-high_temperature_load_mwh',
        'load-low_temperature_load_mwh',
        'load-low_temperature_generation_mwh'
    ]
)
```

The battery will charge with electricity at low prices, and discharge at high prices.  An efficiency penalty is applied to the battery charge energy (energy is lost during charging).

## Generator

Dispatch a CHP (combined heat & power) generator to generate electricity, high & low temperature heat from natural gas.

The `epl.Generator` model can be configured with electric, high and low temperature thermal efficiencies. 

This allows modelling both gas engines and gas turbines.

When optimizing, we can use interval data for the high and low temperature loads.  These thermal loads will be met by gas boilers if the CHP chooses not to generate, or cannot meet thermal demands.  High temperature heat can be let-down into low temperature heat.

The `epl.Generator` is allowed to dump both high temperature and low temperature heat.

The high and low temperature heat demands are supplied alongside the electricity prices when optimizing:

```python
import energypylinear as epl

#  100 MWe gas turbine
asset = epl.chp.Generator(
    electric_power_max_mw=100,
    electric_power_min_mw=50,
    electric_efficiency_pct=0.3,
    high_temperature_efficiency_pct=0.5,
)

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

assert all(
    results.simulation.columns == [
        'site-import_power_mwh',
        'site-export_power_mwh',
        'spill-electric_generation_mwh',
        'spill-electric_load_mwh',
        'spill-high_temperature_generation_mwh',
        'spill-low_temperature_generation_mwh',
        'spill-high_temperature_load_mwh',
        'spill-low_temperature_load_mwh',
        'spill-gas_consumption_mwh',
        'generator-electric_generation_mwh',
        'generator-gas_consumption_mwh',
        'generator-high_temperature_generation_mwh',
        'generator-low_temperature_generation_mwh',
        'boiler-high_temperature_generation_mwh',
        'boiler-gas_consumption_mwh',
        'valve-high_temperature_load_mwh',
        'valve-low_temperature_generation_mwh',
        'total-electric_generation_mwh',
        'total-electric_load_mwh',
        'total-high_temperature_generation_mwh',
        'total-low_temperature_generation_mwh',
        'total-high_temperature_load_mwh',
        'total-low_temperature_load_mwh',
        'total-gas_consumption_mwh',
        'total-electric_charge_mwh',
        'total-electric_discharge_mwh',
        'total-spills_mwh',
        'total-electric_loss_mwh',
        'site-electricity_balance_mwh',
        'electricity_prices',
        'electricity_carbon_intensities',
        'load-high_temperature_load_mwh',
        'load-low_temperature_load_mwh',
        'load-low_temperature_generation_mwh'
    ]
)
```

## Electric Vehicle Charging

Control a number of EV chargers to charge a number of charge events.  

Chargers are configured by their size given in `charger_mws`.  

A `charge_event` is a time interval where an EV can be charged.  This is given as a boolean 2D array, with one binary digit for each charge events, interval pairs.

Each charge event has a required amount of electricity `charge_event_mwh`, that can be delivered when the `charge_event` is 1.  The model is constrained so that each charge event receives all of it's `charge_event_mwh`.

To optimize two 100 MWe chargers for 4 charge events over 5 intervals:

```python
import energypylinear as epl

#  2 100 MW EV chargers
asset = epl.EVs(
    chargers_power_mw=[100, 100],
    charge_events_capacity_mwh = [50, 100, 30, 40],
    charger_turndown=0.1
)

electricity_prices = [-100, 50, 30, 50, 40]
charge_events = [
    [1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0],
]

results = asset.optimize(
    electricity_prices=electricity_prices,
    charge_events=charge_events,
)

assert all(
    results.simulation.columns == [
        'site-import_power_mwh',
        'site-export_power_mwh',
        'spill-electric_generation_mwh',
        'spill-electric_load_mwh',
        'spill-high_temperature_generation_mwh',
        'spill-low_temperature_generation_mwh',
        'spill-high_temperature_load_mwh',
        'spill-low_temperature_load_mwh',
        'spill-gas_consumption_mwh',
        'evs-charger-0-electric_charge_mwh',
        'evs-charger-0-electric_charge_binary',
        'evs-charger-0-electric_discharge_mwh',
        'evs-charger-0-electric_discharge_binary',
        'evs-charger-1-electric_charge_mwh',
        'evs-charger-1-electric_charge_binary',
        'evs-charger-1-electric_discharge_mwh',
        'evs-charger-1-electric_discharge_binary',
        'evs-charge-event-0-electric_charge_mwh',
        'evs-charge-event-0-electric_discharge_mwh',
        'evs-charge-event-0-electric_loss_mwh',
        'evs-charge-event-1-electric_charge_mwh',
        'evs-charge-event-1-electric_discharge_mwh',
        'evs-charge-event-1-electric_loss_mwh',
        'evs-charge-event-2-electric_charge_mwh',
        'evs-charge-event-2-electric_discharge_mwh',
        'evs-charge-event-2-electric_loss_mwh',
        'evs-charge-event-3-electric_charge_mwh',
        'evs-charge-event-3-electric_discharge_mwh',
        'evs-charge-event-3-electric_loss_mwh',
        'evs-charge-event-0-initial_soc_mwh',
        'evs-charge-event-1-initial_soc_mwh',
        'evs-charge-event-2-initial_soc_mwh',
        'evs-charge-event-3-initial_soc_mwh',
        'evs-charge-event-0-final_soc_mwh',
        'evs-charge-event-1-final_soc_mwh',
        'evs-charge-event-2-final_soc_mwh',
        'evs-charge-event-3-final_soc_mwh',
        'evs-charger-spill-evs-electric_charge_mwh',
        'evs-charger-spill-evs-electric_charge_binary',
        'evs-charger-spill-evs-electric_discharge_mwh',
        'evs-charger-spill-evs-electric_discharge_binary',
        'total-electric_generation_mwh',
        'total-electric_load_mwh',
        'total-high_temperature_generation_mwh',
        'total-low_temperature_generation_mwh',
        'total-high_temperature_load_mwh',
        'total-low_temperature_load_mwh',
        'total-gas_consumption_mwh',
        'total-electric_charge_mwh',
        'total-electric_discharge_mwh',
        'total-spills_mwh',
        'total-electric_loss_mwh',
        'site-electricity_balance_mwh',
        'electricity_prices',
        'electricity_carbon_intensities',
        'load-high_temperature_load_mwh',
        'load-low_temperature_load_mwh',
        'load-low_temperature_generation_mwh'
    ]
)
```