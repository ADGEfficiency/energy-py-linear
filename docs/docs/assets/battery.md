# Battery

Dispatch an electric battery operating in wholesale price arbitrage using `epl.Battery`:

```python
import energypylinear as epl

#  optimize a 2.0 MW, 4.0 MWh battery for money
asset = epl.Battery(
    power_mw=2,
    capacity_mwh=4,
    efficiency_pct=0.9,
    electricity_prices=[100.0, 50, 200, -100, 0, 200, 100, -100],
    freq_mins=60,
    initial_charge_mwh=1,
    final_charge_mwh=3,
)
simulation = asset.optimize(objective="price")

#  optimize a 1.0 MW, 3.0 MWh battery for carbon
asset = epl.Battery(
    power_mw=1,
    capacity_mwh=3,
    efficiency_pct=0.9,
    electricity_carbon_intensities = [0.1, 0.2, 0.1, 0.15, 0.01, 0.7, 0.5, 0.01],
    freq_mins=60,
    initial_charge_mwh=0,
    final_charge_mwh=0,
)
simulation = asset.optimize(objective="carbon")

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
        "spill-electric_generation_mwh",
        "spill-electric_load_mwh",
        "spill-high_temperature_generation_mwh",
        "spill-low_temperature_generation_mwh",
        "spill-high_temperature_load_mwh",
        "spill-low_temperature_load_mwh",
        "spill-gas_consumption_mwh",
        "battery-electric_charge_mwh",
        "battery-electric_charge_binary",
        "battery-electric_discharge_mwh",
        "battery-electric_discharge_binary",
        "battery-electric_loss_mwh",
        "battery-initial_charge_mwh",
        "battery-final_charge_mwh",
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

The battery will charge with electricity at low prices & carbon intensities, and discharge at high prices & carbon intensities.

An efficiency penalty is applied to the battery charge energy (energy is lost during charging).
