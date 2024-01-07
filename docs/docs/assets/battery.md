# Battery

The `epl.Battery` model is suitable for modelling an electric battery, such as a lithium-ion battery.

The battery charge rate is defined by `power_mw`, which defines both the maximum rate of charge and discharge. `discharge_power_mw` can be used to define a different rate of maximum discharge.

The battery storage capacity is defined by `capacity_mwh`.  This should be the capacity after taking into account any battery depth of discharge limits.

An efficiency penalty is applied to the battery charge energy, based on the `efficiency_pct` parameter.  No electricity is lost when discharging or during storage.

`initial_charge_mwh` and `final_charge_mwh` control the battery state of charge at the start and end of the simulation.  These can cause infeasible simulations if the battery is not able to charge or discharge enough to meet these constraints.

[You can check the correctness of the battery model here](https://energypylinear.adgefficiency.com/latest/validation/battery/).

```python
import energypylinear as epl

asset = epl.Battery(
    power_mw=2,
    discharge_power_mw=2,
    capacity_mwh=4,
    efficiency_pct=0.9,
    electricity_prices=[100.0, 50, 200, -100, 0, 200, 100, -100],
    freq_mins=60,
    initial_charge_mwh=1,
    final_charge_mwh=3,
    name="battery"
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
        "battery-electric_charge_mwh",
        "battery-electric_charge_binary",
        "battery-electric_discharge_mwh",
        "battery-electric_discharge_binary",
        "battery-electric_loss_mwh",
        "battery-electric_initial_charge_mwh",
        "battery-electric_final_charge_mwh",
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
