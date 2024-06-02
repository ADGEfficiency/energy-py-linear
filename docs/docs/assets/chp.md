The `epl.CHP` (combined heat & power) model can generate electricity, high & low temperature heat from natural gas.

## Assumptions

The `epl.CHP` is configured with electric, high and low temperature thermal efficiencies. This allows modelling open cycle generators, gas engines and gas turbines.

When optimizing, we can use interval data for the high and low temperature loads.  

The high and low temperature loads will be met by gas boilers if the CHP chooses not to generate, or cannot meet thermal demands.  High temperature heat can be let-down into low temperature heat.

The `epl.CHP` is allowed to dump both high temperature and low temperature heat.

## Use

You can optimize a single CHP with `epl.CHP`:

```python
import energypylinear as epl

#  100 MWe gas turbine
asset = epl.CHP(
    electric_power_max_mw=100,
    electric_power_min_mw=50,
    electric_efficiency_pct=0.3,
    high_temperature_efficiency_pct=0.5,
)

#  100 MWe gas engine
asset = epl.CHP(
    electric_power_max_mw=100,
    electric_power_min_mw=10,
    electric_efficiency_pct=0.4,
    high_temperature_efficiency_pct=0.2,
    low_temperature_efficiency_pct=0.2,
    electricity_prices=[100, 50, 200, -100, 0, 200, 100, -100],
    high_temperature_load_mwh=[100, 50, 200, 40, 0, 200, 100, 100],
    low_temperature_load_mwh=20,
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
        "chp-electric_generation_mwh",
        "chp-gas_consumption_mwh",
        "chp-high_temperature_generation_mwh",
        "chp-low_temperature_generation_mwh",
        "boiler-high_temperature_generation_mwh",
        "boiler-gas_consumption_mwh",
        "valve-high_temperature_load_mwh",
        "valve-low_temperature_generation_mwh",
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
