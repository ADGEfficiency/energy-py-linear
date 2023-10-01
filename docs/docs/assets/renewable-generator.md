# Renewable Generator

Dispatch an renewable generator operating in wholesale price arbitrage using `epl.RenewableGenerator`.

This asset type is suitable for modelling either wind or solar generators.

```python
import energypylinear as epl

asset = epl.RenewableGenerator(
    electricity_prices=[1.0, -0.5],
    electric_generation_mwh=[100, 100],
    electric_generation_lower_bound_pct=0.5,
    name="wind",
)
simulation = asset.optimize(objective="price")

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
        "wind-electric_generation_mwh",
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

This renewable generator will turn down when electricity prices are negative.
