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

[You can check the correctness of the renewable generator model here](https://energypylinear.adgefficiency.com/latest/validation/renewable-generator/).

## Validation

A natural response when you get access to something someone else built is to wonder - does this work correctly?

This section will give you confidence in the implementation of the renewable generator asset.

## Carbon Dispatch Behaviour

Let's optimize the renewable generator asset in two intervals:

1. a positive import electricity carbon intensity of `1.0 tC/MWh`,
2. a negative import electricity carbon intensity of `-0.5 tC/MWh`.

If we optimize our `epl.RenewableGenerator` asset with a lower bound on the electricity generation of `1.0`, we generate the full `100 MW` in each interval:

```python
import energypylinear as epl

electricity_carbon_intensities = [1.0, -0.5]
electric_generation_mwh=[100, 100]
electric_generation_lower_bound_pct=1.0

asset = epl.RenewableGenerator(
    electricity_carbon_intensities=electricity_carbon_intensities,
    electric_generation_mwh=electric_generation_mwh,
    name="wind",
    electric_generation_lower_bound_pct=electric_generation_lower_bound_pct
)
simulation = asset.optimize(objective="carbon", verbose=3)
print(simulation.results[
    [
        "site-electricity_carbon_intensities",
        "site-export_power_mwh",
        "wind-electric_generation_mwh",
    ]
])
```

```
   site-electricity_carbon_intensities  site-export_power_mwh  wind-electric_generation_mwh
0                                  1.0                  100.0                         100.0
1                                 -0.5                  100.0                         100.0
```

If we change our lower bound to `0.5`, our renewable generator asset will generate less electricity during the second, negative carbon intensity interval:

```python
import energypylinear as epl

electricity_carbon_intensities = [1.0, -0.5]
electric_generation_mwh=[100, 100]
electric_generation_lower_bound_pct=0.5

asset = epl.RenewableGenerator(
    electricity_carbon_intensities=electricity_carbon_intensities,
    electric_generation_mwh=electric_generation_mwh,
    name="wind",
    electric_generation_lower_bound_pct=electric_generation_lower_bound_pct
)
simulation = asset.optimize(objective="carbon", verbose=3)
print(simulation.results[
    [
        "site-electricity_carbon_intensities",
        "site-export_power_mwh",
        "wind-electric_generation_mwh",
    ]
])
```

```
   site-electricity_carbon_intensities  site-export_power_mwh  wind-electric_generation_mwh
0                                  1.0                  100.0                         100.0
1                                 -0.5                   50.0                          50.0
```
