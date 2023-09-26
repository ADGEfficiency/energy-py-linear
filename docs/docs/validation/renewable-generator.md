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
simulation = asset.optimize(objective="carbon", verbose=False)
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
simulation = asset.optimize(objective="carbon", verbose=False)
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
