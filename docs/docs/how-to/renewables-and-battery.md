# Renewable Generator, Battery and Site Limits

`energypylinear` has the ability to optimize a battery located with renewable electricity generation.

This guide shows how to model a site with a renewable generator and battery that can export electricity to the grid.

## Basic Setup

First, let's set up a site with two assets - a solar generator and battery, with a site export limit of 25 MW:

<!--phmdoctest-share-names-->
```python
import energypylinear as epl

# Create assets:
# - 10 MW / 20 MWh battery with 90% round-trip efficiency
# - Solar generator with a predefined generation profile (10-30 MW across 5 intervals)
# - Generator can be curtailed down to 50% of available generation
# - Custom name "solar" for the generator
assets = [
    epl.Battery(power_mw=10, capacity_mwh=20, efficiency_pct=0.9),
    epl.RenewableGenerator(
        electric_generation_mwh=[10, 20, 30, 20, 10],
        electric_generation_lower_bound_pct=0.5,
        name="solar",
    ),
]

# Configure site:
# - Carbon intensity values for each interval (negative values represent low carbon periods)
# - Maximum export limit of 25 MW to the grid
site = epl.Site(
    assets=assets,
    electricity_carbon_intensities=[0.5, -0.5, 0.5, 0.5, -0.5],
    export_limit_mw=25,
)

# Optimize for carbon minimization rather than profit maximization
simulation = site.optimize(objective="carbon")
```

## Examining the Results

Let's examine the results of our optimization to see how the battery and solar plant work together under carbon optimization.

<!--phmdoctest-share-names-->
```python
print(
    simulation.results[
        [
            "site-electricity_carbon_intensities",
            "site-export_limit_mw",
            "site-export_power_mwh",
            "solar-electric_generation_used_mwh",
            "battery-electric_charge_mwh",
            "battery-electric_discharge_mwh",
            "battery-electric_final_charge_mwh",
        ]
    ]
)
```

This gives us the following output:

```
   site-electricity_carbon_intensities  site-export_limit_mw  site-export_power_mwh  solar-electric_generation_used_mwh  battery-electric_charge_mwh  battery-electric_discharge_mwh  battery-electric_final_charge_mwh
0                                  0.5                    25                  10.00                                10.0                        0.00                          0.00                                0.00
1                                 -0.5                    25                  25.00                                20.0                        5.00                          0.00                                4.50
2                                  0.5                    25                  25.00                                30.0                        0.00                          5.00                                0.00
3                                  0.5                    25                  15.00                                20.0                        0.00                          5.00                                0.00
4                                 -0.5                    25                  10.00                                10.0                        0.00                          0.00                                0.00
```

## Results Interpretation

When optimized for carbon, we can observe the following behavior:

1. During periods with negative carbon intensity (intervals 1 and 4):
   - In interval 1: The battery charges 5 MWh while solar generates 20 MWh, maximizing export to the grid up to the 25 MW limit
   - In interval 4: Solar generation is only 10 MWh, which is fully exported

2. During periods with positive carbon intensity (intervals 0, 2, and 3):
   - The battery discharges in intervals 2 and 3 to maximize export during high carbon periods
   - The battery and solar work together to respect the 25 MW export limit

This strategy minimizes overall carbon emissions by:
- Maximizing export during negative carbon periods (displacing high-carbon grid generation)
- Using the battery to shift energy between periods strategically
