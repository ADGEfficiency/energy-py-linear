# Renewable Generator, Battery and Site Limits

`energypylinear` has the ability to optimize a battery located with renewable electricity generation.

This example shows how to model a site with a renewable generator and battery that can export electricity to the grid.

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
