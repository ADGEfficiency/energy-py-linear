# Dispatching Multiple Assets with the Site API

The site API allows optimizing many assets together.

```python
import energypylinear as epl

site = epl.Site(assets=[
  epl.Battery(power_mw=2.0, capacity_mwh=4.0),
  epl.Generator(
    electric_power_max_mw=100,
    electric_power_min_mw=30,
    electric_efficiency_pct=0.4
  )
])

results = site.optimize(
  electricity_prices=[100.0, 50, 200, -100, 0, 200, 100, -100],
  freq_mins=60,
  initial_charge_mwh=1,
  final_charge_mwh=3
)
```

## Fast & Slow Battery

## Multiple CHP Generators

## Battery & EV Chargers
