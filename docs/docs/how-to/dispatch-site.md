# Multiple Assets with the Site API

`energypylinear` can optimize many assets in a single linear program.

The `epl.Site` accepts a list of `energypylinear` asset models, like `epl.Battery` or `epl.RenewableGenerator`.

Below are examples of typical configurations of multiple energy assets using a `epl.Site`.

## Fast & Slow Battery

Optimize a fast and slow battery alongside each other:

```python
import energypylinear as epl

site = epl.Site(
    assets=[
        epl.Battery(
            power_mw=4.0,
            capacity_mwh=1.0,
            initial_charge_mwh=1,
            final_charge_mwh=1
        ),
        epl.Battery(
            power_mw=2.0,
            capacity_mwh=4.0,
            initial_charge_mwh=4.0,
            final_charge_mwh=0.0,
            name="battery-2",
        ),
    ],
    electricity_prices=[100.0, 50, 200, -100, 0, 200, 100, -100],
    freq_mins=60,
)

simulation = site.optimize()
```

## Battery & EV Chargers

Optimize a battery next to an EV charging station:

```python
import energypylinear as epl

site = epl.Site(
    assets=[
        epl.Battery(
            power_mw=2.0,
            capacity_mwh=4.0,
            initial_charge_mwh=1,
            final_charge_mwh=3,
        ),
        epl.EVs(
            chargers_power_mw=[100, 100],
            charger_turndown=0.1,
            charge_events=[
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
            ],
            charge_events_capacity_mwh=[50, 100, 30, 40],
        ),
    ],
    electricity_prices=[100.0, 50, 200, -100, 0, 200, 100, -100],
    freq_mins=60,
)

simulation = site.optimize()
```

## Battery & CHP

Optimize an electric battery alongside a gas fired CHP:

```python
import energypylinear as epl

site = epl.Site(
    assets=[
        epl.Battery(
            power_mw=2.0,
            capacity_mwh=4.0,
            initial_charge_mwh=1,
            final_charge_mwh=3
        ),
        epl.CHP(
            electric_power_max_mw=100,
            electric_power_min_mw=30,
            electric_efficiency_pct=0.4,
        ),
    ],
    electricity_prices=[100.0, 50, 200, -100, 0, 200, 100, -100],
    freq_mins=60,
)

simulation = site.optimize()
```
