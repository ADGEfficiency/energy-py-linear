# Multiple Assets with the Site API

The site API allows optimizing many assets at the same time.

Multiple assets can be optimized using a single `epl.Site`, which accepts a list of assets.

Below we give some examples of typical configurations of energy assets.

## Battery & CHP

We can use a site to optimize an electric battery alongside a gas fired CHP:

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

## Fast & Slow Battery

We can use a site to optimize a fast and slow battery alongside each other:

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

We can use a site to optimize a battery that sits alongside EV chargers:

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
