# energy-py-linear

<img src="./static/coverage.svg"> [![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

---

Documentation: [energypylinear.adgefficiency.com](https://energypylinear.adgefficiency.com/latest)

---

A Python library for optimizing energy assets with mixed-integer linear programming:

- electric batteries,
- combined heat & power (CHP) generators,
- electric vehicle smart charging,
- heat pumps,
- renewable (wind & solar) generators.

Assets can be optimized to either maximize profit or minimize carbon emissions, or for user defined custom objective functions. Custom constraints can be used to further constrain asset behaviour.

A site is a collection of assets that can be optimized together. Sites can use custom objectives and constraints.

Energy balances are performed on electricity, high, and low temperature heat.

## Setup

Requires Python 3.11 or 3.12:

```shell-session
$ pip install energypylinear
```

## Quick Start

### Asset API

The asset API allows optimizing a single asset at once:

```python
import energypylinear as epl

#  2.0 MW, 4.0 MWh battery
asset = epl.Battery(
    power_mw=2,
    capacity_mwh=4,
    efficiency_pct=0.9,
    # different electricity prices for each interval
    # length of electricity_prices is the length of the simulation
    electricity_prices=[100.0, 50, 200, -100, 0, 200, 100, -100],
    # a constant value for each interval
    export_electricity_prices=40,
)

simulation = asset.optimize()
```

### Site API

The site API allows optimizing multiple assets together:

```python
import energypylinear as epl

assets = [
    #  2.0 MW, 4.0 MWh battery
    epl.Battery(power_mw=2.0, capacity_mwh=4.0),
    #  30 MW open cycle generator
    epl.CHP(
        electric_power_max_mw=100, electric_power_min_mw=30, electric_efficiency_pct=0.4
    ),
    #  2 EV chargers & 4 charge events
    epl.EVs(
        chargers_power_mw=[100, 100],
        charge_events_capacity_mwh=[50, 100, 30, 40],
        charge_events=[
            [1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0],
        ],
    ),
    #  natural gas boiler to generate high temperature heat
    epl.Boiler(),
    #  valve to generate low temperature heat from high temperature heat
    epl.Valve(),
]

site = epl.Site(
    assets=assets,
    # length of energy prices is the length of the simulation
    electricity_prices=[100, 50, 200, -100, 0],
    # these should match the length of the export_electricity_prices
    # if they don't, they will be repeated or cut to match the length of electricity_prices
    high_temperature_load_mwh=[105, 110, 120, 110, 105],
    low_temperature_load_mwh=[105, 110, 120, 110, 105],
)

simulation = site.optimize()
```

## Documentation

[See more asset types & use cases in the documentation](https://energypylinear.adgefficiency.com/latest).

## Test

```shell
$ make test
```
