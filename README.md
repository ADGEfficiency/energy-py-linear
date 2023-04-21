# energy-py-linear

<img src="./static/coverage.svg"> [![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

---

Documentation: https://docs.adgefficiency.com/energy-py-linear

---

A Python library for optimizing energy assets with mixed-integer linear programming:

- electric batteries,
- combined heat & power (CHP) generators,
- electric vehicle smart charging.

Assets can be optimized to either maximize profit or minimize carbon emissions.

## Setup

Requires Python 3.10+:

```shell
$ pip install energypylinear
```

## Quick Start

### Asset API

The asset API allows optimizing a single asset at once.

We can dispatch an electric battery operating in wholesale price arbitrage using `epl.Battery`:

```python
import energypylinear as epl

#  2.0 MW, 4.0 MWh battery
asset = epl.battery.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)

results = asset.optimize(
  electricity_prices=[100.0, 50, 200, -100, 0, 200, 100, -100],
  freq_mins=60,
  initial_charge_mwh=1,
  final_charge_mwh=3
)
```

### Site API

The site API allows optimizing multiple assets at once:

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

## Examples

Examples are in `./examples`:

```shell
$ ls ./examples
./examples
├── battery.py
├── chp.py
├── evs.py
└── forecast-accuracy.py
```

## Test

```shell
$ make test
```
