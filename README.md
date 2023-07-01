# energy-py-linear

<img src="./static/coverage.svg"> [![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

---

Documentation: [energypylinear.adgefficiency.com](https://energypylinear.adgefficiency.com/latest)

---

A Python library for optimizing energy assets with mixed-integer linear programming:

- electric batteries,
- combined heat & power (CHP) generators,
- electric vehicle smart charging.

Assets can be optimized to either maximize profit or minimize carbon emissions.  

Energy balances are performed on electricity, high & low temperature heat.

## Setup

Requires Python 3.10+:

```shell
$ pip install energypylinear
```

## Quick Start

### Asset API

The asset API allows optimizing a single asset at once.

We can optimize an electric battery operating in wholesale price arbitrage using `epl.Battery`:

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

You can find documentation of how to optimize other assets in [how-to/optimize-assets](https://docs.adgefficiency.com/energy-py-linear/how-to/dispach-assets), and Python examples in [energy-py-linear/examples/examples](https://github.com/ADGEfficiency/energy-py-linear/tree/main/examples).

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
  ),
  epl.evs.EVs(charger_mws=[100, 100])
])

results = site.optimize(
  electricity_prices=[100, 50, 200, -100, 0],
  high_temperature_load_mwh=[105, 110, 120, 110, 105],
  low_temperature_load_mwh=[105, 110, 120, 110, 105],
  freq_mins=60,
  initial_charge_mwh=1,
  final_charge_mwh=3,
  charge_events=[
      [1, 0, 0, 0, 0],
      [0, 1, 1, 1, 0],
      [0, 0, 0, 1, 1],
      [0, 1, 0, 0, 0],
  ],
  charge_event_mwh=[50, 100, 30, 40]
)
```

The site API will optimize the assets together, and return the results for each asset. 

### Examples

Examples as independent scripts are `./examples`:

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

## Documentation 

Documentation is hosted at [energypylinear.adgefficiency.com](https://energypylinear.adgefficiency.com/latest).
