# Changelog

## [1.4.1](https://github.com/ADGEfficiency/energy-py-linear/releases/tag/v1.4.1)

### Custom Interval Data

Assets now support custom interval data, allowing you to pass additional time series data that can be used in custom objective functions.

```python
import energypylinear as epl

asset = epl.Battery(
    electricity_prices=[100, 200, 300],
    export_electricity_prices=[90, 180, 270],
    custom_interval_data={"network_charge": [10, 20, 30]}
)
```

[Read more about custom interval data in the documentation](https://energypylinear.adgefficiency.com/latest/how-to/custom-interval-data/).

### Network Charges

New documentation showing how to model network charges in your optimization, using custom interval data and objective functions.

[Read more about network charges in the documentation](https://energypylinear.adgefficiency.com/latest/how-to/network-charges/).

### Battery Degradation How-To

New documentation showing how to model battery degradation costs using custom constraints and objective functions.

[Read more about battery degradation in the documentation](https://energypylinear.adgefficiency.com/latest/how-to/battery-degradation/).

### Python Version

Python 3.10 is no longer supported. Only Python 3.11 and 3.12 are supported.

### Other Changes

- Renewables with battery example documentation.
- Updated pandera dependency.
- Documentation improvements.

## [1.4.0](https://github.com/ADGEfficiency/energy-py-linear/releases/tag/v1.4.0)

### Custom Constraints

It's now possible to add custom constraints to the linear program.

The example below shows how to add a constraint on battery cycles:

```python
import energypylinear as epl
import numpy as np

np.random.seed(42)
cycle_limit_mwh = 30
asset = epl.Battery(
    power_mw=1,
    capacity_mwh=2,
    efficiency_pct=0.98,
    electricity_prices=np.random.normal(0.0, 1000, 48 * 7),
    constraints=[
        epl.Constraint(
            lhs=[
                epl.ConstraintTerm(
                    asset_type="battery", variable="electric_charge_mwh"
                ),
                epl.ConstraintTerm(
                    asset_type="battery", variable="electric_discharge_mwh"
                ),
            ],
            rhs=cycle_limit,
            sense="le",
            interval_aggregation="sum",
        )
    ],
)
```

[Read more about custom constraints in the documentation](https://energypylinear.adgefficiency.com/latest/how-to/custom-constraints/).

### Documentation Refactor

We have moved the asset validation documentation into the documentation for the assets.

A new section `Customization` has been added to the documentation, which contains the documentation for custom constraints and objective functions.

## [1.3.0](https://github.com/ADGEfficiency/energy-py-linear/releases/tag/v1.3.0)

### Different Battery Charge and Discharge Rates

It's now possible to define a different charge and discharge rate in the `epl.Battery` asset.

The example below defines a maximum charge and discharge rate of `2.0`:

```python
epl.Battery(power_mw=2.0)
```

The example below defines a maximum charge rate of `2.0` with a maximum discharge rate of `1.0`:

```python
epl.Battery(power_mw=2.0, discharge_power_mw=1.0)
```

### Complex Objective Function Terms

A complex custom objective term allows you to construct an objective function with a complex set of costs and revenues.

For example, we can define an objective function that includes a cost for the maximum import above a threshold of `40`:

```python
{
    "function": "max_many_variables",
    "variables": {
        "asset_type": "site",
        "variable": "import_power_mwh",
    },
    "constant": 40,
    "coefficient": 200,
    "M": max(electric_load_mwh) * 10
}
```

See [Complex Objective Function Terms](https://energypylinear.adgefficiency.com/latest/how-to/complex-terms) in the documentation for more examples.

### Custom Accounts

To accommodate complex custom objective functions, we have added the ability to include these custom costs and revenues as a custom account:

```python
import energypylinear as epl

chp_size = 50
electric_efficiency = 0.5
electric_load_mwh = 0
electricity_prices = np.array([-1000, -750, -250, -100, 0, 10, 100, 1000])
export_charge = -500
export_threshold_mwh = 5
gas_prices = 20

assets = [
    epl.CHP(
        electric_efficiency_pct=electric_efficiency,
        electric_power_max_mw=chp_size,
    )
]
site = epl.Site(
    assets=assets,
    gas_prices=20,
    electricity_prices=np.array([-1000, -750, -250, -100, 0, 10, 100, 1000]),
    electric_load_mwh=electric_load_mwh,
)

terms: list[dict] = [
    {
        "asset_type": "site",
        "variable": "export_power_mwh",
        "interval_data": "electricity_prices",
        "coefficient": -1,
    },
    {
        "asset_type": "*",
        "variable": "gas_consumption_mwh",
        "interval_data": "gas_prices",
    },
    {
        "type": "complex",
        "function": "min_two_variables",
        "a": {
            "asset_type": "site",
            "variable": "export_power_mwh",
        },
        "b": 5.0,
        "coefficient": export_charge,
        "M": (
            electric_load_mwh
            + assets[0].cfg.electric_power_max_mw
            + export_threshold_mwh
        )
        * 1,
    },
]

simulation = site.optimize(
    verbose=4,
    objective={"terms": terms},
)

accounts = epl.get_accounts(simulation.results, custom_terms=terms[-1:])
print(accounts.custom)
```

```
<Account profit=15000.00 emissions=0.0000>
```

### Optimization Status

The objective function value has been added to the `epl.optimizer.OptimizationStatus` object:

```python
import energypylinear as epl

site = epl.Site(
    assets=[epl.Battery()],
    electricity_prices=np.array([-1000, -750, -250, -100, 0, 10, 100, 1000]),
)
simulation = site.optimize(verbose=4, objective="price")
print(simulation.status)
```

```
OptimizationStatus(status='Optimal', feasible=True, objective=-5811.11111)
```

## [1.2.0](https://github.com/ADGEfficiency/energy-py-linear/releases/tag/v1.2.0)

### Custom Objective Functions

A custom objective function allows users to create their own objective functions in the linear program.  

This allows users to optimize for a custom set of revenues and costs. The objective function can target assets by type or name, and can include multiplication by interval data and/or a coefficient.

The example below shows how to include a cost for battery use (a cycle cost) applied to the battery discharge:

```python
import numpy as np
import energypylinear as epl

assets = [
    epl.Battery(power_mw=20, capacity_mwh=20)
]
site = epl.Site(
    assets=assets,
    electricity_prices=np.random.normal(0, 1000, 48)
)
terms=[
    {
        "asset_type":"site",
        "variable":"import_power_mwh",
        "interval_data":"electricity_prices"
    },
    {
        "asset_type":"site",
        "variable":"export_power_mwh",
        "interval_data":"electricity_prices",
        "coefficient":-1
    },
    {
        "asset_type": "battery",
        "variable": "electric_discharge_mwh",
        "interval_data": "electricity_prices",
        "coefficient": 0.25
    }
]
site.optimize(objective={"terms": terms})
```

See [Custom Objectives](https://energypylinear.adgefficiency.com/latest/how-to/custom-objectives/) in the documentation for more examples.

### Logging Improvements

The dependency on `structlog` has been removed - we now only use `rich.logging.Console` to log to STDOUT. The ability to log to a file has been removed.

The `verbose` flag now accepts either a `bool` or an `int`. The mapping of `verbose` to log levels is as follows:

| `verbose` | Log Level |
|-----------|-----------|
| True      | INFO      |
| False     | ERROR     |
| 1         | DEBUG     |
| 2         | INFO      |
| 3         | WARNING   |
| 4         | ERROR     |

```python
import energypylinear as epl
asset = epl.Battery(electricity_prices=[10, -50, 200, -50, 200])
simulation = asset.optimize(verbose=2)
```

```
INFO     assets.site.optimize: cfg=<SiteConfig name=site, freq_mins=60,         
         import_limit_mw=10000.0, export_limit_mw=10000.0>                      
INFO     assets.site.optimize: cfg=<SiteConfig name=site, freq_mins=60,         
         import_limit_mw=10000.0, export_limit_mw=10000.0>                      
INFO     assets.site.optimize: assets=['battery', 'spill']                      
INFO     assets.site.optimize: assets=['battery', 'spill']                      
INFO     optimizer.solve: status='Optimal'                                      
INFO     optimizer.solve: status='Optimal'                                      
```

### Tighten Optimizer Tolerance

The default relative tolerance of the CBC optimizer has been reduced to `0.0`.

### Optimizer Config can be a Dictionary

It's now possible to use a dictionary in place of the `epl.OptimizerConfig` object:

```python
asset.optimize(optimizer_config={"timeout": 2, "relative_tolerance": 0.1})
```

### Other Changes

We have upgraded Poetry to 1.7.0 and Mypy to 1.7.0.

Plausible analytics added to the documentation.

## [1.1.1](https://github.com/ADGEfficiency/energy-py-linear/releases/tag/v1.1.1)

### Bug Fixes

Fixed a bug where logger was making a `./logs` directory even when `enable_file_logging` was set to false.

Fixed the flaky test of battery export prices by reducing optimizer tolerance to 0 in the test.

### Other Changes

Removed documentation `.png` images from `main`.

## [1.1.0](https://github.com/ADGEfficiency/energy-py-linear/releases/tag/v1.1.0)

### Export Electricity Prices

Assets can now accept export electricity prices - these are an optional time series that can either be a constant value or interval data:

```python
asset = epl.Battery(
    electricity_prices=[100.0, 50, 200, -100, 0, 200, 100, -100],
    export_electricity_prices=40
)
```

These export electricity prices are used to calculate the value of electricity exported from site.

### Optimizer Config

The `.optimize()` method of assets now accepts an `epl.OptimizerConfig` object, which allows configuration of the CBC optimizer used by Pulp:

```python
asset.optimize(
    optimizer_config=epl.OptimizerConfig(timeout=60, relative_tolerance=0.05)
)
```

### Bug Fixes

Fixed a bug on the `allow_infeasible` flag in `epl.Site.optimize`.

Fixed a bug on the `export_limit_mw` in `epl.Site.__init__`.

#### Netting Off Battery Charge and Discharge

`energypylinear` has the ability to constrain battery charge or discharge into a single interval, using binary variables that are linked to the charge and discharge energy.

By default these were turned off, because it slows down the optimization. The effect on the site electricity balance was zero, as the charge and discharge energy were netted off in the balance.

However, as the battery losses are a percentage of battery charge, this led to situations where when electricity prices were negative, the optimizer would be incentivized to have a large simultaneous charge and discharge.  This would also lead to the situation where the losses calculations were correct as a percentage of battery charge, but not of battery net charge.

The solution is to remove the flag that allowed toggling of these binary variables on and off - this now means that the battery model always runs with binary variables limiting only one of charge or discharge to occur in a single interval.


## [1.0.0](https://github.com/ADGEfficiency/energy-py-linear/releases/tag/v1.0.0)

### Add Renewable Generator Asset

The `epl.RenewableGenerator` asset models controllable renewable generation like solar or wind.

```python
import energypylinear as epl

asset = epl.RenewableGenerator(
    electricity_prices=[1.0, -0.5],
    electric_generation_mwh=[100, 100],
    electric_generation_lower_bound_pct=0.5,
    name="wind",
)
```

This asset can clip the lower bound of the generation to a percentage of the total available generation.

This allows the renewable generator asset to reduce its generation during periods of negative prices or carbon intensities.

### Breaking Changes

#### Interval Data Rework

v1.0.0 moves the interval data arguments to asset from `asset.optimize` to `asset.__init__`:

```python
import energypylinear as epl

#  the old way
asset = epl.Battery()
simulation = asset.optimize(electricity_prices=[10, -50, 200, -50, 200])

#  the new way
asset = epl.Battery(electricity_prices=[10, -50, 200, -50, 200])
simulation = asset.optimize()
```

The reasons for this change is that it allows different asset specific interval data to be specified when using the `epl.Site` API.

#### Other Breaking Changes

`electricity_prices` is now optional - only one of `electricity_prices` or `elelectriciy_carbon_intensities` must be specified during the initialization of either an asset or site.

For the `epl.Battery` asset, the argument `efficiency` has been renamed `efficiency_pct`.

The `epl.Generator` asset has been renamed to `epl.CHP`.

The accounting API has been reworked:

```python
account = epl.get_accounts(
    forecasts.results,
    price_results=actuals.results,
    verbose=False
)
```

The simulation results object has been changed - the results `pd.Dataframe` is now the `.results` attribute on the simulation result object:

```python
#  old way
results = asset.optimize()
results = results.simulation

#  new way
simulation = asset.optimize()
results = simulation.results
```

### Bug Fixes

Fixed a bug in the documentation for [optimizing for price and carbon](https://energypylinear.adgefficiency.com/latest/how-to/price-carbon/).

Added the heat pump asset to the `epl.Site` API.

### Documentation

Expanded the asset documentation from a single file into separate files, one per asset. Moved examples into the asset documentation.

Renamed the optimization section into `How To`.

### Other Changes

Adopted [semantic versioning](https://semver.org).

Moved changelog into [docs/changelog](https://energypylinear.adgefficiency.com/latest/changelog).

Updated versions of Pydantic, Pandas & Numpy.

## [0.2.1](https://github.com/ADGEfficiency/energy-py-linear/releases/tag/v0.2.1)

Added the `epl.HeatPump` asset.

## [0.2.0](https://github.com/ADGEfficiency/energy-py-linear/releases/tag/v0.2.0)

Added bi-directional V2G charging to the EV asset.

## [0.1.2](https://github.com/ADGEfficiency/energy-py-linear/releases/tag/v0.1.2)

Added the site API, add hosted documentation.

## 0.1.1

Allowed Python 3.11.

## 0.1.0

Added `energypylinear` to PyPi.
