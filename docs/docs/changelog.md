# Changelog

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

### Bugs

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
