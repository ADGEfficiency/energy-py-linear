# Changelog

## [1.0.0](https://github.com/ADGEfficiency/energy-py-linear/releases/tag/v1.0.0)

### Add Renewable Generator Asset

The `epl.RenewableGenerator` asset models controllable generation, renewable generation like solar or wind.

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

Fixed a bug in the documentation for [optimizing for price and carbon](how-to/price-carbon/).

Added the heat pump asset to the `epl.Site` API.

### Documentation

Expanded the asset documentation from a single file into separate files, one per asset. Moved examples into the asset documentation.

Renamed the optimization section into `How To`.

### Other Changes

Adopted [semantic versioning](https://semver.org).

Moved changelog into [docs/changelog](https://energypylinear.adgefficiency.com/latest/changelog).

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
