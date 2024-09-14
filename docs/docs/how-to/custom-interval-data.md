Interval data is a key input to an `energypylinear` simulation.

By default, `energypylinear` accepts interval data for things like electricity prices, carbon intensities and site electricity and heat consumption:

<!--phmdoctest-mark.skip-->
```python
--8<-- "energypylinear/assets/site.py:site"
```

These arguments are passed to the `SiteIntervalData` object, which is responsible for managing interval data for a site:

```python
import energypylinear as epl

asset = epl.Battery(electricity_prices=[100, 50, 200])
print(asset.site.cfg.interval_data)
```

```
electricity_prices=array([100.,  50., 200.]) export_electricity_prices=array([100.,  50., 200.]) electricity_carbon_intensities=array([0.1, 0.1, 0.1]) gas_prices=array([20, 20, 20]) electric_load_mwh=array([0, 0, 0]) high_temperature_load_mwh=array([0, 0, 0]) low_temperature_load_mwh=array([0, 0, 0]) low_temperature_generation_mwh=array([0, 0, 0]) idx=array([0, 1, 2])
```

## Custom Interval Data

Often you will want to use different interval data for your simulation - for example modelling site network charges.

Additional keyword arguments passed into a site or asset `__init__` are attempted to be parsed into interval data. These will be parsed into site interval data, even if passed into an asset.

For example, when we pass in a `network_charge` argument, we end up with a `network_charge` attribute on our `asset.site.cfg.interval_data` object:

```python
import energypylinear as epl

electricity_prices = [100, 50, 200]
asset = epl.Battery(electricity_prices=[100, 50, 200], network_charges=[10, 20, 30])
print(asset.site.cfg.interval_data)
```

```
electricity_prices=array([100.,  50., 200.]) export_electricity_prices=array([100.,  50., 200.]) electricity_carbon_intensities=array([0.1, 0.1, 0.1]) gas_prices=array([20, 20, 20]) electric_load_mwh=array([0, 0, 0]) high_temperature_load_mwh=array([0, 0, 0]) low_temperature_load_mwh=array([0, 0, 0]) low_temperature_generation_mwh=array([0, 0, 0]) idx=array([0, 1, 2]) network_charges=array([10., 20., 30.])
```

## Custom Interval Data in Simulation Results

All custom interval data will appear in the simulation results:

```python
import energypylinear as epl

asset = epl.Battery(electricity_prices=[100, 50, 200], network_charges=[10, 20, 30])
simulation = asset.optimize(verbose=3)
print(simulation.results["site-network_charges"])
```

```
0    10.0
1    20.0
2    30.0
Name: site-network_charges, dtype: float64
```

## Custom Interval Data in Custom Objective Functions

Custom interval data can be used in a custom objective function:

```python
import energypylinear as epl

asset = epl.Battery(electricity_prices=[100, 50, 200], network_charges=[10, 20, 30])
simulation = asset.optimize(
    objective={
        "terms": [
            {
                "asset_type": "site",
                "variable": "import_power_mwh",
                "interval_data": "electricity_prices",
            },
            {
                "asset_type": "site",
                "variable": "export_power_mwh",
                "interval_data": "electricity_prices",
                "coefficient": -1,
            },
        ]
    },
    verbose=3
)
```
