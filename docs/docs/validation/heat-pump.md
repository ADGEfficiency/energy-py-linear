A natural response when you get access to something someone else built is to wonder - does this work correctly?

This section will give you confidence in the implementation of the heat pump asset.

## Heat Pump

Optimize the operation of a heat pump.  A heat pump uses electricity to convert low temperature heat to high temperature heat.

When using `epl.HeatPump.optimize`, the alternative to the heat pump is generating high temperature heat from a gas boiler.  Under the hood of `epl.HeatPump.optimize`, a `epl.Boiler` asset is used to supply the balance of high temperature heat demand of the site.


The gas price is important as the alternative to using a heat pump to supply the `high_temperature_load_mwh` is using a natural gas boiler.  

In order for the heat pump to work, it needs to have both a source of low temperature heat and a sink of high temperature heat.

The `high_temperature_load_mwh` is the amount of heat consumed by the site, and `low_temperature_generation_mwh` is the amount of available low temperature heat.  

```python
import energypylinear as epl

asset = epl.HeatPump(electric_power_mw=1.0, cop=2)
results = asset.optimize(
    gas_prices=20,
    electricity_prices=[100, -100],
    high_temperature_load_mwh=3.0,
    low_temperature_generation_mwh=3.0,
    verbose=False
)
print(results.simulation[
    [
        "electricity_prices",
        'heat-pump-electric_load_mwh',
        'heat-pump-low_temperature_load_mwh',
        'heat-pump-high_temperature_generation_mwh',
    ]
])

assert all(
    results.simulation.columns
    == [
        'site-import_power_mwh',
        'site-export_power_mwh',
        'spill-electric_generation_mwh',
        'spill-electric_load_mwh',
        'spill-high_temperature_generation_mwh',
        'spill-low_temperature_generation_mwh',
        'spill-high_temperature_load_mwh',
        'spill-low_temperature_load_mwh',
        'spill-gas_consumption_mwh',
        'boiler-high_temperature_generation_mwh',
        'boiler-gas_consumption_mwh',
        'valve-high_temperature_load_mwh',
        'valve-low_temperature_generation_mwh',
        'heat-pump-electric_load_mwh',
        'heat-pump-low_temperature_load_mwh',
        'heat-pump-high_temperature_generation_mwh',
        'total-electric_generation_mwh',
        'total-electric_load_mwh',
        'total-high_temperature_generation_mwh',
        'total-low_temperature_generation_mwh',
        'total-high_temperature_load_mwh',
        'total-low_temperature_load_mwh',
        'total-gas_consumption_mwh',
        'total-electric_charge_mwh',
        'total-electric_discharge_mwh',
        'total-spills_mwh',
        'total-electric_loss_mwh',
        'site-electricity_balance_mwh',
        'electricity_prices',
        'electricity_carbon_intensities',
        'load-high_temperature_load_mwh',
        'load-low_temperature_load_mwh',
        'load-low_temperature_generation_mwh'
    ]
)
```

```output
{
    'n_spills': 1,
    'spill_columns': 8,
    'spills': {'spill-low_temperature_load_mwh': 5.0},
    'event': 'warn_spills',
    'timestamp': '2023-09-08T00:41:33.743617Z',
    'logger':'default_logger',
    'level': 'warning'
}
   electricity_prices  heat-pump-electric_load_mwh  heat-pump-low_temperature_load_mwh  heat-pump-high_temperature_generation_mwh
0                 100                          0.0                                 0.0                                        0.0
1                -100                          1.0                                 1.0                                        2.0
```

Under the hood the heat pump asset also includes a `epl.Spill`, which allows dumping of excess low temperature heat, and a `epl.Valve` to allow high temperature heat to flow into low temperature heat.  

The combination of a `epl.Spill`, `epl.Valve` and negative electricity prices can lead to the heat pump using electricity to generate high temperature heat which is then dumped as low temperature heat.  For this reason the `epl.HeatPump` asset includes a `include_valve: bool` option to turn off the valve.

You could also setup an `epl.Site` with other assets that generate high temperature heat to explore different tradeoffs (such as a heat pump using low temperature heat from a CHP system).

## Price Dispatch Behaviour

Let's optimize the heat pump using a sequence of electricity prices:

```python
import energypylinear as epl

asset = epl.HeatPump(electric_power_mw=1.0, cop=2)
results = asset.optimize(
    gas_prices=20,
    electricity_prices=[100, -100],
    high_temperature_load_mwh=3.0,
    low_temperature_generation_mwh=4.0,
    verbose=False
)
print(results.simulation[
    ["electricity_prices", "heat-pump-electric_load_mwh", "boiler-high_temperature_generation_mwh"]
])
```

```output
{'n_spills': 1, 'spill_columns': 8, 'spills': {'spill-low_temperature_load_mwh': 5.0}, 'event': 'warn_spills', 'timestamp':'2023-08-25T00:29:23.697179Z', 'logger': 'default_logger', 'level': 'warning'}
   electricity_prices  site-electricity_balance_mwh  boiler-high_temperature_generation_mwh  spill-low_temperature_load_mwh
0                 100                           0.0                                     3.0                             3.0
1                -100                           1.0                                     1.0                             2.0
```

For the first interval, with an electricity price of `100`, we see that:

- our heat pump is not operated,
- our `3.0 MWh` of high temperature heat demand has been generated by the gas boiler,
- the `4.0 MWh` of low temperature heat generation has been dumped with the spill 

For the second interval, with an electricity price of `-100`, we see that:

- our heat pump is operating at `1.0 MWe`, which means we expect `2.0 MWh` of high temperature heat,
- only `1.0 MWh` of high temperature heat demand has been generated by the gas boiler,
- we dump less low temperature heat.

## Heat Balance Behaviour

### Without a Valve

Let's first optimize a heat pump without a high temperature to low temperature valve:

```python
import energypylinear as epl

asset = epl.HeatPump(electric_power_mw=1.0, cop=2)
results = asset.optimize(
    gas_prices=20,
    electricity_prices=[-100, -100, -100],
    high_temperature_load_mwh=[3.0, 0.5, 3.0],
    low_temperature_generation_mwh=[4.0, 4.0, 0.5],
    verbose=False,
    include_valve=False
)
print(results.simulation[
    [
        "load-high_temperature_load_mwh",
        "load-low_temperature_generation_mwh",
        "spill-low_temperature_load_mwh",
        "heat-pump-electric_load_mwh",
        "heat-pump-high_temperature_generation_mwh",
        "boiler-high_temperature_generation_mwh"
    ]
])
```
```output
{'n_spills': 1, 'spill_columns': 8, 'spills': {'spill-low_temperature_load_mwh': 6.75}, 'event': 'warn_spills', 'timestamp': '2023-08-29T00:53:46.158557Z', 'logger': 'default_logger', 'level': 'warning'}
   load-high_temperature_load_mwh  load-low_temperature_generation_mwh  spill-low_temperature_load_mwh  heat-pump-electric_load_mwh  heat-pump-high_temperature_generation_mwh  boiler-high_temperature_generation_mwh
0                             3.0                                  4.0                            3.00                         1.00                                        2.0                                     1.0
1                             0.5                                  4.0                            3.75                         0.25                                        0.5                                     0.0
2                             3.0                                  0.5                            0.00                         0.50                                        1.0                                     2.0
```

Both intervals have negative electricity prices, which means we expect the heat pump to work in both intervals.

In the first interval we are unconstrained in terms of heat - our heat pump runs at the full `1.0 MWe` load.

In the second interval we have a limited amount of high temperature heat load in the site, which constrains the heat pupm.

In the third interval we have a limited amount of low temperature heat generation in the site, which 

### With a Valve

Let's the optimize a heat pump with a high temperature to low temperature valve.  This allows heat to flow frow high to low temperature, which means our boiler can generate high temperature heat that ends up as low temperature heat input into the heat pump.  This is economic because the electricity price is negative.

```python
import energypylinear as epl

asset = epl.HeatPump(electric_power_mw=1.0, cop=2)
results = asset.optimize(
    gas_prices=20,
    electricity_prices=[-100, -100, -100],
    high_temperature_load_mwh=[3.0, 0.5, 3.0],
    low_temperature_generation_mwh=[4.0, 4.0, 0.0],
    verbose=False,
    include_valve=True
)
print(results.simulation[
    [
        "load-high_temperature_load_mwh",
        "load-low_temperature_generation_mwh",
        "spill-low_temperature_load_mwh",
        "heat-pump-electric_load_mwh",
        "heat-pump-high_temperature_generation_mwh",
        "boiler-high_temperature_generation_mwh"
    ]
])
```

```output
{'n_spills': 1, 'spill_columns': 8, 'spills': {'spill-low_temperature_load_mwh': 7.5}, 'event': 'warn_spills', 'timestamp': '2023-08-29T00:52:28.353336Z', 'logger': 'default_logger', 'level': 'warning'}
   load-high_temperature_load_mwh  load-low_temperature_generation_mwh  spill-low_temperature_load_mwh  heat-pump-electric_load_mwh  heat-pump-high_temperature_generation_mwh  boiler-high_temperature_generation_mwh
0                             3.0                                  4.0                             3.0                          1.0                                        2.0                                     1.0
1                             0.5                                  4.0                             4.5                          1.0                                        2.0                                     0.0
2                             3.0                                  0.0                             0.0                          1.0                                        2.0                                     2.0
```