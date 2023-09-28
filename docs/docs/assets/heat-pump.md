# Heat Pump

Optimize the operation of a heat pump.  A heat pump uses electricity to convert low temperature heat to high temperature heat.

When using `epl.HeatPump.optimize`, the alternative to the heat pump is generating high temperature heat from a gas boiler.  Under the hood of `epl.HeatPump.optimize`, a `epl.Boiler` asset is used to supply the balance of high temperature heat demand of the site.


The gas price is important as the alternative to using a heat pump to supply the `high_temperature_load_mwh` is using a natural gas boiler.

In order for the heat pump to work, it needs to have both a source of low temperature heat and a sink of high temperature heat.

The `high_temperature_load_mwh` is the amount of heat consumed by the site, and `low_temperature_generation_mwh` is the amount of available low temperature heat.

```python
import energypylinear as epl

asset = epl.HeatPump(
    electric_power_mw=1.0,
    cop=2,
    gas_prices=20,
    electricity_prices=[100, -100],
    high_temperature_load_mwh=3.0,
    low_temperature_generation_mwh=3.0,
)
simulation = asset.optimize(verbose=False)
print(
    simulation.results[
        [
            "site-electricity_prices",
            "heat-pump-electric_load_mwh",
            "heat-pump-low_temperature_load_mwh",
            "heat-pump-high_temperature_generation_mwh",
        ]
    ]
)

assert all(
    simulation.results.columns
    == [
        "site-import_power_mwh",
        "site-export_power_mwh",
        "site-electricity_prices",
        "site-electricity_carbon_intensities",
        "site-high_temperature_load_mwh",
        "site-low_temperature_load_mwh",
        "site-low_temperature_generation_mwh",
        "site-gas_prices",
        "spill-electric_generation_mwh",
        "spill-electric_load_mwh",
        "spill-high_temperature_generation_mwh",
        "spill-low_temperature_generation_mwh",
        "spill-high_temperature_load_mwh",
        "spill-low_temperature_load_mwh",
        "spill-gas_consumption_mwh",
        "boiler-high_temperature_generation_mwh",
        "boiler-gas_consumption_mwh",
        "valve-high_temperature_load_mwh",
        "valve-low_temperature_generation_mwh",
        "heat-pump-electric_load_mwh",
        "heat-pump-low_temperature_load_mwh",
        "heat-pump-high_temperature_generation_mwh",
        "total-electric_generation_mwh",
        "total-electric_load_mwh",
        "total-high_temperature_generation_mwh",
        "total-low_temperature_generation_mwh",
        "total-high_temperature_load_mwh",
        "total-low_temperature_load_mwh",
        "total-gas_consumption_mwh",
        "total-electric_charge_mwh",
        "total-electric_discharge_mwh",
        "total-spills_mwh",
        "total-electric_loss_mwh",
        "site-electricity_balance_mwh",
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
   site-electricity_prices  heat-pump-electric_load_mwh  heat-pump-low_temperature_load_mwh  heat-pump-high_temperature_generation_mwh
0                      100                          0.0                                 0.0                                        0.0
1                     -100                          1.0                                 1.0                                        2.0
```

Under the hood the heat pump asset also includes a `epl.Spill`, which allows dumping of excess low temperature heat, and a `epl.Valve` to allow high temperature heat to flow into low temperature heat.

The combination of a `epl.Spill`, `epl.Valve` and negative electricity prices can lead to the heat pump using electricity to generate high temperature heat which is then dumped as low temperature heat.  For this reason the `epl.HeatPump` asset includes a `include_valve: bool` option to turn off the valve.

You could also setup an `epl.Site` with other assets that generate high temperature heat to explore different tradeoffs (such as a heat pump using low temperature heat from a CHP system).
