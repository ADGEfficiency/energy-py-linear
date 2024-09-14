`energypylinear` has the ability to optimize for a network charge.

A network charge is a tariff applied to a site based on the power consumed in certain intervals.  It's often set to incentive reductions in demand during peak periods.

## No Network Charge

### Asset, Interval Data and Objective Function

First we will setup a battery with no network charge, and optimize it for electricity prices:

<!--phmdoctest-share-names-->
```python
import energypylinear as epl

electricity_prices = [50, 100, 150]
asset = epl.Battery(electricity_prices=electricity_prices, efficiency=0.9)
bau = asset.optimize(
    {
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
    verbose=5
)
```

### Results

We can then calculate the net import and battery charge:

<!--phmdoctest-share-names-->
```python
results = bau.results
results["battery-net_charge_mwh"] = (
    results["battery-electric_charge_mwh"] - results["battery-electric_discharge_mwh"]
)
results["site-net_import_mwh"] = (
    results["site-import_power_mwh"] - results["site-export_power_mwh"]
)
print(
    bau.results[
        [
            "site-electricity_prices",
            "site-net_import_mwh",
            "battery-net_charge_mwh",
            "battery-electric_final_charge_mwh",
        ]
    ]
)
```

As expected, our battery charges during the first two intervals when electricity prices are low, and discharges during the third interval when prices are high:

```
   site-electricity_prices  site-net_import_mwh  battery-net_charge_mwh  battery-electric_final_charge_mwh
0                     50.0             2.000000                2.000000                                1.8
1                    100.0             0.222222                0.222222                                2.0
2                    150.0            -2.000000               -2.000000                                0.0
```

## With Network Charge

### Asset and Interval Data

By default, `energypylinear` uses interval data like `electricity_prices` or `electricity_carbon_intensities`.  This interval data is supplied when initializing an asset or site.

For network charges, we will make use of the ability to supply custom interval data. Any extra keyword arguments supplied to an asset or site will be attempted to be parsed as interval data.

Below we setup a battery with both electricity prices and a network charge:

<!--phmdoctest-share-names-->
```python
import energypylinear as epl

assert electricity_prices == [50, 100, 150]
network_charge = [0, 100, 0]
asset = epl.Battery(electricity_prices=electricity_prices, network_charge=network_charge)
```

### Objective Function

By default, `energypylinear` has two built-in objective functions - `price` and `carbon`.

In order to optimize for a network charge, we need to supply a custom objective function.  This function will be passed to the `optimize` method of an asset or site.

Below we optimize our battery with a custom objective function:

<!--phmdoctest-share-names-->
```python
network_charge = asset.optimize(
    {
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
            {
                "asset_type": "site",
                "variable": "import_power_mwh",
                "interval_data": "network_charge",
                "coefficient": 1,
            },
        ]
    },
    verbose=3
)
```

### Results

<!--phmdoctest-share-names-->
```python
results = network_charge.results
results["battery-net_charge_mwh"] = (
    results["battery-electric_charge_mwh"] - results["battery-electric_discharge_mwh"]
)
results["site-net_import_mwh"] = (
    results["site-import_power_mwh"] - results["site-export_power_mwh"]
)
print(
    network_charge.results[
        [
            "site-electricity_prices",
            "site-network_charge",
            "site-net_import_mwh",
            "battery-net_charge_mwh",
            "battery-electric_final_charge_mwh",
        ]
    ]
)
```

We now see that our battery has not charged during the second interval, where we have a high site network charge:

```
   site-electricity_prices  site-network_charge  site-net_import_mwh  battery-net_charge_mwh  battery-electric_final_charge_mwh
0                     50.0                  0.0                  2.0                     2.0                                1.8
1                    100.0                100.0                  0.0                     0.0                                1.8
2                    150.0                  0.0                 -1.8                    -1.8                                0.0
```
