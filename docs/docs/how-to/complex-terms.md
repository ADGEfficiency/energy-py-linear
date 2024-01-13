## Simple Terms

In `energypylinear` you can use [custom objective functions](https://energypylinear.adgefficiency.com/latest/how-to/custom-objectives/) to define a custom set of incentives and costs in your linear program.

The objective function will often be made up of simple terms, which are the product of a single linear variable (one per interval), interval data and a coefficient:

<!--phmdoctest-mark.skip-->
```python
import dataclasses

--8<-- "energypylinear/objectives.py:term"
# an objective function term for site import power electricity cost
Term(
    variable="import_power_mwh",
    asset_type="site",
    interval_data="electricity_prices"
)

# an objective function term for site export power electricity revenue
Term(
    variable="import_power_mwh",
    asset_type="site",
    interval_data="electricity_prices",
    coefficient=-1
)

# an objective function term for battery cycle cost
Term(
    variable="electric_charge_mwh",
    asset_type="battery",
    coefficient=0.25
)

```

## Complex Terms

Sites will often have more complicated costs and revenues.  `energypylinear` uses complex terms to include these more complicated terms in the objective function.  

<!--phmdoctest-mark.skip-->
```python
--8<-- "energypylinear/objectives.py:complex-terms"
```

Currently the library includes four function terms, which allow adding minimum or maximum constraints on collections of linear program variables and floats:

| Function             | Number of Linear Variables | Number of Floats | Number of Objective Function Terms |
|----------------------|----------------------------|------------------|------------------------------------|
| `min_two_variables`  | 1 or 2                     | 0 or 1           | Interval index length              |
| `max_two_variables`  | 1 or 2                     | 0 or 1           | Interval index length              |
| `max_many_variables` | Interval index length      | 0 or 1           | 1                                  |
| `min_many_variables`  | Interval index length      | 0 or 1           | 1                                  |


## Maximum Demand Charge

A common incentive for many sites is a maximum demand charge, where a site will incur a cost based on the maximum site import over a length of time (commonly a month).

We can model this using the `max_many_variables` function term, which will add a single term to the objective function that is the maximum of many linear program variables and a user supplied constant.

We can demonstrate this by using an example of a site with a variable electric load, with a peak of 50 MW.

We can first optimize the site with an objective function that does not include a demand charge:

<!--phmdoctest-share-names-->
```python
import energypylinear as epl

electric_load_mwh = [30.0, 50.0, 10.0]
electricity_prices = [0.0, 0.0, 0.0]
gas_prices = 20

site = epl.Site(
    assets=[
        epl.CHP(
            electric_efficiency_pct=1.0,
            electric_power_max_mw=50,
            electric_power_min_mw=0,
        )
    ],
    gas_prices=gas_prices,
    electricity_prices=electricity_prices,
    electric_load_mwh=electric_load_mwh,
)

no_demand_charge_simulation = site.optimize(
    verbose=3,
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
            {
                "asset_type": "*",
                "variable": "gas_consumption_mwh",
                "interval_data": "gas_prices",
            },
        ]
    },
)
```

As expected for a site with low electricity prices, this CHP does not generate in any interval:

<!--phmdoctest-share-names-->
```python
print(no_demand_charge_simulation.results['chp-electric_generation_mwh'])
```

```
0    0.0
1    0.0
2    0.0
Name: chp-electric_generation_mwh, dtype: float64
```

Let's now optimize the site with a demand charge.  This demand charge has a minimum of 40 MW, and a charge of 200:

<!--phmdoctest-share-names-->
```python
demand_charge_simulation = site.optimize(
    verbose=3,
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
            {
                "asset_type": "*",
                "variable": "gas_consumption_mwh",
                "interval_data": "gas_prices",
            },
            {
                "function": "max_many_variables",
                "variables": {
                    "asset_type": "site",
                    "variable": "import_power_mwh",
                },
                "constant": 40,
                "coefficient": 200,
                "M": max(electric_load_mwh) * 10
            },
        ]
    },
)
```

Now we see that the CHP generator has generated in the one interval that had a demand higher than our demand charge minimum of 40:

<!--phmdoctest-share-names-->
```python
print(
    demand_charge_simulation.results[
        ["site-electric_load_mwh", "chp-electric_generation_mwh"]
    ]
)
```

```
   site-electric_load_mwh  chp-electric_generation_mwh
0                    30.0                          0.0
1                    50.0                         10.0
2                    10.0                          0.0
```

If we re-run this simulation with a lower demand charge minimum, our CHP generator is now incentivized to generate in other intervals:

<!--phmdoctest-share-names-->
```python
demand_charge_simulation = site.optimize(
    verbose=3,
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
            {
                "asset_type": "*",
                "variable": "gas_consumption_mwh",
                "interval_data": "gas_prices",
            },
            {
                "function": "max_many_variables",
                "variables": {
                    "asset_type": "site",
                    "variable": "import_power_mwh",
                },
                "constant": 20,
                "coefficient": 200,
                "M": max(electric_load_mwh) * 10,
            },
        ]
    },
)

print(
    demand_charge_simulation.results[
        ["site-electric_load_mwh", "chp-electric_generation_mwh"]
    ]
)
```

```
   site-electric_load_mwh  chp-electric_generation_mwh
0                    30.0                         10.0
1                    50.0                         30.0
2                    10.0                          0.0
```

## Minimum Export Incentive

Above we looked at a function term that took the maximum across many linear program variables at once using the `max_many_variables` function term, which results in one term being added to the objective function.

Another type of function term included in `energypylinear` is the `min_two_variables` function term, which adds one term to the objective function for each interval in the linear program. 

The term will represent the minimum of either a linear program variable and another linear program variable, or a linear program variable and a user supplied constant.

To demonstrate this we can look at a site where we want to incentivize a minimum export of 10 MW or greater in each interval.

Let's first setup a site with a CHP system:

```python
import energypylinear as epl

electric_load_mwh = [30.0, 50.0, 10.0]
electricity_prices = [0.0, 0.0, 0.0]
gas_prices = 20

site = epl.Site(
    assets=[
        epl.CHP(
            electric_efficiency_pct=1.0,
            electric_power_max_mw=50,
            electric_power_min_mw=0,
        )
    ],
    gas_prices=gas_prices,
    electricity_prices=electricity_prices,
    electric_load_mwh=electric_load_mwh,
)
```

Let's optimize the site without a minimum export incentive:

```python
no_export_incentive_simulation = site.optimize(
    verbose=3,
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
            {
                "asset_type": "*",
                "variable": "gas_consumption_mwh",
                "interval_data": "gas_prices",
            },
        ]
    },
)

print(no_export_incentive_simulation.results['chp-electric_generation_mwh'])
```

As expected, our CHP system doesn't generate:

```
0    0.0
1    0.0
2    0.0
Name: chp-electric_generation_mwh, dtype: float64
```

Let's now add a minimum export incentive using the `min_two_variables` function term:

```python
no_export_incentive_simulation = site.optimize(
    verbose=3,
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
            {
                "asset_type": "*",
                "variable": "gas_consumption_mwh",
                "interval_data": "gas_prices",
            },
            {
                "function": "min_two_variables",
                "a": {
                    "asset_type": "site",
                    "variable": "export_power_mwh",
                },
                "b": 15,
                "coefficient": -200,
                "M": max(electric_load_mwh) * 10
            },
        ]
    },
)

print(
    no_export_incentive_simulation.results[
        [
            "site-electric_load_mwh",
            "site-export_power_mwh",
            "chp-electric_generation_mwh",
        ]
    ]
)
```

As expected, our CHP system generates to export a minimum of 15 MWh where possible:

```
   site-electric_load_mwh  site-export_power_mwh  chp-electric_generation_mwh
0                    30.0                   15.0                         45.0
1                    50.0                    0.0                          0.0
2                    10.0                   15.0                         25.0
```
