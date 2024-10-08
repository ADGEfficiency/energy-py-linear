Objective functions define the incentives and costs in a linear program. They are how you control what the linear program maximizes or minimizes in your simulation.

`energypylinear` has two different objective functions (price or carbon) built into the library.  These are available as argument to `.optimize()` on an asset or site:

```python
import energypylinear as epl

battery = epl.Battery(electricity_prices=[100, 50, 200])
simulation = battery.optimize(
    objective="carbon",
    verbose=3
)
```

In addition, `energypylinear` allows you to define your own objective functions.

**A custom objective function allows you to optimize for the revenues and costs that are important to you**.

## Custom Objective Function

A custom objective function is a list of terms:

<!--phmdoctest-mark.skip-->
```python
--8<-- "energypylinear/objectives.py:objective"
```

The objective function used in the linear program is the sum of these terms. They can be supplied as either a `epl.Term` and `epl.CustomObjectiveFunction` object or as a dictionaries.

If supplied as dictionary, the term will be coerced to an `epl.Term`.


## Simple Terms

Core to the custom objective function is the `epl.Term`, which represents a single term in the objective function:

<!--phmdoctest-mark.skip-->
```python
import dataclasses

--8<-- "energypylinear/objectives.py:term"
```

A term can target either many assets by type or one asset by name. It can also include multiplication by interval data or by a coefficient.

### Examples

#### Simultaneous Price and Carbon Optimization

`energypylinear` has two different objective functions (price or carbon) built into the library - these optimize for either price or carbon, but not both at the same time.

This example shows how to optimize a battery for an objective that will optimize for both profit and emissions at the same time.

Below we create an objective function where we:

- reduce import when the electricity price or carbon intensity is high,
- increase export when the electricity price or carbon intensity is low.

Key to this is defining a carbon price, which allows us to convert our emissions into money:

<!--phmdoctest-share-names-->
```python
import numpy as np
import energypylinear as epl

def simulate(
    carbon_price: int, seed: int, n: int, verbose: int = 3
) -> epl.SimulationResult:
    """Run a battery simulation with a custom objective function."""
    np.random.seed(seed)
    site = epl.Site(
        assets=[epl.Battery(power_mw=10, capacity_mwh=20)],
        electricity_prices=np.random.normal(100, 1000, n),
        electricity_carbon_intensities=np.clip(
            np.random.normal(1, 10, n), a_min=0, a_max=None
        ),
    )
    return site.optimize(
        objective=epl.CustomObjectiveFunction(
            terms=[
                epl.Term(
                    asset_type="site",
                    variable="import_power_mwh",
                    interval_data="electricity_prices",
                ),
                epl.Term(
                    asset_type="site",
                    variable="export_power_mwh",
                    interval_data="electricity_prices",
                    coefficient=-1,
                ),
                epl.Term(
                    asset_type="site",
                    variable="import_power_mwh",
                    interval_data="electricity_carbon_intensities",
                    coefficient=carbon_price,
                ),
                epl.Term(
                    asset_type="site",
                    variable="export_power_mwh",
                    interval_data="electricity_carbon_intensities",
                    coefficient=-1 * carbon_price,
                ),
            ]
        ),
        verbose=verbose,
    )

print(simulate(carbon_price=50, seed=42, n=72))
```

```
<energypylinear.SimulationResult feasible:True, rows:72, cols:29>
```

We can validate that our custom objective function is working as expected by running simulations across many carbon prices:

<!--phmdoctest-share-names-->
```python
import pandas as pd
from rich import print

results = []
for carbon_price in range(0, 300, 50):
    simulation = simulate(carbon_price=carbon_price, seed=42, n=72, verbose=3)
    accounts = epl.get_accounts(simulation.results)
    results.append(
        {
            "carbon_price": carbon_price,
            "profit": f"{accounts.profit:5.2f}",
            "emissions": f"{accounts.emissions:3.2f}",
        }
    )
print(pd.DataFrame(results))
```

```
   carbon_price     profit emissions
0             0  466212.61    161.16
1            50  452318.68   -579.51
2           100  390152.38  -1403.21
3           150  336073.24  -1848.94
4           200  290186.26  -2098.28
5           250  248371.70  -2288.42
```

As expected as our carbon price increases, both our profit and emissions decrease.

#### Renewables Certificates

In the previous example we used a custom objective function to apply incentives to the site import and export electricity by its asset type.

**A custom objective function can also be used to apply incentives to a single asset by name**.

An example of this is a renewable energy certificate scheme, where the generation from one asset receives additional income for each MWh generated.

In the example below, our `solar` asset receives additional income for each MWh generated.

The site has a constrained export limit, which limits how much both generators can output. The site electric load increases in each interval, which allows us to see which generator is called first:

```python
import energypylinear as epl

assets = [
    epl.RenewableGenerator(
        electric_generation_mwh=50,
        name="wind",
        electric_generation_lower_bound_pct=0.0,
    ),
    epl.RenewableGenerator(
        electric_generation_mwh=50,
        name="solar",
        electric_generation_lower_bound_pct=0.0,
    ),
]
site = epl.Site(
    assets=assets,
    electricity_prices=[250, 250, 250, 250, 250],
    export_limit_mw=25,
    electric_load_mwh=[0, 50, 75, 100, 300],
)
simulation = site.optimize(
    verbose=3,
    objective=epl.CustomObjectiveFunction(
        terms=[
            epl.Term(
                asset_type="site",
                variable="import_power_mwh",
                interval_data="electricity_prices",
            ),
            epl.Term(
                asset_type="site",
                variable="export_power_mwh",
                interval_data="electricity_prices",
                coefficient=-1,
            ),
            epl.Term(
                asset_name="solar",
                variable="electric_generation_mwh",
                coefficient=-25,
            ),
        ]
    ),
)
print(
    simulation.results[
        ["solar-electric_generation_mwh", "wind-electric_generation_mwh"]
    ]
)
```

```
   solar-electric_generation_mwh  wind-electric_generation_mwh
0                           25.0                           0.0
1                           50.0                          25.0
2                           50.0                          50.0
3                           50.0                          50.0
4                           50.0                          50.0
```

As expected, the first generator that is called is the `solar` generator, as it receives additional income for it's output.

As the site demand increases, the `wind` generator is called to make up the remaining demand.

#### Synthetic PPA

A synthetic PPA is a financial instrument that allows swapping of the output of a wholesale exposed generator to a fixed price.

This can be modelled as a custom objective function.

In the example below, we model a site with wholesale exposed import and export, and swap the output of our `wind` generator from the wholesale to a fixed price:

```python
import numpy as np
import energypylinear as epl

np.random.seed(42)
n = 6
wind_mwh = np.random.uniform(0, 100, n)
electricity_prices = np.random.normal(0, 1000, n)

assets = [
    epl.RenewableGenerator(
        electric_generation_mwh=wind_mwh,
        name="wind",
        electric_generation_lower_bound_pct=0.0,
    ),
    epl.Battery(power_mw=20, capacity_mwh=20),
]
site = epl.Site(assets=assets, electricity_prices=electricity_prices)
terms = [
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
        "asset_name": "wind",
        "variable": "electric_generation_mwh",
        "interval_data": "electricity_prices",
        "coefficient": 1,
    },
    {
        "asset_name": "wind",
        "variable": "electric_generation_mwh",
        "coefficient": -70
    },
]
simulation = site.optimize(
    verbose=4,
    objective={"terms": terms},
)
print(simulation.results[["site-electricity_prices", "wind-electric_generation_mwh"]])
```

```
   site-electricity_prices  wind-electric_generation_mwh
0              1579.212816                     37.454012
1               767.434729                     95.071431
2              -469.474386                     73.199394
3               542.560044                     59.865848
4              -463.417693                     15.601864
5              -465.729754                     15.599452
```

As expected, our renewable generator still generates even during times of negative electricity prices - this is because its output is incentivized at a fixed, positive price.

#### Battery Cycle Cost

It's common in battery optimization to include a cost to use the battery - for every MWh of charge, some cost is incurred.

We can model this cost using a custom objective function, by applying a cost to discharging the battery:

<!--phmdoctest-share-names-->
```python
import numpy as np
import energypylinear as epl

np.random.seed(42)
electricity_prices = np.random.normal(0, 1000, 48)

assets = [epl.Battery(power_mw=20, capacity_mwh=20)]
site = epl.Site(assets=assets, electricity_prices=electricity_prices)
terms = [
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
        "asset_type": "battery",
        "variable": "electric_discharge_mwh",
        "coefficient": 0.25
    }
]
site.optimize(verbose=4, objective={"terms": terms})
```

You could also apply this cost to the battery electric charge, or to both the charge and discharge at the same time:

```python
terms = [
    {
        "asset_type": "battery",
        "variable": "electric_charge_mwh",
        "coefficient": 0.25
    },
    {
        "asset_type": "battery",
        "variable": "electric_discharge_mwh",
        "coefficient": 0.25
    }
]
```

We can validate that this works by applying a stronger cycle cost and seeing the battery use decrease:

<!--phmdoctest-share-names-->
```python
import pandas as pd

results = []
for cycle_cost in [0.25, 0.5, 1.0, 2.0]:
    terms = [
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
            "asset_type": "battery",
            "variable": "electric_discharge_mwh",
            "interval_data": "electricity_prices",
            "coefficient": cycle_cost,
        },
    ]
    simulation = site.optimize(verbose=4, objective={"terms": terms})
    results.append(
        {
            "cycle_cost": cycle_cost,
            "battery-electric_discharge_mwh": simulation.results[
                "battery-electric_discharge_mwh"
            ].sum(),
        }
    )
print(pd.DataFrame(results))
```

```
   cycle_cost  battery-electric_discharge_mwh
0        0.25                           306.0
1        0.50                           322.0
2        1.00                           338.0
3        2.00                           264.0
```

As expected, as our cycle cost increases, our battery usage decreases.

## Complex Terms

In `energypylinear` you can use [custom objective functions](https://energypylinear.adgefficiency.com/latest/how-to/custom-objectives/) to define a custom set of incentives and costs in your linear program.

The objective function will often be made up of simple terms, which are the product of a single linear variable (one per interval), interval data and a coefficient. 

Sites will however often have more complicated costs and revenues, that involve taking the minimum or maximum of a collection of variables.

**A complex custom objective term allows you to construct an objective function with a complex set of costs and revenues.**

`energypylinear` uses complex terms to include these more complicated incentives and costs in the objective function:

<!--phmdoctest-mark.skip-->
```python
--8<-- "energypylinear/objectives.py:complex-terms"
```

Currently the library includes four complex terms, which allow adding minimum or maximum constraints on collections of linear program variables and floats:

| Function             | Number of Linear Variables | Number of Floats | Terms Added to Objective Function                |
|----------------------|----------------------------|------------------|-----------------------|
| `min_two_variables`  | 1 or 2                     | 0 or 1           | Interval index length |
| `max_two_variables`  | 1 or 2                     | 0 or 1           | Interval index length |
| `max_many_variables` | Interval index length      | 0 or 1           | 1                     |
| `min_many_variables` | Interval index length      | 0 or 1           | 1                     |

### Examples

#### Maximum Demand Charge

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

As expected for a site with low electricity prices, this CHP does not generate electricity in any interval:

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

Let's now optimize the site with a demand charge.  

This demand charge has a minimum of 40 MW, and a rate of 200 $/MWh:

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

#### Minimum Export Incentive

Above we looked at a function term that took the maximum across many linear program variables at once using the `max_many_variables` function term, which results in one term being added to the objective function.

Another type of function term included in `energypylinear` is the `min_two_variables` function term, which adds one term to the objective function for each interval in the linear program. 

The term will represent the minimum of either a linear program variable and another linear program variable, or a linear program variable and a user supplied constant.

To demonstrate this we can look at a site where we want to incentivize a minimum export of 15 MWh or greater in each interval.  The site will receive the maximum benefit when exporting 15 MW or more, and less benefit when exporting less than 15 MWh. There is no incentive to export more than 15 MWh.

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

Our asset does not generate in the second interval because the site demand is too high to allow the asset to export any electricity.
