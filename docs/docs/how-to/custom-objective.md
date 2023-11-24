## Custom Objective Functions

`energypylinear` can optimize for two different objective functions (price or carbon) built into the library. 

However you may want to optimize for a different objective function in the linear program.

**A custom objective function allows you to construct an objective function as you see fit** - allowing you to optimize a site and assets for the incentives and costs that are important to you.

Core to the custom objective function is the `epl.Term` - representing a single term in the objective function:

<!--phmdoctest-share-names-->
```python
import dataclasses

@dataclasses.dataclass
class Term:
    variable: str
    asset_type: str | None = None
    interval_data: str | None = None
    asset_name: str | None = None
    coefficient: float = 1.0
```

Each term can target either many assets by type or one asset by name. It can also include multiplication by interval data or by a coefficient.

A custom objective function is a list of `epl.Term` - the sum of these terms becomes the objective function.

<!--phmdoctest-share-names-->
```python
@dataclasses.dataclass
class CustomObjectiveFunction:
    terms: list[Term]
```

### Price and Carbon

In this example we will show how to optimize a battery for an objective optimizes for both profit and carbon at the same time.

The example below creates an objective function where we incentive a site to:

- reduce import when the electricity price or carbon intensity is high,
- increase export when the electricity price or carbon intensity is low.

Key to this is defining a carbon price, which allows us to convert our emissions into money:

<!--phmdoctest-share-names-->
```python
import numpy as np
import energypylinear as epl

def simulate(
    carbon_price: int,
    seed: int,
    n: int,
    verbose: int = 2
) -> epl.SimulationResult:
    """Runs one battery simulation at a given carbon price with a custom objective function."""
    np.random.seed(seed)
    site = epl.Site(
        assets=[epl.Battery(power_mw=10, capacity_mwh=20)],
        electricity_prices=np.random.normal(100, 1000, n),
        electricity_carbon_intensities=np.clip(
            np.random.normal(1, 10, n),
            a_min=0,
            a_max=None
        )
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
                    coefficient=carbon_price
                ),
                epl.Term(
                    asset_type="site",
                    variable="export_power_mwh",
                    interval_data="electricity_carbon_intensities",
                    coefficient=-1 * carbon_price
                ),
            ]
        ),
        verbose=verbose
    )

print(simulate(carbon_price=50, seed=42, n=72))
```

```
INFO     assets.site.optimize: cfg=<SiteConfig name=site, freq_mins=60,         
         import_limit_mw=10000.0, export_limit_mw=10000.0>                      
INFO     assets.site.optimize: assets=['battery']                               
INFO     optimizer.solve: status='Optimal'                                      
<energypylinear.SimulationResult feasible:True, rows:72, cols:28>
```

We can validate that our custom objective function is working as expected by running simulations across many carbon prices, and see the effect on the profit and emissions of our site:

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
            "emissions": f"{accounts.emissions:3.2f}"
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

### Renewables Certificates

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
    )
)
print(
    simulation.results[[
        "solar-electric_generation_mwh",
        "wind-electric_generation_mwh"
    ]]
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

### Synthetic PPA

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
site = epl.Site(
    assets=assets,
    electricity_prices=electricity_prices
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
        "asset_name": "wind",
        "variable": "electric_generation_mwh",
        "interval_data": "electricity_prices",
        "coefficient": 1
    },
    {
        "asset_name": "wind",
        "variable": "electric_generation_mwh",
        "coefficient": -70
    }
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


### Battery Cycle Cost

It's common in battery optimization to include a cost to use the battery - for every MWh of charge, some cost is incurred.

We can model this cost using a custom objective function, by applying a cost to discharging the battery:

<!--phmdoctest-share-names-->
```python
import numpy as np
import energypylinear as epl

np.random.seed(42)
electricity_prices = np.random.normal(0, 1000, 48)

assets = [
    epl.Battery(power_mw=20, capacity_mwh=20)
]
site = epl.Site(
    assets=assets,
    electricity_prices=electricity_prices
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
site.optimize(
    verbose=4,
    objective={"terms": terms}
)
```

You could also apply this cost to the battery electric charge, or to both the charge and discharge at the same time:

```python
terms=[
    {
        "asset_type": "battery",
        "variable": "electric_charge_mwh",
        "interval_data": "electricity_prices",
        "coefficient": 0.25
    },
    {
        "asset_type": "battery",
        "variable": "electric_discharge_mwh",
        "interval_data": "electricity_prices",
        "coefficient": 0.25
    }
]
```

We can validate that this works by applying a stronger cycle cost and seeing the battery use descrease:

<!--phmdoctest-share-names-->
```python
import pandas as pd

results = []
for cycle_cost in [0.25, 0.5, 1.0, 2.0]:
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
            "coefficient": cycle_cost
        }
    ]
    simulation = site.optimize(
        verbose=4,
        objective={"terms": terms}
    )
    results.append(
        {
            "cycle_cost": cycle_cost,
            "battery-electric_discharge_mwh": simulation.results["battery-electric_discharge_mwh"].sum()
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
