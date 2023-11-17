`energypylinear` has the ability to optimize for both price and carbon as optimization objectives built into the library.

Sometimes however, you may want to optimize for a different objective - this is where custom objective functions come in.

A custom objective function allows you to construct your objective function as you see fit - allowing you to optimize a site and assets for the incentives and costs that are important to you.

Core to the custom objective function is the `epl.Term` - representing a single term in the objective function:

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

Each term can target either many assets by type or one asset by name.

## Optimizing for Both Price and Carbon

In this example we will show how to optimize a battery for an objective that includes both profit maximization and carbon emissions reduction.

A custom objective function is a list of terms - the sum of these terms becomes the objective function.

The example below creates an objective function where we incentive the site to:

- reduce import when the electricity price or carbon intensity is high,
- increase export when the electricity price or carbon intensity is low.

Key to this is defining a carbon price, which allows us to convert our emissions into money:

<!--phmdoctest-share-names-->
```python
import numpy as np
import energypylinear as epl

def simulate(carbon_price: int, seed: int, n: int, verbose: int = 2) -> epl.SimulationResult:
    """Runs one battery simulation at a given carbon price with a price and carbon objective function."""
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

### Validating our Custom Objective Function

We can validate that our custom objective function is working as expected by running simulations across many carbon prices, and see the effect on the profit and emissions of our site:

<!--phmdoctest-share-names-->
```python
from rich import print

results = []
for carbon_price in range(0, 300, 50):
    simulation = simulate(carbon_price=carbon_price, seed=42, n=72, verbose=3)
    results.append(
        (carbon_price, epl.get_accounts(simulation.results))
    )
print(results)
```

```
[
    (
        0,
        Accounts(
            cost=-466212.61402771604,
            emissions=161.15902025377193,
            profit=466212.61402771604
        )
    ),
    (
        50,
        Accounts(
            cost=-452318.6769574514,
            emissions=-579.5092576655181,
            profit=452318.6769574514
        )
    ),
    (
        100,
        Accounts(
            cost=-390152.3790973575,
            emissions=-1403.2085727250274,
            profit=390152.3790973575
        )
    ),
    (
        150,
        Accounts(
            cost=-336073.24333483365,
            emissions=-1848.9408727284797,
            profit=336073.24333483365
        )
    ),
    (
        200,
        Accounts(
            cost=-290186.2623118541,
            emissions=-2098.2766452348483,
            profit=290186.2623118541
        )
    ),
    (
        250,
        Accounts(
            cost=-248371.69782712747,
            emissions=-2288.4186586585793,
            profit=248371.69782712747
        )
    )
]
```

As expected as our carbon price increases, our emissions decrease, and our profit decreases.

## Renewables Certificates

In the previous example we used a custom objective function to apply incentives to the site import and export electricity.

A custom objective function can also be used to apply incentives to a single asset.

An example of this is a renewable energy certificate scheme, where the generation from one asset receives additional income for each MWh generated.

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

As expected, as the first generator that is called is the `solar` generator, as it receives additional income for it's output.  As the site demand increases, the `wind` generator is called to make up the remaining demand.

## Synthetic PPA

A synthetic PPA is a financial instrument that allows swapping of the output of a wholesale exposed generator to a fixed price.

This can be modelled as a custom objective function:

```python
import numpy as np
import energypylinear as epl

n = 6
np.random.seed(42)
wind_mwh = np.random.uniform(0, 100, n)
electricity_prices = np.random.normal(0, 1000, n)

assets: list[epl.Asset] = [
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

## Synthetic PPA

A synthetic PPA is a financial instrument that allows swapping of the output of a wholesale exposed generator to a fixed price.

This can be modelled as a custom objective function:

```python
import numpy as np
import energypylinear as epl

n = 6
np.random.seed(42)
wind_mwh = np.random.uniform(0, 100, n)
electricity_prices = np.random.normal(0, 1000, n)

assets: list[epl.Asset] = [
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

As expected, our renewable generator still generates during times of negative electricity prices - this is because its output is incentivized at a fixed price.
