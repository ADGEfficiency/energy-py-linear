`energypylinear` has the ability to optimize for both price and carbon as optimization objectives built into the library.

Sometimes however, you may want to optimize for a different objective - this is where custom objective functions come in.

A custom objective function allows you to construct your objective function as you see fit - allowing you to optimize a site and assets for the incentives and costs that are important to you.

## A Custom Objective for Both Price & Carbon

In this example we will show how to optimize a battery for an objective that includes both profit maximization and carbon emissions reduction.

A custom objective function

<!--phmdoctest-share-names-->
```python
import energypylinear as epl

def simulate(carbon_price: int, seed: int, n: int, verbose: int = 2) -> epl.SimulationResult:
    np.random.seed(seed)
    site = epl.Site(
        assets=[epl.Battery(power_mw=10, capacity_mwh=20)],
        electricity_prices=np.random.normal(100, 1000, n),
        electricity_carbon_intensities=np.clip(np.random.normal(1, 10, n), a_min=0, a_max=None)
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
INFO     assets.site.optimize: cfg=<SiteConfig name=site, freq_mins=60, import_limit_mw=10000.0, export_limit_mw=10000.0>
INFO     assets.site.optimize: assets=['battery']
INFO     optimizer.solve: status='Optimal'
<energypylinear.SimulationResult feasible:True, rows:72, cols:28>
```

## Validating our Custom Price & Carbon Objective Function

We can validate that our custom objective function is working as expected by running simulations across many carbon prices, and see the effect on the profit and emissions of our site:

<!--phmdoctest-share-names-->
```python
from rich import print

results = []
for carbon_price in range(0, 300, 50):
    simulation = simulate(carbon_price=carbon_price, seed=42, n=72, verbose=3)
    results.append(
        epl.get_accounts(simulation.results)
    )
print(results)
```

```
[
    Accounts(cost=-466212.61402771604, emissions=161.15902025377193, profit=466212.61402771604),
    Accounts(cost=-462348.5208314832, emissions=-353.91002234343773, profit=462348.5208314832),
    Accounts(cost=-462348.5208314832, emissions=-353.91002234343773, profit=462348.5208314832),
    Accounts(cost=-442491.5493116912, emissions=-761.1011639171603, profit=442491.5493116912),
    Accounts(cost=-418641.72822318133, emissions=-1081.7941891834703, profit=418641.72822318133)
]
```

## Other Examples

### Renewables Certificates

### Heat Dump Cost

### Synthetic PPA

Swap the generation at wholesale for fixed price

maybe show that the reduce load parameter has different effects
