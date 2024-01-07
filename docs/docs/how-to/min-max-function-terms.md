In `energypylinear` you can use [custom objective functions]() to define a custom set of incentives and costs in your linear program.

The objective function will often be made up of simple terms, which are the product of a single linear variable (one per interval), interval data and a coefficient:

```python
import dataclasses

@dataclasses.dataclass
class Term:
    variable: str
    asset_type: str | None = None
    interval_data: str | None = None
    asset_name: str | None = None
    coefficient: float = 1.0

# an objective function term for site import power electricity cost
Term(
    variable="import_power_mwh",
    asset_type="site"
    interval_data="electricity_prices"
)

# an objective function term for site export power electricity revenue
Term(
    variable="import_power_mwh",
    asset_type="site"
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

In addition to these simple objective function terms, sites will often have more complicated costs and revenues.  

`energypylinear` uses function terms to include these more complicated terms in the objective function.  

Currently the library includes four function terms, which allow adding minimum or maximum constraints on collections of linear program variables and floats:

| Function             | Number of Linear Variables | Number of Floats | Number of Objective Function Terms |
|----------------------|----------------------------|------------------|------------------------------------|
| `min_two_variables`  | 1 or 2                     | 0 or 1           | Interval index length              |
| `max_two_variables`  | 1 or 2                     | 0 or 1           | Interval index length              |
| `max_many_variables` | Interval index length      | 0 or 1           | 1                                  |
| `min_many_variables`  | Interval index length      | 0 or 1           | 1                                  |

```python
--8<-- "energypylinear/objectives.py:min-max-function-terms"
```

## Examples

### Maximum Demand Charges

A common incentive for many sites is a maximum demand, where a site will incur a charge based on the maximum site import over a length of time (commonly a month).

We can model this using the `max_many_variables` function term, which will add a single term to the objective function that is the maximum of many linear program variables and a user supplied constant.

We can demonstrate this by using an example of a site with a variable electric load, with a peak of 50 MW.

We can first optimize the site with an objective function that does not include a demand charge:

```python
import energypylinear as epl

electric_load_mwh = [10.0, 50.0, 10.0]
electricity_prices = [0.0, 0.0, 0.0]
gas_prices = 20

site = epl.Site(
    assets=[
        epl.CHP(
            electric_efficiency_pct=1.0,
            electric_power_max_mw=10,
            electric_power_min_mw=10,
        )
    ],
    gas_prices=gas_prices,
    electricity_prices=electricity_prices,
    electric_load_mwh=electric_load_mwh,
)

no_demand_charge_simulation = site.optimize(
    verbose=0,
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

Let's now optimize the site with a demand charge.  This demand charge has a minimum of 40 MW, and a charge of 200:

```python
site = epl.Site(
    assets=[
        epl.CHP(
            electric_efficiency_pct=1.0,
            electric_power_max_mw=10,
            electric_power_min_mw=10,
        )
    ],
    gas_prices=gas_prices,
    electricity_prices=electricity_prices,
    electric_load_mwh=electric_load_mwh,
)

demand_charge_simulation = site.optimize(
    verbose=0,
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
                "M": max(import_power_mwh) * 10
            },
        ]
    },
)
```

Now we see that TODO - site import / generation has changed

```python
```

We can also check that the demand charge is being applied correctly by looking the objective function values for the two simulations:

```python
```



### Minimum Export Incentive

Above we looked at a function term that took the maximum across many linear program variables at once, which results in one term being added to the objective function.

Another type of function term included in `energypylinear` is the `min_two_variables` function term, which adds a term to the objective function for each interval in the linear program. 

The term will represent the minimum of either a linear program variable and another linear program variable, or a linear program variable and a user supplied constant.

To demonstrate this we can look at a site where we want to incentivize a minimum export of 10 MW or greater in each interval:
