## Context 

In linear programming, constraints define the feasible region of the linear program.  They are how you control what is and isn't possible in a simulation.

The assets and site in `energypylinear` apply constraints to the linear program, ranging from electricity balances around a battery to constraining how much electricity can be generated from a renewable generator.

In addition, `energypylinear` allows you to define your own, custom constraints.

**A custom constraint allows you to construct constraints that can control what can and cannot happen based on what is important to you**.

## Custom Constraint

<!--phmdoctest-mark.skip-->
```python
--8 < --"energypylinear/constraints.py:constraint"
```

A custom constraint has:

- a left hand side,
- a sense (either `le` for less than or equal to `<=`, `eq` for equal to `==`, or `ge` for greater than or equal to `>=`),
- a right hand side.

```pseudocode
# a constraint with a sense of le
LHS <= RHS

# a constraint with a sense of eq
LHS == RHS

# a constraint with a sense of ge
LHS >= RHS
```

Both the left and right hand sides are list of constraint terms. A constraint term can be either a constant, an `epl.ConstraintTerm` or a dictionary.

If supplied as dictionary, the constraint term will be coerced to an `epl.ConstraintTerm`.

## Constraint Terms

Core to the custom objective function is the `epl.ConstraintTerm`, which represents a single term in a constraint:

<!--phmdoctest-mark.skip-->
```python
--8 < --"energypylinear/constraints.py:constraint-term"
```

## Examples

### Limiting Battery Cycles

The example below shows how to optimize a battery with a constraint on battery cycles.

We define battery cycles as the sum of the total battery charge and discharge, and constraint it to be less than or equal to 15 cycles of 2 MWh per cycle:

```python
import energypylinear as epl
import numpy as np

np.random.seed(42)

cycle_limit = 2 * 15
asset = epl.Battery(
    power_mw=1,
    capacity_mwh=2,
    efficiency_pct=0.98,
    electricity_prices=np.random.normal(0.0, 1000, 48 * 7),
    constraints=[
        epl.Constraint(
            lhs=[
                epl.ConstraintTerm(
                    asset_type="battery", variable="electric_charge_mwh"
                ),
                epl.ConstraintTerm(
                    asset_type="battery", variable="electric_discharge_mwh"
                ),
            ],
            rhs=cycle_limit,
            sense="le",
            interval_aggregation="sum",
        )
    ],
)
simulation = asset.optimize(verbose=3)
total_cycles = simulation.results.sum()[
    ["battery-electric_charge_mwh", "battery-electric_discharge_mwh"]
].sum()
print(total_cycles)
```

After simulation we can see our total cycles are constrained to an upper limit of 30 (with a small floating point error):

```
30.000000002
```

### Constraining Total Generation

The example below shows how to constrain the total generation in a site.

We define a site with a solar and electric generator asset, with the available solar power increasing with time:

```python
import energypylinear as epl
import numpy as np

np.random.seed(42)

idx_len = 4
generator_size = 100
solar_gen = [10.0, 20, 30, 40]
site = epl.Site(
    assets=[
        epl.RenewableGenerator(
            electric_generation_mwh=solar_gen,
            name="solar",
            electric_generation_lower_bound_pct=0.0,
        ),
        epl.CHP(electric_power_max_mw=generator_size, electric_efficiency_pct=0.5),
    ],
    electricity_prices=np.full(idx_len, 400),
    gas_prices=10,
    constraints=[
        {
            "lhs": {"variable": "electric_generation_mwh", "asset_type": "*"},
            "rhs": 25,
            "sense": "le",
        }
    ],
)
simulation = site.optimize(verbose=3)
print(
    simulation.results[["chp-electric_generation_mwh", "solar-electric_generation_mwh", "total-electric_generation_mwh"]]
)
```

After simulation

```
   chp-electric_generation_mwh  solar-electric_generation_mwh  total-electric_generation_mwh
0                         15.0                           10.0                           25.0
1                          5.0                           20.0                           25.0
2                          0.0                           25.0                           25.0
3                          0.0                           25.0                           25.0
```
