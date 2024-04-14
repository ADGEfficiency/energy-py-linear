In linear programming, constraints define the feasible region of the program.  They are how you control what is and isn't possible in a program.

The assets and site in `energypylinear` apply a range of constraints to the linear program, ranging from electricity balances around a battery to constraining how much electricity can be generated from a renewable generator.

In addition, `energypylinear` allows you to define your own, custom constraints.

**A custom constraint allows you to construct constraints that can control what can and cannot happen based on what is important to you**.

## Custom Constraint

```
--8<-- "energypylinear/constraints.py:constraint"
```

A custom constraint has:

- a left hand side,
- a sense (either `le` for less than or equal to `<=`, `eq` for equal to `==`, or `ge` for greater than or equal to `>=`),
- a right hand side.

```
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

```python
--8<-- "energypylinear/constraints.py:constraint-term"
```

## Examples

### Limiting Battery Cycles

```python
import energypylinear as epl

asset = epl.Battery(
    power_mw=1,
    capacity_mwh=2,
    efficiency_pct=0.98,
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
            rhs=2 * 15,
            sense="le"
        )
    ],
)
```


### Constraining Renewable Generation

Constrain renewable generator to a percentage of total generation.

Use a CHP with cheap gas

```python

```
