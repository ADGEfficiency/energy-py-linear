A natural response when you get access to a new modelling framework or tool *is does this work correctly?*

The purpose of this section is to give you confidence in the implementation of the energy modelling.

## Battery Model Validation

### Price Dispatch Behaviour

Let's optimize a battery using a sequence of five prices.

We expect that the battery will charge when prices are low, and will discharge when prices are high.

In `energypylinear`, a positive site electricity balance is importing, and a negative site electricity balance is exporting.

```python
import energypylinear as epl

asset = epl.Battery()
results = asset.optimize(electricity_prices=[10, -50, 200, -50, 200])
print(results.simulation[["electricity_prices", "site-electricity_balance_mwh"]])
"""
   electricity_prices  site-electricity_balance_mwh
0                  10                      0.444444
1                 -50                      2.000000
2                 200                     -2.000000
3                 -50                      2.000000
4                 200                     -2.000000
"""
```

As expected, the battery charges (with a site that is positive) when prices are low and discharges (with a negative site electricity balance) when prices are high.

Now let's change the prices and see how the dispatch changes:

```python
import energypylinear as epl

asset = epl.Battery()
results = asset.optimize(electricity_prices=[200, -50, -50, 200, 200])
print(results.simulation[["electricity_prices", "site-electricity_balance_mwh"]])
"""
   electricity_prices  site-electricity_balance_mwh
0                 200                           0.0
1                 -50                           2.0
2                 -50                           2.0
3                 200                          -2.0
4                 200                          -1.6
"""
```

As expected, the battery continues to charge during low electricity price intervals, and discharge when electricity prices are high.

### Energy Balance

Let's return to our original set of prices and check the energy balance of the battery:

```python
import energypylinear as epl

asset = epl.Battery()
results = asset.optimize(electricity_prices=[10, -50, 200, -50, 200])

balance = epl.results.check_electricity_balance(results.simulation)
print(balance)
"""
     import  generation  export  load    charge  discharge  balance      loss  spills  initial_charge  final_charge
0  0.444444         0.0     0.0   0.0  0.444444        0.0     True  0.044444     0.0             0.0           0.4
1  2.000000         0.0     0.0   0.0  2.000000        0.0     True  0.200000     0.0             0.4           2.2
2  0.000000         0.0     2.0   0.0  0.000000        2.0     True  0.000000     0.0             2.2           0.2
3  2.000000         0.0     0.0   0.0  2.000000        0.0     True  0.200000     0.0             0.2           2.0
4  0.000000         0.0     2.0   0.0  0.000000        2.0     True  0.000000     0.0             2.0           0.0
"""
```

In the first interval, we charge the battery with `0.444444 MWh` - `0.4 MWh` goes into increasing the battery state of charge from `0.0 MWh` to `0.4 MWh`, with the balance `0.044444 MWh` going to battery losses.

### Battery Efficiency

- check that when efficiency change, dispatch changes, losses change

```python
import energypylinear as epl

asset = epl.Battery()
results = asset.optimize(electricity_prices=[10, -50, 200, -50, 200])
print(results.simulation[["electricity_prices", "site-electricity_balance_mwh"]])
"""
   electricity_prices  site-electricity_balance_mwh
0                  10                      0.444444
1                 -50                      2.000000
2                 200                     -2.000000
3                 -50                      2.000000
4                 200                     -2.000000
"""
```

### State of Charge & Power Ratings

- check capacity limits?  check power rating?
- initial and final state of charge, how charge calculated
