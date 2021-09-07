# energy-py-linear

energy-py-linear is a library for optimizing energy systems using mixed integer linear programming.

Linear programming can guarantee convergence to the optimal solution of convex linear equations. Battery storage and combined heat and power are two examples of energy systems that can be optimized as linear programs.

## Usage

A battery is optimized against a set of prices - `info` is a list of ordered dictionaries, one per timestep:

```python
import energypylinear as epl

model = epl.Battery(power=2, capacity=4, efficiency=1.0)
prices = [10, 50, 10, 50, 10]
info = model.optimize(prices, timestep='30min')
```

pandas can be used to make the output human readable:

```python
import pandas as pd

pd.DataFrame().from_dict(info)
   Import [MW]  Export [MW]  Power [MW]  Charge [MWh]
0          2.0          0.0         2.0      0.000000
1          0.0          2.0        -2.0      0.066667
2          2.0          0.0         2.0      0.000000
3          0.0          2.0        -2.0      0.066667
4          NaN          NaN         NaN      0.000000

```
Note that the last row is all `NaN` except for the `Charge` - this is because the `Charge` indicates the battery position at the start of each interval.  The last row is included so we can see the battery level at the end of the optimization run.


It is also possible to send in forecast prices along with actual prices, and to change the initial charge.  The model optimizes for the forecasts - this allows measurement of forecast quality by comparing actual with forecast costs.

```python
#  a forecast that is the inverse of the prices we used above
forecasts = [50, 10, 50, 10, 50]

info = model.optimize(prices, forecasts=forecasts, timestep='30min')
```

## Setup

```bash
$ pip install .
```

The main dependency of this project is [PuLP](https://github.com/coin-or/pulp).  Further reading on PuLP:
- [the white paper - An Introduction to pulp for Python Programmers](https://projects.coin-or.org/PuLP/export/330/trunk/doc/KPyCon2009/PulpForPythonProgrammers.pdf)
- the blog post series [Introduction to Linear Programming with Python and PuLP](http://benalexkeen.com/linear-programming-with-python-and-pulp/) - especially [Part 6](http://benalexkeen.com/linear-programming-with-python-and-pulp-part-6/) which covers how to formulate more complex conditional constraints.
