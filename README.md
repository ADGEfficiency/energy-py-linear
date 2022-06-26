# energy-py-linear

A Python library for optimizing energy systems using mixed integer linear programming.  This library has two programs - an electric battery operating in price arbitrage and a combined heat & power plant.

The battery can be dispatched with perfect foresight of electricity prices or with a forecast, and will keep track of both financial and carbon benefits.  The battery can also be dispatched against carbon intensities to maximize carbon benefit rather than economic benefit.


## Use

The battery model can optimized against a set of prices. 

The optimization returns a list of dictionaries - one per time interval:

```python
import energypylinear as epl
model = epl.Battery(power=2, capacity=4, efficiency=1.0)
results = model.optimize(prices=[10, 50, 10, 50, 10], freq="60T")
```

`pandas` can be used to transform the results into a dataframe:

```python
import pandas as pd

pd.DataFrame().from_dict(results)

   Import [MW]  Export [MW]  Power [MW]  Charge [MWh]
0          2.0          0.0         2.0      0.000000
1          0.0          2.0        -2.0      0.066667
2          2.0          0.0         2.0      0.000000
3          0.0          2.0        -2.0      0.066667
4          NaN          NaN         NaN      0.000000
```

The last row is all `NaN` except for `Charge` - `Charge` indicates the battery position at the start of each interval.  The last row is included so we can see the battery level at the end of the optimization run.

*An improvement here would be to use two separate columns for charge (one for initial charge at the start of the interval and one at the end) - will get around to it one day!*

It is also possible to change the initial charge or to dispatch for forecast prices alongside with actual prices.  This allows measurement of forecast quality by comparing actual with forecast costs.

```python
#  a forecast that is the inverse of the prices we used above
forecasts = [50, 10, 50, 10, 50]
results = model.optimize(prices, forecasts=forecasts, timestep="60T", objective="forecasts", initial_charge=0.5)
```

We can also switch the optimization to focus on carbon:

```python
carbon = [0.5, 0.5, 0.9, 0.1, 0.5]
results = model.optimize(prices, carbon=carbon, timestep="60T", objective="carbon")
```


## Setup

Install as a Python package:

```bash
$ make setup
```

The main dependency of this project is [PuLP](https://github.com/coin-or/pulp) for a linear programming framework.

Further reading on PuLP:

- [An Introduction to pulp for Python Programmers](https://projects.coin-or.org/PuLP/export/330/trunk/doc/KPyCon2009/PulpForPythonProgrammers.pdf),
- the blog post series [Introduction to Linear Programming with Python and PuLP](http://benalexkeen.com/linear-programming-with-python-and-pulp/) - especially [Part 6](http://benalexkeen.com/linear-programming-with-python-and-pulp-part-6/) which covers how to formulate more complex conditional constraints.


## Test

```bash
$ make test
```
