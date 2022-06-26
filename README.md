# energy-py-linear

A library for optimizing energy systems using mixed integer linear programming.

Currently the library has two models: 
- electric battery operating in price arbitrage,
- a combined heat & power plant.


## Use

The battery model is optimized against a set of prices, and returns a list of dictionaries - one per interval:

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

It is also possible to send in forecast prices along with actual prices, and to change the initial charge.

If the model receives forecasts it will optimize for them - this allows measurement of forecast quality by comparing actual with forecast costs:

```python
#  a forecast that is the inverse of the prices we used above
forecasts = [50, 10, 50, 10, 50]
results = model.optimize(prices, forecasts=forecasts, timestep='60T', objective='forecasts')
```

The battery model also accounts for carbon.  If no carbon profile is passed in, a constant value of 0.5 tC/MWh is assumed.

We can switch the optimization to focus on carbon:

```python
results = model.optimize(prices, forecasts=forecasts, carbon=carbon, timestep="60T", objective='carbon')
```


## Setup

Install as an editable package:

```bash
$ make setup
```

The main dependency of this project is [PuLP](https://github.com/coin-or/pulp).  For further reading on PuLP:

- [An Introduction to pulp for Python Programmers](https://projects.coin-or.org/PuLP/export/330/trunk/doc/KPyCon2009/PulpForPythonProgrammers.pdf),
- the blog post series [Introduction to Linear Programming with Python and PuLP](http://benalexkeen.com/linear-programming-with-python-and-pulp/) - especially [Part 6](http://benalexkeen.com/linear-programming-with-python-and-pulp-part-6/) which covers how to formulate more complex conditional constraints.


## Test

```bash
$ make test
```
