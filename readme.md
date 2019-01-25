## energy-py-linear

energy-py-linear is a library for optimizing energy systems using mixed integer linear programming.  Linear programming is a technique that guarantees finding the global optimum for a linear system.  Battery storage and combined heat and power are two examples of energy systems that can be optimized as linear programs.

## Usage

```python
>>> import energypylinear as epl

>>> model = epl.Battery(power=2, capacity=4, timestep='30min')

>>> prices = [10, 50, 10, 50, 10]

>>> info = model.optimize(prices)

   Import [MW]  Export [MW]  Power [MW]  Charge [MWh]
0          2.0          0.0         2.0      0.000000
1          0.0          2.0        -2.0      0.066667
2          2.0          0.0         2.0      0.000000
3          0.0          2.0        -2.0      0.066667
4          NaN          NaN         NaN      0.000000
```

It is also possible to send in forecast prices along with actual prices, and to change the initial charge.  The model optimizes for the forecasts - this allows measurement of forecast quality by comparing actual with forecast costs.

```python
#  a forecast that is the inverse of the prices we used above
>>> forecasts = [50, 10, 50, 10, 50]

>>> info = model.optimize(prices, forecasts=forecasts, initial_charge=4)
```

## Installation

```bash
$ git clone https://github.com/ADGEfficiency/energy-py-linear

$ python setup.py install
```

The main dependency of this project is [PuLP](https://github.com/coin-or/pulp).  For further reading on PuLP I recommend [Introduction to Linear Programming with Python and PuLP](http://benalexkeen.com/linear-programming-with-python-and-pulp/) - especially [Part 6](http://benalexkeen.com/linear-programming-with-python-and-pulp-part-6/) which covers how to formulate more complex conditional constraints.
