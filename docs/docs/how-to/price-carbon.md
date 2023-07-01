`energypylinear` has the ability to optimize for both price and carbon as optimization objectives.

This ability comes from two things - an objective function, which can be either for price or carbon, along with accounting of both price and carbon emissions.

We can dispatch a battery to minimize carbon emissions by passing in `objective='carbon'`:

```python
import energypylinear as epl

asset = epl.battery.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)

results = asset.optimize(
  electricity_prices=[100, 50, 200, -100, 0, 200, 100, -100],
  electricity_carbon_intensities = [0.1, 0.2, 0.1, 0.15, 0.01, 0.7, 0.5, 0.01],
  objective='carbon'
)
```

We can compare these results above with a simulation that optimizes for price, using a `energypylinear.accounting.Account` to compare both simulations.  

Our optimization for price has a high negative cost.  

The optimization for carbon has lower emissions, but at a higher cost:

```python
import energypylinear as epl

#  interval data
electricity_prices = [100, 50, 200, -100, 0, 200, 100, -100]
electricity_carbon_intensities = [0.1, 0.2, 0.1, 0.15, 0.01, 0.7, 0.5, 0.01]

#  battery asset
asset = epl.battery.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)

#  optimize for money
price = asset.optimize(electricity_prices=electricity_prices)

#  optimize for the planet
carbon = asset.optimize(
  electricity_prices=electricity_prices,
  electricity_carbon_intensities=electricity_carbon_intensities,
  objective='carbon'
)

#  get an account representing the difference between the two
price = epl.get_accounts(
  price.interval_data,
  price.simulation,
)
carbon = epl.get_accounts(
  carbon.interval_data,
  carbon.simulation,
)

print(price)
# cost=-1057.777778 emissions=0.08222222199999996 profit=1057.777778

print(carbon)
# cost=-134.44444399999998 emissions=-2.2733333339999997 profit=134.44444399999998

variance = price - carbon
print(variance)
# cost=-923.3333339999999 emissions=2.3555555559999997

print(-variance.cost / variance.emissions)
# 391.9811322845319
```

The accounting API is in it's first iteration - expect it to change in the future.
