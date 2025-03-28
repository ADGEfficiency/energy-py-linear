`energypylinear` can optimize for both price and carbon as optimization objectives.

This ability comes from two things - an objective function, which can be either for price or carbon, along with accounting of both price and carbon emissions.

We can dispatch a battery to minimize carbon emissions by passing in `objective='carbon'`:

## Setup Interval Data

<!--phmdoctest-share-names-->
```python
import energypylinear as epl

electricity_prices = [100, 50, 200, -100, 0, 200, 100, -100]
electricity_carbon_intensities = [0.1, 0.2, 0.1, 0.15, 0.01, 0.7, 0.5, 0.01]
```

## Optimize for Carbon

<!--phmdoctest-share-names-->
```python
asset = epl.Battery(
    power_mw=2,
    capacity_mwh=4,
    efficiency_pct=0.9,
    electricity_prices=electricity_prices,
    electricity_carbon_intensities=electricity_carbon_intensities,
)
carbon = asset.optimize(objective="carbon", verbose=3)

carbon_account = epl.get_accounts(carbon.results, verbose=3)
print(f"{carbon_account=}")
```

```
carbon_account=<Accounts profit=134.44 emissions=-2.2733>
```

## Optimize for Money

We can compare these results above with a simulation that optimizes for price, using a `energypylinear.accounting.Account` to compare both simulations.

Our optimization for price has a high negative cost.

The optimization for carbon has lower emissions, but at a higher cost:

<!--phmdoctest-share-names-->
```python
asset = epl.Battery(
    power_mw=2,
    capacity_mwh=4,
    efficiency_pct=0.9,
    electricity_prices=electricity_prices,
    electricity_carbon_intensities=electricity_carbon_intensities,
)
price = asset.optimize(
    objective="price",
    verbose=3
)

price_account = epl.get_accounts(price.results, verbose=3)
print(f"{price_account=}")
```

```
price_account=<Accounts profit=1037.78 emissions=-1.6578>
```

## Calculate Variance Between Accounts

<!--phmdoctest-share-names-->
```python
variance = price_account - carbon_account
print(f"{variance=}")
print(f"{-variance.cost / variance.emissions:.2f} $/tC")
```

```
variance=<Account profit=903.33 emissions=0.6156>
1467.51 $/tC
```

## Full Example

```python
import energypylinear as epl

electricity_prices = [100, 50, 200, -100, 0, 200, 100, -100]
electricity_carbon_intensities = [0.1, 0.2, 0.1, 0.15, 0.01, 0.7, 0.5, 0.01]
asset = epl.Battery(
    power_mw=2,
    capacity_mwh=4,
    efficiency_pct=0.9,
    electricity_prices=electricity_prices,
    electricity_carbon_intensities=electricity_carbon_intensities,
)

# optimize for carbon
carbon = asset.optimize(objective="carbon", verbose=3)
carbon_account = epl.get_accounts(carbon.results, verbose=3)
print(f"{carbon_account=}")

# optimize for money
price = asset.optimize(
    objective="price",
    verbose=3
)
price_account = epl.get_accounts(price.results, verbose=3)
print(f"{price_account=}")

# calculate variance (difference) between accounts
variance = price_account - carbon_account
print(f"{variance=}")
print(f"{-variance.cost / variance.emissions:.2f} $/tC")
```
