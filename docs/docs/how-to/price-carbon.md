`energypylinear` has the ability to optimize for both price and carbon as optimization objectives.

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
carbon = asset.optimize(objective="carbon", verbose=False)

carbon_account = epl.get_accounts(carbon.results, verbose=False)
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
    verbose=False
)

price_account = epl.get_accounts(price.results, verbose=False)
print(f"{price_account=}")
```

```
price_account=<Accounts profit=1057.78 emissions=-1.6558>
```

## Calculate Variance Between Accounts

<!--phmdoctest-share-names-->
```python
variance = price_account - carbon_account
print(f"{variance=}")
print(f"{-variance.cost / variance.emissions:.2f} $/tC")
```

```
variance=<Account profit=923.33 emissions=0.6176>
1495.14 $/tC
```
