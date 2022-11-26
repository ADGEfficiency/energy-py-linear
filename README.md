
# Assets

## Spill Generator & Spill Boiler

# Dispatch versus Accounting

# Dispatch APIs


## Asset API

The asset API is designed to support optimization of a single asset - under the hood it uses the Site API with a single asset (battery or generator) and a spill generator & boiler to avoid infeasible simulations when site limits or demand can't be met by the single asset.

```python
import energypylinear as epl
asset = epl.Battery(power_mw=2, capacity_mwh=4, efficiency=0.95)
results = asset.optimize(prices=[10, 50, 10, 50, 10], freq=30)
```


# Accounting API
