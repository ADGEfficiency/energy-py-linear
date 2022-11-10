## Asset API

The asset API is designed to support optimization of a single asset - under the hood it uses the new Site API.

```python
import energypylinear as epl
asset = epl.Battery(power_mw=2, capacity_mwh=4, efficiency=0.95)
results = asset.optimize(prices=[10, 50, 10, 50, 10], freq=30)
```
