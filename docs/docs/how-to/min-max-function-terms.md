
## Minimum and Maximum Function Terms

In `energypylinear` you can use [custom objective functions]() to define a custom set of incentives and costs in the `energypylinear` linear program.

Most commonly the objective function will be made up of simple terms, which are the product of a linear variable and a coefficient:

```python
import dataclasses

@dataclasses.dataclass
class Term:
    variable: str
    asset_type: str | None = None
    interval_data: str | None = None
    asset_name: str | None = None
    coefficient: float = 1.0

# an objective function term for site import power electricity cost
Term(
    variable="import_power_mwh",
    asset_type="site"
    interval_data="electricity_prices"
)

# an objective function term for battery cycle cost
Term(
    variable="import_power_mwh",
    asset_type="site"
    interval_data="electricity_prices"
)

```
