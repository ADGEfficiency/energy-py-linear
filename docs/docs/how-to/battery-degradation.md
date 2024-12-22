Battery degradation is where battery performance reduces with time or battery use.

The performance of the battery is defined by the parameters of power (MW), capacity (MWh) and efficiency (%).

`energypylinear` does not model battery degradation within a single simulation - degradation can be handled by splitting up the battery lifetime into multiple simulations.

## Modelling a Single Year in Monthly Chunks

To handle battery degradation over a year, we will split the year into 12 months and run a simulation for each month:

<!--phmdoctest-share-names-->
```python
import numpy as np
import pandas as pd

import energypylinear as epl

np.random.seed(42)
days = 35
dataset = pd.DataFrame({
    "timestamp": pd.date_range("2021-01-01", periods=days * 24, freq="h"),
    "prices": np.random.normal(-1000, 1000, days * 24) + 100
})
battery_params = {
    "power_mw": 4,
    "capacity_mwh": 10,
    "efficiency_pct": 0.9,
    "freq_mins": 60
}

results = []
objs = []
for month, group in dataset.groupby(dataset['timestamp'].dt.month):
    print(f"Month {month}")
    battery = epl.Battery(electricity_prices=group['prices'], **battery_params)
    simulation = battery.optimize(verbose=3)
    results.append(simulation.results)
    objs.append(simulation.status.objective)

year = pd.concat(results)
assert year.shape[0] == days * 24
account = epl.get_accounts(year, verbose=3)
np.testing.assert_allclose(account.profit, -1 * sum(objs))
print(account)
```

```
Month 1
Month 2
<Accounts profit=2501349.07 emissions=15.7333>
```

The results above do not include any battery degradation - battery parameters are the same at the start of each month.

## Modelling Degradation

To model degradation, we need to take a view on how our battery parameters change over time.

For our simulation, we will model:

- battery power decays by 0.1 MW for each 150 MWh of battery charge,
- battery capacity decays by 0.1 MWh for each 150 MWh of battery charge,
- battery efficiency decays by 0.1% over 30 days.

<!--phmdoctest-share-names-->
```python
def get_battery_params(cumulative_charge_mwh: float = 0, cumulative_days: float = 0) -> dict:
    """Get degraded battery parameters based on usage and time."""
    power_decay_mw_per_mwh = 0.1 / 150
    capacity_decay_mwh_per_mwh = 0.1 / 150
    efficiency_decay_pct_per_day = 0.1 / 30
    return {
        "power_mw": 4 - power_decay_mw_per_mwh * cumulative_charge_mwh,
        "capacity_mwh": 10 - capacity_decay_mwh_per_mwh * cumulative_charge_mwh,
        "efficiency_pct": 0.9 - efficiency_decay_pct_per_day * cumulative_days,
        "freq_mins": 60
    }
```

For a fresh battery, our battery parameters are:

<!--phmdoctest-share-names-->
```python
print(get_battery_params())
```

```
{'power_mw': 4.0, 'capacity_mwh': 10.0, 'efficiency_pct': 0.9, 'freq_mins': 60}
```

For a battery that has been charged with 300 MWh over 60 days, our battery parameters are:

<!--phmdoctest-share-names-->
```python
print(get_battery_params(cumulative_charge_mwh=300, cumulative_days=60))
```

```
{'power_mw': 3.8, 'capacity_mwh': 9.8, 'efficiency_pct': 0.7, 'freq_mins': 60}
```

## Modelling a Single Year in Monthly Chunks with Degradation

We can include our battery degradation model in our simulation by keeping track of our battery usage and updating the battery parameters at the start of each month:

<!--phmdoctest-share-names-->
```python
import collections

results = []
cumulative = collections.defaultdict(float)
for month, group in dataset.groupby(dataset['timestamp'].dt.month):
    battery_params = get_battery_params(
        cumulative_charge_mwh=cumulative['charge_mwh'],
        cumulative_days=cumulative['days']
    )
    print(f"Month: {month}, Battery Params: {battery_params}")
    battery = epl.Battery(electricity_prices=group['prices'], **battery_params)
    simulation = battery.optimize(verbose=3)
    results.append(simulation.results)
    cumulative['charge_mwh'] += simulation.results['battery-electric_charge_mwh'].sum()
    cumulative['days'] += group.shape[0] / 24

year = pd.concat(results)
assert year.shape[0] == days * 24
account = epl.get_accounts(year, verbose=3)
print(account)
```

```
Month: 1, Battery Params: {'power_mw': 4.0, 'capacity_mwh': 10.0, 'efficiency_pct': 0.9, 'freq_mins': 60}
Month: 2, Battery Params: {'power_mw': 3.0663703705399996, 'capacity_mwh': 9.06637037054, 'efficiency_pct': 0.7966666666666666, 'freq_mins': 60}
<Accounts profit=2460059.00 emissions=16.9273>
```

## Full Example

```python
import collections

import numpy as np
import pandas as pd

import energypylinear as epl

def get_battery_params(cumulative_charge_mwh: float = 0, cumulative_days: float = 0) -> dict:
    """Get degraded battery parameters based on usage and time."""
    power_decay_mw_per_mwh = 0.1 / 150
    capacity_decay_mwh_per_mwh = 0.1 / 150
    efficiency_decay_pct_per_day = 0.1 / 30
    return {
        "power_mw": 4 - power_decay_mw_per_mwh * cumulative_charge_mwh,
        "capacity_mwh": 10 - capacity_decay_mwh_per_mwh * cumulative_charge_mwh,
        "efficiency_pct": 0.9 - efficiency_decay_pct_per_day * cumulative_days,
        "freq_mins": 60
    }

np.random.seed(42)
days = 35
dataset = pd.DataFrame({
    "timestamp": pd.date_range("2021-01-01", periods=days * 24, freq="h"),
    "prices": np.random.normal(-1000, 1000, days * 24) + 100
})

results = []
cumulative = collections.defaultdict(float)
for month, group in dataset.groupby(dataset['timestamp'].dt.month):
    battery_params = get_battery_params(
        cumulative_charge_mwh=cumulative['charge_mwh'],
        cumulative_days=cumulative['days']
    )
    print(f"Month: {month}, Battery Params: {battery_params}")
    battery = epl.Battery(electricity_prices=group['prices'], **battery_params)
    simulation = battery.optimize(verbose=3)
    results.append(simulation.results)
    cumulative['charge_mwh'] += simulation.results['battery-electric_charge_mwh'].sum()
    cumulative['days'] += group.shape[0] / 24

year = pd.concat(results)
assert year.shape[0] == days * 24
account = epl.get_accounts(year, verbose=3)
print(account)
```

```
Month: 1, Battery Params: {'power_mw': 4.0, 'capacity_mwh': 10.0, 'efficiency_pct': 0.9, 'freq_mins': 60}
Month: 2, Battery Params: {'power_mw': 3.0663703705399996, 'capacity_mwh': 9.06637037054, 'efficiency_pct': 0.7966666666666666, 'freq_mins': 60}
<Accounts profit=2460059.00 emissions=16.9273>
```
