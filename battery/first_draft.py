import numpy as np
import pandas as pd
from pulp import LpProblem, LpMinimize, lpSum, LpVariable, LpStatus

prices = [20, 10, 30, 30]
forecasts = [20, 30, 10, 30]

prices = [np.random.randint(30) for _ in range(20)]
forecasts = prices

idx = range(0, len(prices))

power = 2.5
capacity = 5

#  12 = 5min, 2 = 30 min, 1 = 1 hr
step = 1

imports = LpVariable.dicts(
    'imports', idx[:-1], lowBound=0, upBound=power, cat='Continuous'
)

exports = LpVariable.dicts(
    'exports', idx[:-1], lowBound=0, upBound=power, cat='Continuous'
)

charges = LpVariable.dicts(
    'charge', idx, lowBound=0, cat='Continuous'
)

prob = LpProblem('cost minimization', LpMinimize)

prob += lpSum(
    [imports[i] * prices[i] for i in idx[:-1]] +
    [exports[i] * -prices[i] for i in idx[:-1]]
)

#  initial charge
prob += charges[0] == 1

for i in idx[:-1]:
    #  energy balance across two time periods
    prob += charges[i+1] == charges[i] + (imports[i] - exports[i]) / step

    #  constrain battery charge level
    prob += charges[i] <= capacity
    prob += charges[i] >= 0

prob.solve()
print(LpStatus[prob.status])

def show_vars():
    for v in prob.variables():
        print('{} {}'.format(v.name, v.varValue))

out = pd.DataFrame().from_dict({
    'Import [MW]': [imports[i].varValue for i in idx[:-1]] + [np.nan],
    'Export [MW]': [exports[i].varValue for i in idx[:-1]] + [np.nan],
    'Charge [MWh]': [charges[i].varValue for i in idx[:]],
    'Prices [$/MWh]': prices,
    'Forecast [$/MWh]': forecasts
})

out.loc[:, 'Power [MW]'] = out.loc[:, 'Import [MW]'] - out.loc[:, 'Export [MW]']

actual_costs = out.loc[:, 'Import [MW]'] * out.loc[:, 'Prices [$/MWh]'] + \
    -out.loc[:, 'Export [MW]'] * out.loc[:, 'Prices [$/MWh]']

out.loc[:, 'Actual costs [$]'] = actual_costs

forecast_costs = out.loc[:, 'Import [MW]'] * out.loc[:, 'Forecast [$/MWh]'] + \
    -out.loc[:, 'Export [MW]'] * out.loc[:, 'Forecast [$/MWh]']
out.loc[:, 'Forecast costs [$]'] = actual_costs


print(actual_costs.sum(), forecast_costs.sum())

def autoflex():
    dispatch = []

    for i in idx[:-1]:
        if forecasts[i+1] > forecasts[i]:
            d = power
        else:
            d = -power
        dispatch.append(d)
    return dispatch

af = autoflex()
af += [np.nan]

out.loc[:, 'Autoflex [MW]'] = af
out.loc[:, 'Difference [MW]'] = out.loc[:, 'Autoflex [MW]'] - out.loc[:, 'Power [MW]'] 

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(out.loc[:, ['Forecast [$/MWh]', 'Power [MW]', 'Autoflex [MW]',
                      'Difference [MW]']])
