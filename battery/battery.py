from pulp import LpProblem, LpMinimize, lpSum, LpVariable, LpStatus


prices = [20, 10, 30, 30]
idx = range(0, len(prices))

power = 5
capacity = 5

b_import = LpVariable.dicts(
    'import', idx[:-1], lowBound=0, cat='Continuous'
)

b_export = LpVariable.dicts(
    'export', idx[:-1], lowBound=0, cat='Continuous'
)

battery_charge = LpVariable.dicts(
    'charge', idx, lowBound=0, cat='Continuous'
)

prob = LpProblem('cost minimization', LpMinimize)

prob += lpSum(
    [b_import[i] * prices[i] for i in idx[:-1]] + 
    [b_export[i] * -prices[i] for i in idx[:-1]]
)

prob += battery_charge[0] == 1.0

for i in idx[:-1]:
    #  energy balance across two time periods
    prob += battery_charge[i+1] == battery_charge[i] + b_import[i] - b_export[i]

    #  constrain battery charge level
    prob += battery_charge[i] <= capacity
    prob += battery_charge[i] >= 0

prob.solve()
print(LpStatus[prob.status])

def show_vars():
    for v in prob.variables():
        print('{} {}'.format(v.name, v.varValue))

show_vars()
