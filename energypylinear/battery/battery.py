import logging
import json

import numpy as np
import pandas as pd
from pulp import LpProblem, LpMinimize, lpSum, LpVariable, LpStatus

from energypylinear import make_logger, read_logs

logger = make_logger()

#  factor used to convert MW to MWh
#  MWh = MW / step
#  5min=12, 30min=2, 60min=1 etc

steps = {
    '5min': 60/5,
    '30min': 60/30,
    '60min': 1,
    '1hr': 1
    }


class Battery(object):
    """
    Electric battery operating in price arbitrage

    power      float [MW] same for charge & discharge
    capacity   float [MWh]
    efficiency float [%] round trip, applied to
    step   str   5min, 1hr etc
    """

    def __init__(
            self,
            power,
            capacity,
            efficiency=0.9,
            timestep='5min'

    ):
        self.power = float(power)
        self.capacity = float(capacity)
        self.efficiency = float(efficiency)
        self.timestep = timestep
        self.step = steps[self.timestep]

        args = {
            "name": "args",
            "power": self.power,
            "capacity": self.capacity,
            "efficiency": self.efficiency,
            "timestep": self.timestep,
            "step": self.step
        }

        logger.info(json.dumps(args))

        self.prob = LpProblem('cost minimization', LpMinimize)

    def setup_vars(self, idx):
        """ creates a dictionary with the pulp variables """

        return {
            'imports': LpVariable.dicts(
                'import', idx[:-1], lowBound=0, upBound=self.power, cat='Continuous'
            ),

            'exports': LpVariable.dicts(
                'export', idx[:-1], lowBound=0, upBound=self.power, cat='Continuous'
            ),

            'charges': LpVariable.dicts(
                'charge', idx, lowBound=0, cat='Continuous'
            ),

            'losses': LpVariable.dicts(
                'loss', idx, lowBound=0, cat='Continuous'
            )
        }

    def optimize(
            self,
            prices,
            forecasts=None,
            initial_charge=0
    ):
        """
        runs the linear program to optimize the battery

        prices         list [$/MWh]
        forecasts      list [$/MWh]
        initial_charge float [MWh]
        """
        if forecasts is None:
            forecasts = prices

        assert len(forecasts) == len(prices)
        assert initial_charge <= self.capacity

        #  used to index timesteps
        idx = range(0, len(prices))

        self.vars = self.setup_vars(idx)

        imports = self.vars['imports']
        exports = self.vars['exports']
        # gross_power = self.vars['gross_power']
        charges = self.vars['charges']
        losses = self.vars['losses']

        #  the objective function we are minimizing
        self.prob += lpSum(
            [imports[i] * forecasts[i] for i in idx[:-1]] +
            [-exports[i] * forecasts[i] for i in idx[:-1]]
        )

        #  initial charge
        self.prob += charges[0] == initial_charge

        #  TODO comment
        for i in idx[:-1]:
            #  energy balance across two time periods
            self.prob += charges[i+1] == charges[i] + (imports[i] - exports[i] - losses[i]) / self.step

            #  constrain battery charge level
            self.prob += charges[i] <= self.capacity
            self.prob += charges[i] >= 0

            self.prob += losses[i] == exports[i] * (1 - self.efficiency)

        self.prob.solve()

        optimization_results = {
            "name": "optimization_results",
            "status": LpStatus[self.prob.status]
        }

        logger.info(json.dumps(optimization_results))

        self.info = self.generate_outputs(prices, forecasts, idx)

        return self.info

    def generate_outputs(
            self, prices, forecasts, idx
    ):
        """ creates a DataFrame summarizing the optimization run """

        imports = self.vars['imports']
        exports = self.vars['exports']
        charges = self.vars['charges']
        losses = self.vars['losses']

        info = pd.DataFrame().from_dict({
            'Import [MW]': [imports[i].varValue for i in idx[:-1]] + [np.nan],
            'Export [MW]': [exports[i].varValue for i in idx[:-1]] + [np.nan],
            'Losses [MW]': [losses[i].varValue for i in idx[:-1]] + [np.nan],
            'Charge [MWh]': [charges[i].varValue for i in idx[:]],
            'Prices [$/MWh]': prices,
            'Forecast [$/MWh]': forecasts
        })

        info.loc[:, 'Net [MW]'] = info.loc[:, 'Import [MW]'] - info.loc[:, 'Export [MW]'] + info.loc[:, 'Losses [MW]']
        info.loc[:, 'Gross [MW]'] = info.loc[:, 'Import [MW]'] - info.loc[:, 'Export [MW]'] 

        actual_costs = info.loc[:, 'Net [MW]'] * info.loc[:, 'Prices [$/MWh]'] / self.step
        info.loc[:, 'Actual [$/{}]'.format(self.timestep)] = actual_costs

        forecast_costs = info.loc[:, 'Net [MW]'] * info.loc[:, 'Forecast [$/MWh]'] / self.step
        info.loc[:, 'Forecast [$/{}]'.format(self.timestep)] = forecast_costs

        info = info.loc[:, [
            'Import [MW]', 'Export [MW]', 'Gross [MW]', 'Net [MW]', 'Losses [MW]', 'Charge [MWh]',
            'Prices [$/MWh]', 'Forecast [$/MWh]',
            'Actual [$/{}]'.format(self.timestep),
            'Forecast [$/{}]'.format(self.timestep)]]

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(info)

        return info

if __name__ == '__main__':

    model = Battery(power=2, capacity=4, timestep='1hr')

    prices = [50, 10, 10, 50, 50, 10]

    info = model.optimize(prices, initial_charge=1)
