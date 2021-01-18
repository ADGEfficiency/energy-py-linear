from collections import OrderedDict
import json
import logging
from pulp import LpProblem, LpMinimize, lpSum, LpVariable, LpStatus
import re
from datetime import timedelta


logger = logging.getLogger(__name__)


def parse_timedelta(time_str):
    """Parses a string (e.g. 24h, 24hours, 30m) into a timedelta"""
    regex = re.compile(r'((?P<hours>\d+?)h)?((?P<minutes>\d+?)m)?((?P<seconds>\d+?)s)?')
    parts = regex.match(time_str)
    if not parts:
        return
    parts = parts.groupdict()
    time_params = {}
    for (name, param) in parts.items():
        if param:
            time_params[name] = int(param)
    return timedelta(**time_params)


class Battery(object):
    """Electric battery operating in price arbitrage.

    power       float [MW] same for charge & discharge
    capacity    float [MWh]
    efficiency  float [%] round trip, applied to
    """

    def __init__(self, power, capacity, efficiency=0.9):
        self.power = float(power)
        self.capacity = float(capacity)
        self.efficiency = float(efficiency)

        args = {
            "name": "args",
            "power": self.power,
            "capacity": self.capacity,
            "efficiency": self.efficiency,
        }

        logger.info(json.dumps(args))

    def __repr__(self):
        return 'Battery(power={}, capacity={}, eff={})'.format(
            self.power, self.capacity, self.efficiency
        )

    def setup_vars(self, idx):
        """Create a dictionary with the pulp variables."""
        return {
            'imports': LpVariable.dicts(
                'import', idx[:-1], lowBound=0,
                upBound=self.power, cat='Continuous'
            ),

            'exports': LpVariable.dicts(
                'export', idx[:-1], lowBound=0,
                upBound=self.power, cat='Continuous'
            ),

            'charges': LpVariable.dicts(
                'charge', idx, lowBound=0, cat='Continuous'
            ),

            'losses': LpVariable.dicts(
                'loss', idx[:-1], lowBound=0, cat='Continuous'
            )
        }

    def optimize(self, prices, forecasts=None, initial_charge=0, timestep='5min'):
        """Run the linear program to optimize the battery.

        prices         list [$/MWh]
        forecasts      list [$/MWh]
        initial_charge float [MWh]
        timestep       str   5min, 1hr etc
        """
        self.prob = LpProblem('cost minimization', LpMinimize)

        self.timestep = timestep
        timestep_timedelta = parse_timedelta(timestep)
        timestep_hours = timestep_timedelta.total_seconds() / (60*60)
        self.step = 1 / timestep_hours
        #  append a NaN onto the prices list to represent the price
        #  during the last reported period, which is only used to give the
        #  final charge, and not included in the optimization
        prices = list(prices)
        prices.append(None)

        if forecasts is None:
            forecasts = prices
        else:
            # If we're not inheriting the prices, we need to append to forecast
            # to match the price list.
            forecasts.append(None)

        forecast_len = len(forecasts)
        price_len = len(prices)
        len_msg = """
            The number of forecasts({}) should match the number of prices({}).
        """.format(forecast_len, price_len)
        assert forecast_len == price_len, len_msg

        assert initial_charge <= self.capacity
        assert initial_charge >= 0

        #  used to index timesteps
        idx = range(0, len(prices))

        self.vars = self.setup_vars(idx)

        imports = self.vars['imports']
        exports = self.vars['exports']
        charges = self.vars['charges']
        losses = self.vars['losses']

        #  the objective function we are minimizing
        self.prob += lpSum(
            [imports[i] * forecasts[i] for i in idx[:-1]] +
            [-(exports[i] - losses[i]) * forecasts[i] for i in idx[:-1]]
        )

        #  initial charge
        self.prob += charges[0] == initial_charge

        #  last item in the index isn't used because the last timestep only
        #  represents the final charge level - no import or export is done
        for i in idx[:-1]:
            #  energy balance across two time periods
            self.prob += charges[i + 1] == charges[i] + (imports[i] - (losses[i] + exports[i])) / self.step

            #  constrain battery charge level
            self.prob += charges[i] <= self.capacity
            self.prob += charges[i] >= 0

            self.prob += losses[i] == exports[i] * (1 - self.efficiency)

        print('starting linear program for {}'.format(self))
        self.prob.solve()

        opt_results = {
            "name": "optimization_results",
            "status": LpStatus[self.prob.status]
        }

        print('linear program for {} done - {}'.format(self, opt_results['status']))

        logger.info(json.dumps(opt_results))

        self.info = self.generate_outputs(prices, forecasts, idx,
                                          initial_charge)

        return self.info

    def calc_net(self, imp, exp, loss):
        """Calculate the Net, or None if inputs are None."""
        return imp - exp + loss

    def calc_cost(self, energy, price, step):
        """Calculate the cost, or None if energy is None."""
        return (energy * price) / step

    def calc_gross(self, imp, exp):
        """Calculate the Gross, or None if Import or Export are None."""
        return imp - exp

    def generate_outputs(self, prices, forecasts, idx_range, initial_charge):
        """Create a dictionary of results and summaries."""

        results = []
        for row_id in idx_range[:-1]:
            imp = self.vars['imports'].get(row_id).value()
            exp = self.vars['exports'].get(row_id).value()
            loss = self.vars['losses'].get(row_id).value()

            #  'charges' is the charge at the start of the interval
            initial_chg = self.vars['charges'].get(row_id).value()
            final_chg = self.vars['charges'].get(row_id + 1).value()

            price = prices[row_id]
            forecast = forecasts[row_id]

            net = self.calc_net(imp, exp, loss)
            actual_costs = self.calc_cost(net, price, self.step)
            forecast_costs = self.calc_cost(net, forecast, self.step)
            gross = self.calc_gross(imp, exp)

            result = [
                ('Import [MW]', imp),
                ('Export [MW]', exp),
                ('Gross [MW]', gross),
                ('Net [MW]', net),
                ('Losses [MW]', loss),
                ('Initial charge [MWh]', initial_chg),
                ('Final charge [MWh]', final_chg),
                ('Prices [$/MWh]', price),
                ('Forecast [$/MWh]', forecast),
                ('Actual [$/{}]'.format(self.timestep), actual_costs),
                ('Forecast [$/{}]'.format(self.timestep), forecast_costs)
            ]

            #  added ordered dict to get consistent iteration across results
            out = OrderedDict()
            for key, value in result:
                out[key] = value

            results.append(out)

        return results

if __name__ == '__main__':
    model = Battery(power=2, capacity=4)
    prices = [50, 10, 10, 50, 50, 10]
    info = model.optimize(prices, initial_charge=0, timestep='1hr')
    from pprint import pprint as pp
    pp(info)
