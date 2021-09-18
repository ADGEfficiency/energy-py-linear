from collections import OrderedDict
import json
import logging
from pulp import LpProblem, LpMinimize, lpSum, LpVariable, LpStatus
import re
from datetime import timedelta


logger = logging.getLogger(__name__)


# def get_step(self, timestep):
#     self.timestep = timestep
#     timestep_timedelta = parse_timedelta(timestep)
#     timestep_hours = timestep_timedelta.total_seconds() / (60 * 60)
#     step = 1 / timestep_hours
#     return step


def timestep_to_freq(timestep):
    mins = int(60 / timestep)
    return f"{mins}T"


def freq_to_timestep(freq):
    assert freq in set(("60T", "30T", "15T", "5T"))
    mins = float(freq.replace("T", ""))
    return 60 / mins


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

    def optimize(
        self,
        prices=None,
        forecasts=None,
        carbon=0.5,
        initial_charge=0,
        objective="price",
        freq="5T",
    ):
        """Run the linear program to optimize the battery.

        prices         list             [$/MWh]
        forecasts      list             [$/MWh]
        carbon         list or float    [tC/MWh] carbon intensity
        initial_charge float            [MWh]
        objective      str              one of {prices, forecast, carbon}
        freq           str              one of {5T, 15T, 30T, 60T}
        """
        prices, forecasts = self.process_actuals_forecasts(prices, forecasts)

        #  we can pass in a constant carbon intensity
        if isinstance(carbon, float):
            carbon = [carbon for _ in prices]

        carbon, _ = self.process_actuals_forecasts(carbon, None)

        objectives = {
            "price": prices,
            "forecast": forecasts,
            "carbon": carbon,
        }

        objective_prices = objectives[objective]
        idx, variables = self.run_linear_program(objective_prices, initial_charge, freq)

        return self.generate_outputs(
            prices, forecasts, carbon, idx, initial_charge, variables, freq
        )

    def __repr__(self):
        return "Battery(power={}, capacity={}, eff={})".format(
            self.power, self.capacity, self.efficiency
        )

    def setup_vars(self, idx):
        """Create a dictionary with the pulp variables."""
        return {
            "imports": LpVariable.dicts(
                "import", idx[:-1], lowBound=0, upBound=self.power, cat="Continuous"
            ),
            "exports": LpVariable.dicts(
                "export", idx[:-1], lowBound=0, upBound=self.power, cat="Continuous"
            ),
            "charges": LpVariable.dicts("charge", idx, lowBound=0, cat="Continuous"),
            "losses": LpVariable.dicts("loss", idx[:-1], lowBound=0, cat="Continuous"),
        }

    def process_actuals_forecasts(self, prices, forecasts):
        """
        append a NaN onto the prices list to represent the price
        during the last reported period, which is only used to give the
        final charge, and not included in the optimization
        """
        prices = list(prices)
        prices.append(None)

        if forecasts is None:
            forecasts = prices
        else:
            # not inheriting prices, we need to append to forecast to match the price list
            forecasts.append(None)

        forecast_len = len(forecasts)
        price_len = len(prices)
        len_msg = """
            The number of forecasts({}) should match the number of prices({}).
        """.format(
            forecast_len, price_len
        )
        assert forecast_len == price_len, len_msg

        return prices, forecasts

    def run_linear_program(self, forecasts, initial_charge, freq):
        timestep = freq_to_timestep(freq)
        #  used to index timesteps
        assert initial_charge <= self.capacity
        assert initial_charge >= 0
        idx = range(0, len(forecasts))
        self.prob = LpProblem("cost minimization", LpMinimize)
        variables = self.setup_vars(idx)

        imports = variables["imports"]
        exports = variables["exports"]
        charges = variables["charges"]
        losses = variables["losses"]

        #  the objective function we are minimizing
        self.prob += lpSum(
            [imports[i] * forecasts[i] for i in idx[:-1]]
            + [-(exports[i] - losses[i]) * forecasts[i] for i in idx[:-1]]
        )

        #  initial charge
        self.prob += charges[0] == initial_charge

        #  last item in the index isn't used because the last timestep only
        #  represents the final charge level - no import or export is done
        for i in idx[:-1]:
            #  energy balance across two time periods
            self.prob += (
                charges[i + 1] == charges[i] + (imports[i] - exports[i]) / timestep
            )

            #  constrain battery charge level
            self.prob += charges[i] <= self.capacity
            self.prob += charges[i] >= 0

            self.prob += losses[i] == exports[i] * (1 - self.efficiency)

        print("starting linear program for {}".format(self))
        self.prob.solve()

        opt_results = {
            "name": "optimization_results",
            "status": LpStatus[self.prob.status],
        }

        print("linear program for {} done - {}".format(self, opt_results["status"]))

        logger.info(json.dumps(opt_results))
        return idx, variables

    def calc_net(self, imp, exp, loss):
        """Calculate the Net, or None if inputs are None."""
        return imp - exp + loss

    def calc_cost(self, energy, price, step):
        """Calculate the cost, or None if energy is None."""
        return (energy * price) / step

    def calc_gross(self, imp, exp):
        """Calculate the Gross, or None if Import or Export are None."""
        return imp - exp

    def generate_outputs(
        self, prices, forecasts, carbon, idx_range, initial_charge, variables, freq
    ):
        """Create a dictionary of results and summaries."""
        timestep = freq_to_timestep(freq)

        results = []
        for row_id in idx_range[:-1]:
            imp = variables["imports"].get(row_id).value()
            exp = variables["exports"].get(row_id).value()
            loss = variables["losses"].get(row_id).value()

            #  'charges' is the charge at the start of the interval
            initial_chg = variables["charges"].get(row_id).value()
            final_chg = variables["charges"].get(row_id + 1).value()

            price = prices[row_id]
            forecast = forecasts[row_id]

            net = self.calc_net(imp, exp, loss)
            actual_costs = self.calc_cost(net, price, timestep)
            forecast_costs = self.calc_cost(net, forecast, timestep)
            gross = self.calc_gross(imp, exp)

            carb = carbon[row_id]
            carbon_costs = self.calc_cost(net, carb, timestep)

            result = [
                ("Import [MW]", imp),
                ("Export [MW]", exp),
                ("Gross [MW]", gross),
                ("Net [MW]", net),
                ("Losses [MW]", loss),
                ("Initial Charge [MWh]", initial_chg),
                ("Final Charge [MWh]", final_chg),
                ("Price [$/MWh]", price),
                ("Forecast Price [$/MWh]", forecast),
                ("Carbon Intensity [tC/MWh]", carb),
                (f"Actual Cost [$/{freq}]", actual_costs),
                (f"Forecast Cost [$/{freq}]", forecast_costs),
                (f"Carbon Cost [tC/{freq}]", carbon_costs),
            ]

            #  ordered dict to get consistent iteration across results
            out = OrderedDict()
            for key, value in result:
                out[key] = value

            results.append(out)

        return results
