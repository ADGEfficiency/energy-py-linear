import pulp

from pulp import LpProblem, LpMinimize, LpVariable, LpStatus

"""
all efficiencies are HHV

you cant use divide by with pulp ?

obj needs all the costs
- gas
- elect

balances need
- steam
- elect
"""


class Asset(object):
    enthalpy =  (2851.34 - 418.991) / 1000
    """
    Generic CHP asset
    """

    def __init__(self):
        pass

    def steam_generated(self):
        return 0.0

    def gas_burnt(self):
        return 0.0

    def power_generated(self):
        return 0.0


class GasTurbine(Asset):
    lb = 50
    ub = 100

    def __init__(
            self,
            size,
            name,
    ):
        super().__init__() 

        self.size = size

        self.cont = LpVariable(
            name, 0, self.ub
        )

        self.binary = LpVariable(
            '{}_bin'.format(name), lowBound=0, upBound=1, cat='Integer'
        )

        self.load = self.cont * (1/100)

        self.effy = {
            'electrical': 0.28,
            'thermal': 0.4
        }

    def steam_generated(self):
        """ t/h """
        heat_generated = self.size * self.load * (1 / self.effy['electrical']) * self.effy['thermal']
        return heat_generated * (1 / self.enthalpy) * 3.6

    def gas_burnt(self):
        """  MW HHV """
        return self.size * self.load * (1 / self.effy['electrical'])

    def power_generated(self):
        """  MW """
        return self.load * self.size


class Boiler(Asset):

    def __init__(
            self,
            size,
            name,
            min_turndown=0.0,
            parasitics=0.1,
            efficiency=0.8
    ):
        super().__init__() 
        self.lb = min_turndown
        self.ub = size

        self.cont = LpVariable(
            name, 0, size
        )

        self.binary = LpVariable(
            '{}_bin'.format(name), lowBound=0, upBound=1, cat='Integer'
        )

        self.load = self.cont

        self.effy = {
            'thermal': efficiency
        }

        self.parasitics = parasitics

    def steam_generated(self):
        """ t/h """
        return self.load

    def gas_burnt(self):
        """ MW HHV """
        #  https://www.tlv.com/global/TI/calculator/steam-table-temperature.html
        #  MJ/kg = kJ/kg * MJ/kJ

        #  MW = t/h * kg/t * hr/sec * MJ/kg / effy
        return self.load * (1/3.6) * self.enthalpy * (1/self.effy['thermal'])

    def power_generated(self):
        """ MW """
        return self.parasitics


class SteamTurbine(Asset):

    def __init__(
            self,
            name,
            size
    ):
        super().__init__() 

        #  t/h
        self.lb = 15
        self.ub = 30

        #  MW
        self.slope = (size - 0) / (30 - 0)

        self.cont = LpVariable(
            name, 0, self.ub
        )

        self.binary = LpVariable(
            '{}_bin'.format(name), lowBound=0, upBound=1, cat='Integer'
        )

    def steam_generated(self):
        return -self.cont

    def gas_burnt(self):
        return 0

    def power_generated(self):
        """  MW """
        return self.cont * self.slope


if __name__ == '__main__':
    gas_price = 20
    electricity_price = 1000

    site_steam_demand = 100
    site_power_demand = 50

    prob = LpProblem('cost minimization', LpMinimize)

    assets = [
        GasTurbine(size=10, name='gt1'),
        Boiler(size=100, name='blr1'),
        Boiler(size=100, name='blr2', efficiency=0.9),
        SteamTurbine(size=6, name='st1')
    ]

    net_grid = LpVariable('net_power_to_site', -100, 100)

    #  need to form objective function first
    prob += sum([asset.gas_burnt() for asset in assets]) * gas_price \
        + net_grid * electricity_price

    prob += sum([asset.steam_generated() for asset in assets]) == site_steam_demand, 'steam_balance'

    prob += sum([asset.power_generated() for asset in assets]) + net_grid == site_power_demand, 'power_balance'

    #  constraints on the asset min & maxes
    for asset in assets:
        prob += asset.cont - asset.ub * asset.binary <= 0
        prob += asset.lb * asset.binary - asset.cont <= 0

    prob.solve()

    print(LpStatus[prob.status])

    for v in prob.variables():
        print('{} {}'.format(v.name, v.varValue))

    steam_generated = sum(
        [max(asset.steam_generated().value(), 0) for asset in assets]
    )
    steam_consumed = sum(
        [-min(asset.steam_generated().value(), 0) for asset in assets]
    )

    print('total steam generated {:2.1f} t/h'.format(steam_generated))
    print('total steam consumed {:2.1f} t/h'.format(steam_consumed))
    print('steam to site {:2.1f} t/h'.format(site_steam_demand))

    def calc_value(value):
        try:
            return float(value)
        except TypeError:
            return float(value.value())

    power_generated = sum(
        [max(calc_value(asset.power_generated()), 0) for asset in assets]
    )
    power_consumed = sum(
        [-min(calc_value(asset.power_generated()), 0) for asset in assets]
    )
    net_grid = float(net_grid.value())

    print('total power generated {:2.1f} MWe'.format(power_generated))
    print('total power consumed {:2.1f} MWe'.format(power_consumed))
    print('net grid {:2.1f} MWe'.format(net_grid))
    print('power to site {:2.1f} MWe'.format(site_power_demand))
