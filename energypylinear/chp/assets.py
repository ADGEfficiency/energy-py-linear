"""
All efficiencies are in % HHV
"""
from pulp import LpVariable


class Asset(object):
        #  https://www.tlv.com/global/TI/calculator/steam-table-temperature.html
    enthalpy =  (2851.34 - 418.991) / 1000

    def __init__(self):
        pass

    def steam_generated(self):
        """ t/h """
        return 0.0

    def steam_consumed(self):
        """ t/h """
        return 0.0

    def gas_burnt(self):
        """ MW HHV """
        return 0.0

    def power_generated(self):
        """ MWe """
        return 0.0

    def power_consumed(self):
        """ MWe """
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
        """ MWe """
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
        """ 
        MW HHV 

        MJ/kg = kJ/kg * MJ/kJ
        MW = t/h * kg/t * hr/sec * MJ/kg / effy
        """
        return self.load * (1/3.6) * self.enthalpy * (1/self.effy['thermal'])

    def power_consumed(self):
        """ MWe """
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

    def steam_consumed(self):
        """ t/h """
        return self.cont

    def power_generated(self):
        """ MWe """
        return self.cont * self.slope
