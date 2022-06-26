"""
All efficiencies are in % HHV
"""
import pulp
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus


class Asset(object):
    #  https://www.tlv.com/global/TI/calculator/steam-table-temperature.html
    enthalpy = (2851.34 - 418.991) / 1000

    def __init__(self):
        pass

    def steam_generated(self):
        """t/h"""
        return 0.0

    def steam_consumed(self):
        """t/h"""
        return 0.0

    def gas_burnt(self):
        """MW HHV"""
        return 0.0

    def power_generated(self):
        """MWe"""
        return 0.0

    def power_consumed(self):
        """MWe"""
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

        self.cont = LpVariable(name, 0, self.ub)
        self.binary = LpVariable(
            "{}_bin".format(name), lowBound=0, upBound=1, cat="Integer"
        )

        self.load = self.cont * (1 / 100)

        self.effy = {"electrical": 0.28, "thermal": 0.4}

    def steam_generated(self):
        """t/h"""
        heat_generated = (
            self.size * self.load * (1 / self.effy["electrical"]) * self.effy["thermal"]
        )
        return heat_generated * (1 / self.enthalpy) * 3.6

    def gas_burnt(self):
        """MW HHV"""
        return self.size * self.load * (1 / self.effy["electrical"])

    def power_generated(self):
        """MWe"""
        return self.load * self.size


class Boiler(Asset):
    def __init__(self, size, name, min_turndown=0.0, parasitics=0.1, efficiency=0.8):
        super().__init__()

        self.lb = min_turndown
        self.ub = size

        self.cont = LpVariable(name, 0, size)
        self.binary = LpVariable(
            "{}_bin".format(name), lowBound=0, upBound=1, cat="Integer"
        )

        self.load = self.cont

        self.effy = {"thermal": efficiency}

        self.parasitics = parasitics

    def steam_generated(self):
        """t/h"""
        return self.load

    def gas_burnt(self):
        """
        MW HHV

        MJ/kg = kJ/kg * MJ/kJ
        MW = t/h * kg/t * hr/sec * MJ/kg / effy
        """
        return self.load * (1 / 3.6) * self.enthalpy * (1 / self.effy["thermal"])

    def power_consumed(self):
        """MWe"""
        return self.parasitics


class SteamTurbine(Asset):
    def __init__(self, name, size):
        super().__init__()

        #  t/h
        self.lb = 15
        self.ub = 30

        #  MW
        self.slope = (size - 0) / (30 - 0)

        self.cont = LpVariable(name, 0, self.ub)
        self.binary = LpVariable(
            "{}_bin".format(name), lowBound=0, upBound=1, cat="Integer"
        )

    def steam_consumed(self):
        """t/h"""
        return self.cont

    def power_generated(self):
        """MWe"""
        return self.cont * self.slope


def calc_value(value):
    try:
        return float(value)
    except TypeError:
        return float(value.value())


def optimize(
    assets, gas_price, electricity_price, site_steam_demand, site_power_demand
):

    prob = LpProblem("cost minimization", LpMinimize)

    net_grid = LpVariable("net_power_to_site", -100, 100)

    prob += (
        sum([asset.gas_burnt() for asset in assets]) * gas_price
        + net_grid * electricity_price
    )

    prob += (
        sum([asset.steam_generated() for asset in assets])
        - sum([asset.steam_consumed() for asset in assets])
        == site_steam_demand,
        "steam_balance",
    )

    prob += (
        sum([asset.power_generated() for asset in assets])
        - sum([asset.power_consumed() for asset in assets])
        + net_grid
        == site_power_demand,
        "power_balance",
    )

    #  constraints on the asset min & maxes
    for asset in assets:
        prob += asset.cont - asset.ub * asset.binary <= 0
        prob += asset.lb * asset.binary - asset.cont <= 0

    prob.solve()
    info = generate_outputs(assets, site_steam_demand, site_power_demand, net_grid)

    return info


def generate_outputs(assets, site_steam_demand, site_power_demand, net_grid):
    steam_generated = sum([calc_value(asset.steam_generated()) for asset in assets])
    steam_consumed = sum([calc_value(asset.steam_consumed()) for asset in assets])
    outputs = {}
    outputs["total steam generated {:2.1f} t/h"] = steam_generated
    outputs["total steam consumed {:2.1f} t/h"] = steam_consumed
    outputs["steam to site {:2.1f} t/h"] = site_steam_demand

    power_generated = sum([calc_value(asset.power_generated()) for asset in assets])
    power_consumed = sum([calc_value(asset.power_consumed()) for asset in assets])
    net_grid = float(net_grid.value())

    outputs["total power generated {:2.1f} MWe"] = power_generated
    outputs["total power consumed {:2.1f} MWe"] = power_consumed
    outputs["net grid {:2.1f} MWe"] = net_grid
    outputs["power to site {:2.1f} MWe"] = site_power_demand
    return outputs


def main():
    assets = [
        GasTurbine(size=10, name="gt1"),
        Boiler(size=100, name="blr1"),
        Boiler(size=100, name="blr2", efficiency=0.9),
        SteamTurbine(size=6, name="st1"),
    ]

    info = optimize(
        assets,
        gas_price=20,
        electricity_price=1000,
        site_steam_demand=100,
        site_power_demand=100,
    )


if __name__ == "__main__":
    main()
