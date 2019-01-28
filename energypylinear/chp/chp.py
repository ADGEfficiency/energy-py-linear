import pulp

from pulp import LpProblem, LpMinimize, LpVariable, LpStatus

from energypylinear.chp import Boiler, GasTurbine, SteamTurbine


if __name__ == '__main__':
    gas_price = 20
    electricity_price = 1000

    site_steam_demand = 100
    site_power_demand = 50

    def optimize_sing
    prob = LpProblem('cost minimization', LpMinimize)

    assets = [
        GasTurbine(size=10, name='gt1'),
        Boiler(size=100, name='blr1'),
        Boiler(size=100, name='blr2', efficiency=0.9),
        SteamTurbine(size=6, name='st1')
    ]

    net_grid = LpVariable('net_power_to_site', -100, 100)

    prob += sum([asset.gas_burnt() for asset in assets]) * gas_price \
        + net_grid * electricity_price

    prob += sum([asset.steam_generated() for asset in assets]) \
        - sum([asset.steam_consumed() for asset in assets]) \
        == site_steam_demand, 'steam_balance'

    prob += sum([asset.power_generated() for asset in assets]) \
        - sum([asset.power_consumed() for asset in assets]) \
        + net_grid == site_power_demand, 'power_balance'

    #  constraints on the asset min & maxes
    for asset in assets:
        prob += asset.cont - asset.ub * asset.binary <= 0
        prob += asset.lb * asset.binary - asset.cont <= 0

    prob.solve()

    print(LpStatus[prob.status])

    for v in prob.variables():
        print('{} {}'.format(v.name, v.varValue))

    def calc_value(value):
        try:
            return float(value)
        except TypeError:
            return float(value.value())

    steam_generated = sum(
        [calc_value(asset.steam_generated()) for asset in assets]
    )
    steam_consumed = sum(
        [calc_value(asset.steam_consumed()) for asset in assets]
    )

    print('total steam generated {:2.1f} t/h'.format(steam_generated))
    print('total steam consumed {:2.1f} t/h'.format(steam_consumed))
    print('steam to site {:2.1f} t/h'.format(site_steam_demand))

    power_generated = sum(
        [calc_value(asset.power_generated()) for asset in assets]
    )
    power_consumed = sum(
        [calc_value(asset.power_consumed()) for asset in assets]
    )
    net_grid = float(net_grid.value())

    print('total power generated {:2.1f} MWe'.format(power_generated))
    print('total power consumed {:2.1f} MWe'.format(power_consumed))
    print('net grid {:2.1f} MWe'.format(net_grid))
    print('power to site {:2.1f} MWe'.format(site_power_demand))
