import pulp
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus

from energypylinear.chp import Boiler, GasTurbine, SteamTurbine


def calc_value(value):
    try:
        return float(value)
    except TypeError:
        return float(value.value())


def optimize(
        assets,
        gas_price,
        electricity_price,
        site_steam_demand,
        site_power_demand
):

    prob = LpProblem('cost minimization', LpMinimize)

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
    info = generate_outputs(
        assets,
        site_steam_demand,
        site_power_demand,
        net_grid
    )

    return info

def generate_outputs(
        assets,
        site_steam_demand,
        site_power_demand,
        net_grid
):
    steam_generated = sum(
        [calc_value(asset.steam_generated()) for asset in assets]
    )
    steam_consumed = sum(
        [calc_value(asset.steam_consumed()) for asset in assets]
    )
    outputs = {}
    outputs['total steam generated {:2.1f} t/h'] = steam_generated
    outputs['total steam consumed {:2.1f} t/h'] = steam_consumed
    outputs['steam to site {:2.1f} t/h'] = site_steam_demand

    power_generated = sum(
        [calc_value(asset.power_generated()) for asset in assets]
    )
    power_consumed = sum(
        [calc_value(asset.power_consumed()) for asset in assets]
    )
    net_grid = float(net_grid.value())

    outputs['total power generated {:2.1f} MWe'] = power_generated
    outputs['total power consumed {:2.1f} MWe'] = power_consumed
    outputs['net grid {:2.1f} MWe'] = net_grid
    outputs['power to site {:2.1f} MWe'] = site_power_demand
    return outputs


if __name__ == '__main__':
    assets = [
        GasTurbine(size=10, name='gt1'),
        Boiler(size=100, name='blr1'),
        Boiler(size=100, name='blr2', efficiency=0.9),
        SteamTurbine(size=6, name='st1')
    ]

    info = optimize(
        assets,
        gas_price=20,
        electricity_price=1000,
        site_steam_demand=100,
        site_power_demand=100,
   )
