"""Linear programming objective cost functions for price and carbon."""
import numpy as np
import pulp

import energypylinear as epl
from energypylinear.defaults import defaults


def price_objective(
    optimizer: epl.optimizer.Optimizer,
    vars: dict,
    interval_data: "epl.interval_data.IntervalData",
) -> pulp.LpAffineExpression:
    """
    Linear programming objective for cost minimization.  Equivilant to profit maximization.

    The objective is expressed as a linear combination of the costs for site import/export of power,
    spillage, charge for spillage EVs, gas consumption by generators and boilers.

    Inputs:
        optimizer: an instance of `epl.optimizer.Optimizer` class.
        vars: a dictionary of linear programming variables in the optimization problem.
        interval_data: interaval data used in the simulation.

    Returns:
        A linear programming objective as an instance of `pulp.LpAffineExpression` class.
    """

    sites = vars["sites"]
    spills = vars["spills"]
    spill_evs = vars["spill-evs"]
    generators = vars.get("generators", [])
    boilers = vars.get("boilers", [])

    if len(generators) == 0:
        generators = [[epl.assets.asset.AssetOneInterval()] for i in interval_data.idx]
    if len(boilers) == 0:
        boilers = [[epl.assets.asset.AssetOneInterval()] for i in interval_data.idx]
    if len(spill_evs) == 0:
        spill_evs = [[epl.assets.asset.AssetOneInterval()] for i in interval_data.idx]

    assert isinstance(interval_data.gas_prices, np.ndarray)
    assert isinstance(interval_data.electricity_prices, np.ndarray)

    obj = [
        sites[i].import_power_mwh * interval_data.electricity_prices[i]
        - sites[i].export_power_mwh * interval_data.electricity_prices[i]
        + spills[i].electric_generation_mwh * defaults.spill_objective_penalty
        + spills[i].high_temperature_generation_mwh * defaults.spill_objective_penalty
        + spills[i].electric_load_mwh * defaults.spill_objective_penalty
        #  dumping heat has no penalty
        + spills[i].high_temperature_load_mwh
        + [
            spill_ev.charge_mwh * defaults.spill_objective_penalty
            for spill_ev in spill_evs[i]
        ]
        + [
            generator.gas_consumption_mwh * interval_data.gas_prices[i]
            for generator in generators[i]
        ]
        + [
            boiler.gas_consumption_mwh * interval_data.gas_prices[i]
            for boiler in boilers[i]
        ]
        for i in interval_data.idx
    ]
    return optimizer.sum(obj)


def carbon_objective(
    optimizer: epl.optimizer.Optimizer,
    vars: dict,
    interval_data: "epl.interval_data.IntervalData",
) -> pulp.LpAffineExpression:
    """
    Linear programming objective for carbon emission minimization.

    The objective is expressed as a linear combination of the costs for site import/export of power,
    spillage, charge for spillage EVs, gas consumption by generators and boilers.

    Inputs:
        optimizer: an instance of `epl.optimizer.Optimizer` class.
        vars: a dictionary of linear programming variables in the optimization problem.
        interval_data: interaval data used in the simulation.

    Returns:
        A linear programming objective as an instance of `pulp.LpAffineExpression` class.
    """

    sites = vars["sites"]
    spills = vars["spills"]
    generators = vars.get("generators", [])
    boilers = vars.get("boilers", [])

    assert isinstance(interval_data.electricity_carbon_intensities, np.ndarray)
    obj = [
        sites[i].import_power_mwh * interval_data.electricity_carbon_intensities[i]
        - sites[i].export_power_mwh * interval_data.electricity_carbon_intensities[i]
        #  could turn this off TODO
        + spills[i].electric_generation_mwh * defaults.spill_objective_penalty
        + spills[i].high_temperature_generation_mwh * defaults.spill_objective_penalty
        + spills[i].electric_load_mwh * defaults.spill_objective_penalty
        #  dumping heat has no penalty
        + spills[i].high_temperature_load_mwh
        for i in interval_data.idx
    ]
    if generators:
        obj += [
            generator.gas_consumption_mwh * defaults.gas_carbon_intensity
            for i in interval_data.idx
            for generator in generators[i]
        ]
    if boilers:
        obj += [
            boiler.gas_consumption_mwh * defaults.gas_carbon_intensity
            for i in interval_data.idx
            for boiler in boilers[i]
        ]
    return optimizer.sum(obj)


objectives = {"price": price_objective, "carbon": carbon_objective}
