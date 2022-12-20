import pulp

import energypylinear as epl
from energypylinear.defaults import defaults


def price_objective(
    optimizer: epl.optimizer.Optimizer, vars, interval_data: epl.data.IntervalData
) -> pulp.LpAffineExpression:

    sites = vars["sites"]
    spills = vars["spills"]
    spill_evs = vars["spill-evs"]
    generators = vars.get("generators", [])
    boilers = vars.get("boilers", [])

    if len(generators) == 0:
        generators = [[epl.assets.asset.Asset()] for i in interval_data.idx]
    if len(boilers) == 0:
        boilers = [[epl.assets.asset.Asset()] for i in interval_data.idx]
    if len(spill_evs) == 0:
        spill_evs = [[epl.assets.asset.Asset()] for i in interval_data.idx]

    assert isinstance(interval_data.gas_prices, list)

    obj = [
        sites[i].import_power_mwh * interval_data.electricity_prices[i]
        - sites[i].export_power_mwh * interval_data.electricity_prices[i]
        + spills[i].electric_generation_mwh * defaults.spill_objective_penalty
        + spills[i].high_temperature_generation_mwh * defaults.spill_objective_penalty
        + spills[i].electric_load_mwh * defaults.spill_objective_penalty
        #  dumping heat has no penalty
        + spills[i].high_temperature_load_mwh
        + [
            spill_ev.charge_mwh
            * defaults.spill_objective_penalty
            * interval_data.electricity_prices[i]
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
    optimizer: epl.optimizer.Optimizer, vars, interval_data: epl.data.IntervalData
) -> pulp.LpAffineExpression:

    sites = vars["sites"]
    spills = vars["spills"]
    generators = vars.get("generators", [])
    boilers = vars.get("boilers", [])

    assert isinstance(interval_data.carbon_intensities, list)
    obj = [
        sites[i].import_power_mwh * interval_data.carbon_intensities[i]
        - sites[i].export_power_mwh * interval_data.carbon_intensities[i]
        + spills[i].electric_generation_mwh * defaults.spill_objective_penalty
        + spills[i].high_temperature_generation_mwh * defaults.spill_objective_penalty
        + spills[i].electric_load_mwh * defaults.spill_objective_penalty
        #  dumping heat has no penalty
        + spills[i].high_temperature_load_mwh
        for i in interval_data.idx
    ]
    if generators:
        obj += [
            generator.gas_consumption_mwh * defaults.gas_carbon_intensity_tc_per_mwh
            for i in interval_data.idx
            for generator in generators[i]
        ]
    if boilers:
        obj += [
            boiler.gas_consumption_mwh * defaults.gas_carbon_intensity_tc_per_mwh
            for i in interval_data.idx
            for boiler in boilers[i]
        ]
    return optimizer.sum(obj)
