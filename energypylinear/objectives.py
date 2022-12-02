import pulp

import energypylinear as epl
from energypylinear.defaults import defaults


def price_objective(
    optimizer: epl.optimizer.Pulp, vars, interval_data: epl.data.IntervalData
) -> pulp.LpAffineExpression:

    sites = vars["sites"]
    spills = vars["spills"]
    generators = vars.get("generators", [])
    boilers = vars.get("boilers", [])

    obj = [
        sites[i].import_power_mwh * interval_data.electricity_prices[i]
        - sites[i].export_power_mwh * interval_data.electricity_prices[i]
        + spills[i].electric_generation_mwh * defaults.spill_objective_penalty
        + spills[i].high_temperature_generation_mwh * defaults.spill_objective_penalty
        + spills[i].electric_load_mwh * defaults.spill_objective_penalty
        #  dumping heat has no penalty
        + spills[i].high_temperature_load_mwh
        for i in interval_data.idx
    ]
    if generators:
        obj += [
            generator.gas_consumption_mwh * interval_data.gas_prices[i]
            for i in interval_data.idx
            for generator in generators[i]
        ]
    if boilers:
        obj += [
            boiler.gas_consumption_mwh * interval_data.gas_prices[i]
            for i in interval_data.idx
            for boiler in boilers[i]
        ]
    return optimizer.sum(obj)


def carbon_objective(
    optimizer: epl.optimizer.Pulp, vars, interval_data: epl.data.IntervalData
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
