import pulp

from energypylinear.defaults import defaults


def price_objective(optimizer, vars, interval_data) -> pulp.LpAffineExpression:

    sites = vars["sites"]
    spills = vars["spills"]
    generators = vars["generators"]
    boilers = vars["boilers"]

    return optimizer.sum(
        [
            generator.gas_consumption_mwh * interval_data.gas_prices[i]
            for generator in generators[i]
        ]
        + [
            boiler.gas_consumption_mwh * interval_data.gas_prices[i]
            for boiler in boilers[i]
        ]
        + sites[i].import_power_mwh * interval_data.electricity_prices[i]
        - sites[i].export_power_mwh * interval_data.electricity_prices[i]
        + spills[i].electric_generation_mwh * defaults.spill_objective_penalty
        + spills[i].high_temperature_generation_mwh * defaults.spill_objective_penalty
        + spills[i].electric_load_mwh * defaults.spill_objective_penalty
        #  dumping heat has no penalty
        + spills[i].high_temperature_load_mwh
        for i in interval_data.idx
    )
