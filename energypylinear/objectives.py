import pulp

from energypylinear.defaults import defaults


def price_objective(optimizer, vars, interval_data) -> pulp.LpAffineExpression:

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


"""


        forecast_objective = self.optimizer.sum(
            sites[i].import_power_mwh * interval_data.forecasts[i]
            - sites[i].export_power_mwh * interval_data.forecasts[i]
            for i in interval_data.idx
        )
        carbon_objective = self.optimizer.sum(
            sites[i].import_power_mwh * interval_data.carbon_intensities[i]
            - sites[i].export_power_mwh * interval_data.carbon_intensities[i]
            for i in interval_data.idx
        )

"""
