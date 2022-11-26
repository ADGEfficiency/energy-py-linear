import pulp


def price_objective(optimizer, vars, interval_data) -> pulp.LpAffineExpression:

    sites = vars["sites"]
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
        for i in interval_data.idx
    )
