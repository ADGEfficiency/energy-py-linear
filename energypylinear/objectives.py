"""Linear programming objective cost functions for price and carbon."""
import dataclasses
import typing

import numpy as np
import pulp

import energypylinear as epl
from energypylinear.defaults import defaults


@dataclasses.dataclass
class Term:
    """A single term in the objective function.

    objective = Term + Term + Term"""

    variable: str
    asset_type: str | None = None
    interval_data: str | None = None
    asset_name: str | None = None
    coefficient: float = 1.0


@dataclasses.dataclass
class CustomObjectiveFunction:
    """The objective function - a sum of terms.

    CustomObjectiveFunction = Term + Term + Term"""

    terms: list[Term] = dataclasses.field(default_factory=list)


def price_objective(
    optimizer: "epl.Optimizer",
    ivars: "epl.IntervalVars",
    interval_data: "epl.assets.site.SiteIntervalData",
) -> pulp.LpAffineExpression:
    """
    Linear programming objective for cost minimization.  Equivalent to profit maximization.

    The objective is expressed as a linear combination of the costs for site import/export of power,
    spillage, charge for spillage EVs, gas consumption by generators and boilers.

    Inputs:
        optimizer: an instance of `epl.Optimizer` class.
        vars: a dictionary of linear programming variables in the optimization problem.
        interval_data: interaval data used in the simulation.

    Returns:
        A linear programming objective as an instance of `pulp.LpAffineExpression` class.
    """
    #  TODO cheating here with the site name
    sites = typing.cast(
        list[list["epl.assets.site.SiteOneInterval"]],
        ivars.filter_objective_variables_all_intervals(
            epl.assets.site.SiteOneInterval, asset_name="site"
        ),
    )
    spills = typing.cast(
        list[list["epl.assets.spill.SpillOneInterval"]],
        ivars.filter_objective_variables_all_intervals(
            epl.assets.spill.SpillOneInterval
        ),
    )
    spill_evs = typing.cast(
        list[list["epl.assets.evs.EVSpillOneInterval"]],
        ivars.filter_objective_variables_all_intervals(
            epl.assets.evs.EVSpillOneInterval
        ),
    )
    generators = typing.cast(
        list[list["epl.assets.chp.CHPOneInterval"]],
        ivars.filter_objective_variables_all_intervals(epl.assets.chp.CHPOneInterval),
    )
    boilers = typing.cast(
        list[list["epl.assets.boiler.BoilerOneInterval"]],
        ivars.filter_objective_variables_all_intervals(
            epl.assets.boiler.BoilerOneInterval
        ),
    )

    assert isinstance(interval_data.gas_prices, np.ndarray)
    assert isinstance(interval_data.electricity_prices, np.ndarray)
    assert isinstance(interval_data.export_electricity_prices, np.ndarray)

    obj: list[typing.Any | float] = [
        [
            site.import_power_mwh * interval_data.electricity_prices[i]
            - site.export_power_mwh * interval_data.export_electricity_prices[i]
            for site in sites[i]
        ]
        + [
            spill.electric_generation_mwh * defaults.spill_objective_penalty
            + spill.high_temperature_generation_mwh * defaults.spill_objective_penalty
            + spill.electric_load_mwh * defaults.spill_objective_penalty
            + spill.high_temperature_load_mwh * defaults.spill_objective_penalty
            for spill in spills[i]
        ]
        + [
            spill_ev.electric_charge_mwh * defaults.spill_objective_penalty
            + spill_ev.electric_discharge_mwh * defaults.spill_objective_penalty
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
    optimizer: "epl.Optimizer",
    ivars: "epl.interval_data.IntervalVars",
    interval_data: "epl.assets.site.SiteIntervalData",
) -> pulp.LpAffineExpression:
    """
    Linear programming objective for carbon emission minimization.

    The objective is expressed as a linear combination of the costs for site import/export of power,
    spillage, charge for spillage EVs, gas consumption by generators and boilers.

    Inputs:
        optimizer: an instance of `epl.Optimizer` class.
        vars: a dictionary of linear programming variables in the optimization problem.
        interval_data: interaval data used in the simulation.

    Returns:
        A linear programming objective as an instance of `pulp.LpAffineExpression` class.
    """
    #  TODO cheating here with the site name
    sites = typing.cast(
        list[list["epl.assets.site.SiteOneInterval"]],
        ivars.filter_objective_variables_all_intervals(
            epl.assets.site.SiteOneInterval, asset_name="site"
        ),
    )
    spills = typing.cast(
        list[list["epl.assets.spill.SpillOneInterval"]],
        ivars.filter_objective_variables_all_intervals(
            epl.assets.spill.SpillOneInterval
        ),
    )
    spill_evs = typing.cast(
        list[list["epl.assets.evs.EVSpillOneInterval"]],
        ivars.filter_objective_variables_all_intervals(
            epl.assets.evs.EVSpillOneInterval
        ),
    )
    generators = typing.cast(
        list[list["epl.assets.chp.CHPOneInterval"]],
        ivars.filter_objective_variables_all_intervals(epl.assets.chp.CHPOneInterval),
    )
    boilers = typing.cast(
        list[list["epl.assets.boiler.BoilerOneInterval"]],
        ivars.filter_objective_variables_all_intervals(
            epl.assets.boiler.BoilerOneInterval
        ),
    )

    assert isinstance(interval_data.electricity_carbon_intensities, np.ndarray)
    obj: list[typing.Any | float] = [
        [
            site.import_power_mwh * interval_data.electricity_carbon_intensities[i]
            - site.export_power_mwh * interval_data.electricity_carbon_intensities[i]
            for site in sites[i]
        ]
        + [
            spill.electric_generation_mwh * defaults.spill_objective_penalty
            + spill.high_temperature_generation_mwh * defaults.spill_objective_penalty
            + spill.electric_load_mwh * defaults.spill_objective_penalty
            + spill.electric_charge_mwh * defaults.spill_objective_penalty
            + spill.electric_discharge_mwh * defaults.spill_objective_penalty
            #  dumping heat has no penalty
            #  so high_temperature_load_mwh and low_temperature_load_mwh
            #  are not included here
            for spill in spills[i]
        ]
        + [
            spill.electric_generation_mwh * defaults.spill_objective_penalty
            + spill.high_temperature_generation_mwh * defaults.spill_objective_penalty
            + spill.electric_load_mwh * defaults.spill_objective_penalty
            + spill.electric_charge_mwh * defaults.spill_objective_penalty
            + spill.electric_discharge_mwh * defaults.spill_objective_penalty
            for spill in spill_evs[i]
        ]
        + [
            generator.gas_consumption_mwh * defaults.gas_carbon_intensity
            for generator in generators[i]
        ]
        + [
            boiler.gas_consumption_mwh * defaults.gas_carbon_intensity
            for boiler in boilers[i]
        ]
        for i in interval_data.idx
    ]
    return optimizer.sum(obj)


def get_objective(
    objective: str | dict | CustomObjectiveFunction,
    optimizer: "epl.Optimizer",
    ivars: "epl.interval_data.IntervalVars",
    interval_data: "epl.assets.site.SiteIntervalData",
) -> pulp.LpAffineExpression:
    """Creates the objective function - either from a hardcoded function or from a custom objective function."""
    hardcoded_objectives = {"price": price_objective, "carbon": carbon_objective}

    if isinstance(objective, str):
        if objective in hardcoded_objectives:
            return hardcoded_objectives[objective](optimizer, ivars, interval_data)
        else:
            raise ValueError(
                f"objective {objective} not in objectives, available objectives: {hardcoded_objectives.keys()}"
            )

    elif isinstance(objective, dict):
        objective = CustomObjectiveFunction(
            terms=[Term(**t) for t in objective["terms"]]
        )

    else:
        assert isinstance(objective, CustomObjectiveFunction)

    obj: list[typing.Any | float] = []
    for i in interval_data.idx:
        for term in objective.terms:
            if term.asset_type == "*":
                assets = ivars[i]
            else:
                assets = ivars.filter_objective_variables(
                    instance_type=term.asset_type, i=i, asset_name=term.asset_name
                )

            for asset in assets:
                obj.extend(
                    [
                        getattr(asset, term.variable)
                        * (
                            getattr(interval_data, term.interval_data)[i]
                            if term.interval_data is not None
                            else 1
                        )
                        * term.coefficient,
                    ]
                )
    return optimizer.sum(obj)
