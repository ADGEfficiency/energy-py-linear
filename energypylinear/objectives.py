"""Linear programming objective cost functions for price and carbon."""
import dataclasses
import typing

import numpy as np
import pulp

import energypylinear as epl
from energypylinear.defaults import defaults
from energypylinear.logger import logger, set_logging_level


# --8<-- [start:term]
@dataclasses.dataclass
class Term:
    """A simple term in the objective function.

    This will add `i` terms to the objective function, where `i` is the length of the interval index (the number of intervals in the simulation).
    """

    variable: str
    asset_type: str | None = None
    interval_data: str | None = None
    asset_name: str | None = None
    coefficient: float = 1.0
    type: typing.Literal["simple"] = "simple"


# --8<-- [end:term]


# --8<-- [start:complex-terms]
@dataclasses.dataclass
class FunctionTermTwoVariables:
    """A function term for constraining two variables.

    This will add `i` terms to the objective function, where `i` is the length of the interval index (the number of intervals in the simulation).
    """

    function: typing.Literal["max_two_variables", "min_two_variables"]
    a: Term | float
    b: Term | float
    M: float
    interval_data: str | None = None
    coefficient: float = 1.0
    type: typing.Literal["complex"] = "complex"


@dataclasses.dataclass
class FunctionTermManyVariables:
    """A function term for constraining many variables.

    This will add 1 term to the objective function.
    """

    function: typing.Literal["max_many_variables", "min_many_variables"]
    variables: Term
    M: float
    constant: float = 0.0
    coefficient: float = 1.0
    type: typing.Literal["complex"] = "complex"


# --8<-- [end:complex-terms]


# --8<-- [start:objective]
OneTerm = Term | FunctionTermTwoVariables | FunctionTermManyVariables


@dataclasses.dataclass
class CustomObjectiveFunction:
    """A custom objective function - a sum of terms."""

    terms: list[OneTerm]


# --8<-- [end:objective]


def term_or_float(t_or_f: dict | float) -> Term | float:
    """Parse a dict or float into either a Term or float.

    Args:
        t_or_f: A dictionary representing a Term object, or a float.

    Returns:
        A Term object if input is a dictionary, or a float if input is a float.
    """
    if isinstance(t_or_f, dict):
        return Term(**t_or_f)
    assert isinstance(float(t_or_f), float)
    return float(t_or_f)


def term_factory(term: dict) -> OneTerm:
    """Resolve a dictionary into an objective function term.

    Handles deciding whether the term is simple or complex, and which type of
    complex term.

    Args:
        term: an objective function term as a dictionary.

    Returns:
        an objective function term as a Pydantic class.
    """
    if "function" not in term:
        return Term(**term)

    if "two_variables" in term["function"]:
        term["a"] = term_or_float(term["a"])
        term["b"] = term_or_float(term["b"])
        term["M"] = float(term["M"])
        return FunctionTermTwoVariables(**term)

    assert "many_variables" in term["function"]
    term["variables"] = Term(**term["variables"])
    term["M"] = float(term["M"])
    return FunctionTermManyVariables(**term)


def append_simple_objective_function_terms(
    obj: list,
    assets: list,
    i: int,
    interval_data: "epl.assets.site.SiteIntervalData",
    term: Term,
) -> None:
    """Add objective function terms to the objective function.

    Args:
        obj: the objective function.
        assets: a list of OneInterval asset objects.
        i: the interval index.
        interval_data: simulation interval data.
        term: an objective function term.
    """
    for asset in assets:
        obj.extend(
            [
                # TODO - could raise an error here if the getattr is none
                # this can happen
                getattr(asset, term.variable)
                * (
                    getattr(interval_data, term.interval_data)[i]
                    if term.interval_data is not None
                    else 1
                )
                * term.coefficient,
            ]
        )


def add_simple_terms(
    interval_data: "epl.assets.site.SiteIntervalData",
    objective: "CustomObjectiveFunction",
    ivars: "epl.interval_data.IntervalVars",
    obj: list,
) -> None:
    """Add simple objective function terms to the objective function.

    Args:
        interval_data: Simulation interval data.
        objective: The objective function as a Pydantic object.
        ivars: Linear program variables.
        obj: The objective function as a list.
    """
    for term in objective.terms:
        if term.type == "simple":
            for i in interval_data.idx:
                if term.asset_type == "*":
                    assets = ivars[i]
                else:
                    assets = ivars.filter_objective_variables(
                        instance_type=term.asset_type,
                        i=i,
                        asset_name=term.asset_name,
                    )
                append_simple_objective_function_terms(
                    obj, assets, i, interval_data, term=term
                )


def add_two_variable_terms(
    optimizer: "epl.Optimizer",
    interval_data: "epl.assets.site.SiteIntervalData",
    objective: "CustomObjectiveFunction",
    ivars: "epl.interval_data.IntervalVars",
    obj: list,
) -> None:
    """Add two variable function terms to the objective function.

    Args:
        optimizer: Linear program optimizer.
        interval_data: Simulation interval data.
        objective: The objective function as a Pydantic object.
        ivars: Linear program variables.
        obj: The objective function as a list.
    """
    function_factory = {
        "max_two_variables": optimizer.max_two_variables,
        "min_two_variables": optimizer.min_two_variables,
    }
    for term in objective.terms:
        if term.type == "complex" and term.function in function_factory:
            assert isinstance(term, FunctionTermTwoVariables)
            for i in interval_data.idx:
                """
                Currently have a limitation on only applying this kind of constraint to one asset per term
                - a can have one asset (or be a float)
                - b can have one asset (or be a float)

                In the future we could apply this to have
                - a have many assets, b be a float
                - a have many assets, b be one asset
                - a have many assets, b have many assets

                It's a bit complex, as you would need to
                - broadcast sometimes,
                - other times sum

                Perhaps could have an `aggregation` option as part of the a or b - defaults to broadcast

                TODO - leaving as debt for now
                """

                for t in [term.a, term.b]:
                    if isinstance(t, Term):
                        assert t.asset_type is None or t.asset_type == "site"
                        assert t.asset_type != "*"

                a: pulp.LpVariable | float = (
                    getattr(
                        ivars.filter_objective_variables(
                            asset_name=term.a.asset_name, i=i
                        )[0],
                        term.a.variable,
                    )
                    if isinstance(term.a, Term)
                    else float(term.a)
                )

                b: pulp.LpVariable | float = (
                    getattr(
                        ivars.filter_objective_variables(
                            asset_name=term.b.asset_name, i=i
                        )[0],
                        term.b.variable,
                    )
                    if isinstance(term.b, Term)
                    else float(term.b)
                )

                c = function_factory[term.function](
                    f"{term.function}-{i}", a=a, b=b, M=term.M
                )
                obj.append(
                    c
                    * (
                        getattr(interval_data, term.interval_data)[i]
                        if term.interval_data is not None
                        else 1
                    )
                    * term.coefficient
                )


def add_many_variables_terms(
    optimizer: "epl.Optimizer",
    objective: "CustomObjectiveFunction",
    ivars: "epl.interval_data.IntervalVars",
    obj: list,
) -> None:
    """Add many variable function terms to the objective function.

    Args:
        optimizer: Linear program optimizer.
        objective: The objective function as a Pydantic object.
        ivars: Linear program variables.
        obj: The objective function as a list.
    """
    function_factory = {
        "max_many_variables": optimizer.max_many_variables,
        "min_many_variables": optimizer.min_many_variables,
    }
    for term in objective.terms:
        if term.type == "complex" and term.function in function_factory:
            assert isinstance(term, FunctionTermManyVariables)
            if term.variables.asset_type == "*":
                assets = ivars.filter_objective_variables_all_intervals(
                    instance_type=term.variables.asset_type,
                )
            else:
                assets = ivars.filter_objective_variables_all_intervals(
                    instance_type=term.variables.asset_type,
                    asset_name=term.variables.asset_name,
                )
            variables = []
            for ass in assets:
                for a in ass:
                    if (v := getattr(a, term.variables.variable)) is not None:
                        variables.append(v)

            # assert len(variables) == len(interval_data.idx) * len(ass)
            variables.append(float(term.constant))

            c = function_factory[term.function](
                f"{term.function}", variables=variables, M=term.M
            )

            obj.append(c * term.coefficient)


def price_objective(
    optimizer: "epl.Optimizer",
    ivars: "epl.IntervalVars",
    interval_data: "epl.assets.site.SiteIntervalData",
) -> pulp.LpAffineExpression:
    """Linear programming objective for cost minimization.  Equivalent to profit maximization.

    The objective is expressed as a linear combination of the costs for site import/export of power,
    spillage, charge for spillage EVs, gas consumption by generators and boilers.

    Args:
        optimizer: Linear program optimizer.
        ivars: Linear program variables.
        interval_data: Simulation interval data.

    Returns:
        Linear programming objective function.
    """
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
    """Linear programming objective for carbon emission minimization.

    The objective is expressed as a linear combination of the costs for site import/export of power,
    spillage, charge for spillage EVs, gas consumption by generators and boilers.

    Args:
        optimizer: Linear program optimizer.
        ivars: Linear program variables.
        interval_data: Simulation interval data.

    Returns:
        Linear programming objective function.
    """
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
    verbose: int | bool = 2,
) -> pulp.LpAffineExpression:
    """Create the objective function - either from a hardcoded function or from a custom objective function.

    Args:
        objective: The objective function.
        optimizer: Linear program optimizer.
        ivars: Linear program variables.
        interval_data: Simulation interval data.
        verbose: Level of printing.

    Returns:
        Linear programming objective function.

    Raises:
        ValueError: If objective is not a string or a dictionary.
    """
    hardcoded_objectives = {"price": price_objective, "carbon": carbon_objective}

    if isinstance(objective, str):
        if objective in hardcoded_objectives:
            return hardcoded_objectives[objective](optimizer, ivars, interval_data)
        raise ValueError(
            f"objective {objective} not in objectives, available objectives: {hardcoded_objectives.keys()}"
        )

    if isinstance(objective, dict):
        terms: list[OneTerm] = []
        for term in objective["terms"]:
            t = term_factory(term)
            assert t is not None
            terms.append(t)

        objective = CustomObjectiveFunction(terms=terms)

    else:
        assert isinstance(objective, CustomObjectiveFunction)

    obj: list[typing.Any | float] = []
    add_simple_terms(interval_data, objective, ivars, obj)
    n_simple_terms = len(obj)

    set_logging_level(logger, verbose)
    logger.debug(f"objectives.get_objective: simple_terms={n_simple_terms}")

    add_two_variable_terms(optimizer, interval_data, objective, ivars, obj)
    n_min_max_two_variables_terms = len(obj) - n_simple_terms
    logger.debug(
        f"objectives.get_objective: two_variables_terms={n_min_max_two_variables_terms})"
    )

    add_many_variables_terms(optimizer, objective, ivars, obj)
    n_many_variables_terms = len(obj) - n_simple_terms
    logger.debug(
        f"objectives.get_objective: n_many_variables_terms={n_many_variables_terms})"
    )

    return optimizer.sum(obj)
