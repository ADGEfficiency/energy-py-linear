import dataclasses
import typing
import energypylinear as epl
import pulp


# --8<-- [start:constraint-term]
@dataclasses.dataclass
class ConstraintTerm:
    variable: str
    coefficient: float = 1.0
    asset_type: str | None = None
    interval_data: str | None = None
    asset_name: str | None = None


# --8<-- [end:constraint-term]


# --8<-- [start:constraint]
@dataclasses.dataclass
class Constraint:
    """

    An aggregation of None is element wise constraint, with one constraint per interval
    sum will be one constraint per simulation

    Aggregation is across the time / interval dimension
    """

    lhs: float | ConstraintTerm | dict
    rhs: float | ConstraintTerm | dict
    sense: typing.Literal["le", "eq", "ge"]

    # TODO - could call `interval_aggregation`
    aggregation: typing.Literal["sum"] | None = None

    def dict(self) -> dict:
        return dataclasses.asdict(self)


# --8<-- [end:constraint]


def _resolve_constraint_term(
    term: "epl.ConstraintTerm | dict | float", ivars, interval_data, i
) -> list:
    if isinstance(term, (float, int)):
        return [term]

    if isinstance(term, dict):
        term = epl.ConstraintTerm(**term)

    assert isinstance(term, epl.ConstraintTerm)

    if i is not None:
        vars = ivars.filter_objective_variables(
            i=i, instance_type=term.asset_type, asset_name=term.asset_name
        )
        # TODO - add interval data
        return [getattr(v, term.variable) for v in vars]
    else:
        vars = ivars.filter_objective_variables_all_intervals(
            instance_type=term.asset_type, asset_name=term.asset_name
        )
        # TODO - add interval data
        return [getattr(v, term.variable) for interval in vars for v in interval]


def _add_terms(
    constraint_side,
    terms: list,
    ivars: "epl.IntervalVars",
    interval_data: "epl.SiteIntervalData",
    i,
) -> None:
    if isinstance(constraint_side, (list, tuple)):
        for t in constraint_side:
            terms.extend(_resolve_constraint_term(t, ivars, interval_data, i=i))
    else:
        terms.extend(
            _resolve_constraint_term(constraint_side, ivars, interval_data, i=i)
        )


def add_custom_constraint(
    optimizer: "epl.Optimizer",
    constraint: "epl.Constraint",
    ivars: "epl.IntervalVars",
    flags: "epl.Flags",
    interval_data: "epl.SiteIntervalData",
    freq: "epl.Freq",
) -> None:
    sense_mapper = {
        "le": pulp.LpConstraintLE,
        "eq": pulp.LpConstraintEQ,
        "ge": pulp.LpConstraintGE,
    }

    if isinstance(constraint, dict):
        constraint = epl.Constraint(**constraint)

    if constraint.aggregation == "sum":
        lhs = []
        rhs = []
        for i in interval_data.idx:
            _add_terms(constraint.lhs, lhs, ivars, interval_data, i)
            _add_terms(constraint.rhs, rhs, ivars, interval_data, i)
            assert len(lhs) > 0
            assert len(rhs) > 0

        optimizer.constrain(
            pulp.LpConstraint(
                e=pulp.lpSum(lhs) - pulp.lpSum(rhs),
                sense=sense_mapper[constraint.sense],
                rhs=0.0,
            )
        )

    else:
        for i in interval_data.idx:
            lhs = []
            rhs = []
            _add_terms(constraint.lhs, lhs, ivars, interval_data, i)
            _add_terms(constraint.rhs, rhs, ivars, interval_data, i)
            assert len(lhs) > 0
            assert len(rhs) > 0

            optimizer.constrain(
                pulp.LpConstraint(
                    e=pulp.lpSum(lhs) - pulp.lpSum(rhs),
                    sense=sense_mapper[constraint.sense],
                    rhs=0.0,
                )
            )
