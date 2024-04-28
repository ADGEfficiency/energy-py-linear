import dataclasses
import typing
import energypylinear as epl


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
):
    if isinstance(term, (float, int)):
        return term

    if isinstance(term, dict):
        term = epl.ConstraintTerm(**term)

    assert isinstance(term, epl.ConstraintTerm)

    if i is not None:
        vars = ivars.filter_objective_variables(
            i=i, instance_type=term.asset_type, asset_name=term.asset_name
        )
        return [getattr(v, term.variable) for v in vars]
    else:
        vars = ivars.filter_objective_variables_all_intervals(
            instance_type=term.asset_type, asset_name=term.asset_name
        )
        return [getattr(v, term.variable) for interval in vars for v in interval]


def add_custom_constraint(
    optimizer: "epl.Optimizer",
    constraint: "epl.Constraint",
    ivars: "epl.IntervalVars",
    flags: "epl.Flags",
    interval_data: "epl.SiteIntervalData",
    freq: "epl.Freq",
) -> None:
    # TODO - will need to deal with list here at some point

    if isinstance(constraint, dict):
        constraint = epl.Constraint(**constraint)

    if constraint.aggregation == "sum":
        import pulp

        lhs = []
        rhs = []
        for i in interval_data.idx:
            lhs.append(
                _resolve_constraint_term(constraint.lhs, ivars, interval_data, i=i)
            )
            rhs.append(
                _resolve_constraint_term(constraint.rhs, ivars, interval_data, i=i)
            )
        sense_mapper = {
            "le": pulp.LpConstraintLE,
            "eq": pulp.LpConstraintEQ,
            "ge": pulp.LpConstraintGE,
        }
        optimizer.constrain(
            pulp.LpConstraint(
                e=pulp.lpSum(lhs) - pulp.lpSum(rhs),
                sense=sense_mapper[constraint.sense],
                rhs=0.0,
            )
        )

    else:
        for i in interval_data.idx:
            lhs = _resolve_constraint_term(constraint.lhs, ivars, interval_data, i=i)
            rhs = _resolve_constraint_term(constraint.rhs, ivars, interval_data, i=i)

            if not isinstance(lhs, list):
                lhs = [lhs]

            if not isinstance(rhs, list):
                rhs = [rhs]

            if len(lhs) < len(rhs):
                assert len(lhs) == 1
                lhs = lhs * len(rhs)

            if len(rhs) < len(lhs):
                assert len(rhs) == 1
                rhs = rhs * len(lhs)

            assert len(lhs) == len(rhs)

            for l, r in zip(lhs, rhs):
                import pulp

                sense_mapper = {
                    "le": pulp.LpConstraintLE,
                    "eq": pulp.LpConstraintEQ,
                    "ge": pulp.LpConstraintGE,
                }
                optimizer.constrain(
                    pulp.LpConstraint(e=l, sense=sense_mapper[constraint.sense], rhs=r)
                )
