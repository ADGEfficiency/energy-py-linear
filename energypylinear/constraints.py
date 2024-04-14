import dataclasses
import typing
import energypylinear as epl


# --8<-- [start:constraint-term]
@dataclasses.dataclass
class ConstraintTerm:
    """

    An aggregation of None is element wise constraint, with one constraint per interval
    sum will be one constraint per simulation

    Aggregation is across the time / interval dimension
    """

    variable: str
    coefficient: float = 1.0
    asset_type: str | None = None
    interval_data: str | None = None
    asset_name: str | None = None
    aggregation: typing.Literal["sum"] | None = None


# --8<-- [end:constraint-term]


# --8<-- [start:constraint]
@dataclasses.dataclass
class Constraint:
    lhs: list[float | ConstraintTerm | dict]
    rhs: list[float | ConstraintTerm | dict]
    sense: typing.Literal["le", "eq", "ge"]


# --8<-- [end:constraint]


def add_custom_constraint(
    optimizer: "epl.Optimizer",
    constraint: "epl.Constraint",
    ivars: "epl.IntervalVars",
    flags: "epl.Flags",
    interval_data: "epl.SiteIntervalData",
    freq: "epl.Freq",
) -> None:
    breakpoint()  # fmt: skip

    lhs = []
    for term in constraint.lhs:
        if isinstance(term, float):
            lhs.append(term)
        else:
            assert isinstance(term, epl.ConstraintTerm)

    optimizer.constrain()
