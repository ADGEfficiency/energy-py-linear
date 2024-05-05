import dataclasses
import typing
import energypylinear as epl
import pulp

import pydantic


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
class Constraint(pydantic.BaseModel):
    """

    An aggregation of None is element wise constraint, with one constraint per interval
    sum will be one constraint per simulation

    Aggregation is across the time / interval dimension
    """

    lhs: float | ConstraintTerm | dict | list[float | ConstraintTerm | dict]
    rhs: float | ConstraintTerm | dict | list[float | ConstraintTerm | dict]
    sense: typing.Literal["le", "eq", "ge"]

    # TODO - could call `interval_aggregation`
    aggregation: typing.Literal["sum"] | None = None

    @pydantic.field_validator("lhs", "rhs")
    def parse_dicts_to_constraint_terms(
        cls, value: float | ConstraintTerm | dict | list[float | ConstraintTerm | dict]
    ) -> float | ConstraintTerm | list[float | ConstraintTerm]:
        if isinstance(value, (float, ConstraintTerm)):
            return value

        elif isinstance(value, dict):
            return ConstraintTerm(**value)

        terms = []
        values = value
        assert isinstance(values, list)
        for t in values:
            if isinstance(t, (float, ConstraintTerm)):
                terms.append(t)
            else:
                assert isinstance(t, dict)
                terms.append(ConstraintTerm(**t))

        assert len(terms) == len(values)
        return terms

    @pydantic.model_validator(mode="after")
    def validate_all(self):
        if isinstance(self.lhs, ConstraintTerm):
            lhs_all_floats = False
        elif isinstance(self.lhs, float):
            lhs_all_floats = True
        else:
            assert isinstance(self.lhs, list)
            lhs_all_floats = all([isinstance(t, float) for t in self.lhs])

        if isinstance(self.rhs, ConstraintTerm):
            rhs_all_floats = False
        elif isinstance(self.rhs, float):
            rhs_all_floats = True
        else:
            assert isinstance(self.rhs, list)
            rhs_all_floats = all([isinstance(t, float) for t in self.rhs])

        if lhs_all_floats and rhs_all_floats:
            raise ValueError("lhs and rhs cannot be all floats")

        return self


# --8<-- [end:constraint]


def _resolve_constraint_term(
    term: "epl.ConstraintTerm | dict | float",
    ivars,
    interval_data,
    i,
    divide_constant_by_idx_len: bool = False,
) -> list:
    if isinstance(term, (float, int)):
        return [term / (len(interval_data.idx) if divide_constant_by_idx_len else 1)]

    if isinstance(term, dict):
        term = epl.ConstraintTerm(**term)

    assert isinstance(term, epl.ConstraintTerm)

    if i is not None:
        vars = ivars.filter_objective_variables(
            i=i, instance_type=term.asset_type, asset_name=term.asset_name
        )
        """
        TODO - interesting thing here

        for a Site, `electric_generation_mwh` is None

        I do tackle this issue elsewhere - cannot remember where at the moment

        Why wouldn't I have `electric_generation_mwh` be 0 ?

        """
        return [
            (getattr(v, term.variable) if getattr(v, term.variable) is not None else 0)
            * (
                getattr(interval_data, term.interval_data)[i]
                if term.interval_data is not None
                else 1
            )
            * term.coefficient
            for v in vars
        ]
    else:
        vars = ivars.filter_objective_variables_all_intervals(
            instance_type=term.asset_type, asset_name=term.asset_name
        )
        assert len(vars) == len(interval_data.idx)
        return [
            (getattr(v, term.variable) if getattr(v, term.variable) is not None else 0)
            * (
                getattr(interval_data, term.interval_data)[i]
                if term.interval_data is not None
                else 1
            )
            * term.coefficient
            for i, interval in enumerate(vars)
            for v in interval
        ]


def _add_terms(
    constraint_side,
    terms: list,
    ivars: "epl.IntervalVars",
    interval_data: "epl.SiteIntervalData",
    i,
    divide_constant_by_idx_len: bool = False,
) -> None:
    if isinstance(constraint_side, (list, tuple)):
        for t in constraint_side:
            terms.extend(
                _resolve_constraint_term(
                    t,
                    ivars,
                    interval_data,
                    i=i,
                    divide_constant_by_idx_len=divide_constant_by_idx_len,
                )
            )
    else:
        terms.extend(
            _resolve_constraint_term(
                constraint_side,
                ivars,
                interval_data,
                i=i,
                divide_constant_by_idx_len=divide_constant_by_idx_len,
            )
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
            _add_terms(
                constraint.lhs,
                lhs,
                ivars,
                interval_data,
                i,
                divide_constant_by_idx_len=True,
            )
            _add_terms(
                constraint.rhs,
                rhs,
                ivars,
                interval_data,
                i,
                divide_constant_by_idx_len=True,
            )
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
