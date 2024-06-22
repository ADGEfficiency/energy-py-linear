"""Add custom constraints to the linear program."""
import dataclasses
import typing

import pulp
import pydantic

import energypylinear as epl


# --8<-- [start:constraint-term]
@dataclasses.dataclass
class ConstraintTerm:
    """A term in a constraint.

    The sum of terms creates the two sides of a constraint,
    the left-hand side (LHS) and right-hand side (RHS).

    Examples:

    ```python
    # a constraint term for site import power electricity cost
    ConstraintTerm(
        variable="import_power_mwh",
        asset_type="site",
        interval_data="electricity_prices"
    )

    # a constraint term for site export power electricity revenue
    ConstraintTerm(
        variable="import_power_mwh",
        asset_type="site",
        interval_data="electricity_prices",
        coefficient=-1
    )

    # a constraint term for battery cycle cost
    ConstraintTerm(
        variable="electric_charge_mwh",
        asset_type="battery",
        coefficient=0.25
    )
    ```

    Attributes:
        variable: The linear program variable.  This will be an
            attribute of a OneInterval object, like `import_power_mwh`
            or `gas_consumption_mwh`.
        asset_type: The type of asset, such as `battery` or `chp`.
            `*` will include all assets.
        interval_data: The interval data variable, such as
            `electricity_prices` or `gas_prices`.
        asset_name: The name of a specific asset.
        coefficient: A constant multipler for the term.
    """

    variable: str
    asset_type: str | None = None
    interval_data: str | None = None
    asset_name: str | None = None
    coefficient: float = 1.0


# --8<-- [end:constraint-term]


# --8<-- [start:constraint]
class Constraint(pydantic.BaseModel):
    """A custom constraint.

    Made of a left-hand side (LHS), a right-hand side (RHS) and a sense (<=, ==, >=).

    Attributes:
        lhs: The left-hand side of the constraint.
        rhs: The right-hand side of the constraint.
        sense: The constraint sense and a sense (<=, ==, >=).
        interval_aggregation: How to aggregate terms across intervals.
            None will result in one constraint per interval.
            "sum" will result in one constraint per simulation.
    """

    lhs: float | ConstraintTerm | dict | list[float | ConstraintTerm | dict]
    rhs: float | ConstraintTerm | dict | list[float | ConstraintTerm | dict]
    sense: typing.Literal["le", "eq", "ge"]
    interval_aggregation: typing.Literal["sum"] | None = None

    # --8<-- [end:constraint]

    @pydantic.field_validator("lhs", "rhs")
    def parse_dicts_to_constraint_terms(
        cls, value: float | ConstraintTerm | dict | list[float | ConstraintTerm | dict]
    ) -> float | ConstraintTerm | list[float | ConstraintTerm]:
        """Converts a dict to a ConstraintTerm."""
        if isinstance(value, (float, ConstraintTerm)):
            return value

        # elif isinstance(value, dict):
        #     return ConstraintTerm(**value)

        terms = []
        values = value
        assert isinstance(values, list)
        for t in values:
            """
            if isinstance(t, (float, ConstraintTerm)):
                terms.append(t)
            else:
                assert isinstance(t, dict)
                terms.append(ConstraintTerm(**t))
            """
            assert isinstance(t, (float, ConstraintTerm))
            terms.append(t)

        assert len(terms) == len(values)
        return terms

    @pydantic.model_validator(mode="after")
    def validate_all(self) -> typing.Self:
        """Validates that our LHS and RHS make sense."""
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


def _resolve_constraint_term(
    term: "epl.ConstraintTerm | dict | float",
    ivars: "epl.IntervalVars",
    interval_data: "epl.SiteIntervalData",
    i: int,
    divide_constant_by_idx_len: bool = False,
) -> list:
    """Converts a ConstraintTerm to a float or list of linear program variables."""
    if isinstance(term, (float, int)):
        return [term / (len(interval_data.idx) if divide_constant_by_idx_len else 1)]

    assert isinstance(term, epl.ConstraintTerm)

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


def _add_terms(
    constraint_side: float
    | ConstraintTerm
    | dict
    | list[float | ConstraintTerm | dict],
    terms: list,
    ivars: "epl.IntervalVars",
    interval_data: "epl.SiteIntervalData",
    i: int,
    divide_constant_by_idx_len: bool = False,
) -> None:
    """Iterates across terms in the LHS or RHS of a constraint."""
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
    interval_data: "epl.SiteIntervalData",
    constraint: "epl.Constraint | dict",
    ivars: "epl.IntervalVars",
) -> None:
    """Add a custom constraint to the linear program.

    Args:
        optimizer: Linear program optimizer.
        interval_data: Simulation interval data.
        constraint: A custom constraint.
        ivars: Linear program variables.
    """
    sense_mapper = {
        "le": pulp.LpConstraintLE,
        "eq": pulp.LpConstraintEQ,
        "ge": pulp.LpConstraintGE,
    }

    if isinstance(constraint, dict):
        constraint = epl.Constraint(**constraint)

    lhs: list = []
    rhs: list = []
    for i in interval_data.idx:
        _add_terms(
            constraint.lhs,
            lhs,
            ivars,
            interval_data,
            i,
            divide_constant_by_idx_len=constraint.interval_aggregation == "sum",
        )
        _add_terms(
            constraint.rhs,
            rhs,
            ivars,
            interval_data,
            i,
            divide_constant_by_idx_len=constraint.interval_aggregation == "sum",
        )
        assert len(lhs) > 0
        assert len(rhs) > 0

        if constraint.interval_aggregation != "sum":
            optimizer.constrain(
                pulp.LpConstraint(
                    e=pulp.lpSum(lhs) - pulp.lpSum(rhs),
                    sense=sense_mapper[constraint.sense],
                    rhs=0.0,
                )
            )
            lhs = []
            rhs = []

    if constraint.interval_aggregation == "sum":
        optimizer.constrain(
            pulp.LpConstraint(
                e=pulp.lpSum(lhs) - pulp.lpSum(rhs),
                sense=sense_mapper[constraint.sense],
                rhs=0.0,
            )
        )
