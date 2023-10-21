"""
Interface to the `pulp` optimization library to solve linear programming problems.

The `Optimizer` allows creating linear constraints, variables, and objectives, along with a linear program solver.
"""
import dataclasses
import datetime
import typing

import numpy as np
import pulp

from energypylinear.logger import logger


@dataclasses.dataclass
class OptimizationStatus:
    """Result of a linear program optimization."""

    status: str
    feasible: bool


@dataclasses.dataclass
class OptimizerConfig:
    """Configures the CBC optimizer.

    See https://coin-or.github.io/pulp/technical/solvers.html#pulp.apis.PULP_CBC_CMD
    """

    verbose: bool = False
    presolve: bool = True
    relative_tolerance: float = 0.02
    timeout: int = 60 * 3

    def dict(self) -> dict:
        """Creates a dictionary."""
        return dataclasses.asdict(self)


class Optimizer:
    """
    Solver for linear programs. Interfaces with `pulp`.

    Attributes:
        prob: problem to be optimized.
        solver: solver to use for solving the optimization problem.
    """

    def __init__(self, cfg: OptimizerConfig = OptimizerConfig()) -> None:
        """Initialize an Optimizer."""
        self.cfg = cfg
        name = str(datetime.datetime.now())
        name = name.replace(" ", "-")
        self.prob = pulp.LpProblem(name, pulp.LpMinimize)
        self.solver = pulp.PULP_CBC_CMD(
            msg=self.cfg.verbose,
            presolve=self.cfg.presolve,
            gapRel=self.cfg.relative_tolerance,
            timeLimit=self.cfg.timeout,
        )

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<energypylinear.Optimizer cfg: {self.cfg} variables: {len(self.variables())} constraints: {len(self.constraints())}>"

    def continuous(
        self, name: str, low: float = 0, up: float | None = None
    ) -> pulp.LpVariable:
        """Creates a new continuous linear programming variable.

        Args:
            name: The name of the variable.
            low: The lower bound of the variable.
            up: The upper bound of the variable.
        """
        # logger.debug("optimizer.continuous", name=name)
        return pulp.LpVariable(name=name, lowBound=low, upBound=up, cat="Continuous")

    def binary(self, name: str) -> pulp.LpVariable:
        """Creates a new binary linear programming variable.

        Args:
            name: The name of the variable.
        """
        # logger.debug("optimizer.binary", name=name)
        return pulp.LpVariable(name=name, cat="Binary")

    def sum(
        self, vector: list[pulp.LpAffineExpression | float]
    ) -> pulp.LpAffineExpression:
        """Sums a list of linear expressions.

        Args:
            vector: list of `LpAffineExpression` objects to sum.
        """
        return pulp.lpSum(vector)

    def constrain(
        self, constraint: pulp.LpConstraint, name: str | None = None
    ) -> pulp.LpConstraint:
        """Create a linear program constrain.

        Args:
            constraint: equality or inequality expression.
            name: optional name to give to the constraint.
        """
        return self.prob.addConstraint(constraint, name)

    def objective(self, objective: pulp.LpAffineExpression) -> pulp.LpConstraint:
        """Sets the linear program objective function.

        Args:
            objective: cost function to optimize.
        """
        return self.prob.setObjective(objective)

    def solve(
        self, verbose: bool = False, allow_infeasible: bool = False
    ) -> OptimizationStatus:
        """Solve the optimization problem.

        Args:
            verbose: a flag indicating how verbose the output should be.  0 for no output.
            allow_infeasible: whether an infeasible solution should raise an error.
        """
        logger.debug(
            "optimizer.solve",
            variables=len(self.variables()),
            constraints=len(self.constraints()),
        )
        self.assert_no_duplicate_variables()
        self.solver.solve(self.prob)

        status = self.status()
        if verbose > 0:
            logger.info("optimizer.solve", status=status)

        feasible = status == "Optimal"
        if not allow_infeasible:
            assert feasible, "Infeasible simulation!"

        return OptimizationStatus(status=status, feasible=feasible)

    def assert_no_duplicate_variables(self) -> None:
        """Check there are no duplicate variable names in the optimization problem."""
        variables = self.variables()
        names = [v.name for v in variables]
        assert len(names) == len(
            set(names)
        ), f"duplicate variables detected - {len([x for x in names if names.count(x) >= 2])} of {len(names)}\n{sorted(set([x for x in names if names.count(x) >= 2]))}"

    def status(self) -> str:
        """Return the status of the optimization problem."""
        return pulp.LpStatus[self.prob.status]

    def constraints(self) -> list[pulp.LpConstraint]:
        """Constraints of the optimization problem."""
        return self.prob.constraints

    def variables(self) -> list[pulp.LpVariable]:
        """Variables of the optimization problem."""
        return self.prob.variables()

    def constrain_max(
        self, continuous: pulp.LpVariable, binary: pulp.LpVariable | int, max: float
    ) -> pulp.LpConstraint:
        """Constrain the maximum value of a continuous variable.

        Args:
            continuous: a continuous variable, linked to the binary variable.
            binary: a binary variable, linked to the continuous variable.
            max: the allowed maximum value.
        """
        return self.constrain(continuous - binary * max <= 0)

    def constrain_min(
        self, continuous: pulp.LpVariable, binary: pulp.LpVariable, min: float
    ) -> pulp.LpConstraint:
        """Constrain the minimum value of a continuous variable.

        Args:
            continuous: a continuous variable, linked to the binary variable.
            binary: a binary variable, linked to the continuous variable.
            max: the allowed minimum value.
        """
        return self.constrain(-continuous + binary * min <= 0)

    def value(
        self,
        variable: typing.Union[float, pulp.LpVariable],
        clip_from_zero: bool = False,
    ) -> float:
        """Return the value of a linear program variable.

        Args:
            variable: either a pulp variable or number.
            clip_from_zero: optionally clip the left side at zero.
        """
        val = (
            variable.value()
            if isinstance(variable, pulp.LpVariable)
            else float(variable)
        )
        assert val is not None
        if clip_from_zero:
            val = float(np.clip(val, 0, None))
        return val
