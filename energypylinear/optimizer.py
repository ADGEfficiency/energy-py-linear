"""Interface to the `pulp` optimization library to solve linear programming problems.

The `Optimizer` allows creating linear constraints, variables, and objectives, along with a linear program solver."""
import dataclasses
import datetime
import typing

import numpy as np
import pulp

from energypylinear.logger import logger, set_logging_level


@dataclasses.dataclass
class OptimizationStatus:
    """Result of a linear program optimization."""

    status: str
    feasible: bool
    objective: float


@dataclasses.dataclass
class OptimizerConfig:
    """Configures the CBC optimizer.

    See https://coin-or.github.io/pulp/technical/solvers.html#pulp.apis.PULP_CBC_CMD
    """

    verbose: bool = False
    presolve: bool = True
    relative_tolerance: float = 0.0
    timeout: int = 60 * 3

    def dict(self) -> dict:
        """Creates a dictionary."""
        return dataclasses.asdict(self)


class Optimizer:
    """Solver for linear programs. Interfaces with `pulp`.

    Attributes:
        cfg: an OptimizerConfig
        prob: problem to be optimized.
        solver: solver to use for solving the optimization problem.
    """

    def __init__(self, cfg: OptimizerConfig | dict = OptimizerConfig()) -> None:
        """Initialize an Optimizer."""

        if isinstance(cfg, dict):
            cfg = OptimizerConfig(**cfg)

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
        self, name: str, low: float | None = 0, up: float | None = None
    ) -> pulp.LpVariable:
        """Creates a new continuous linear programming variable.

        Args:
            name: The name of the variable.
            low: The lower bound of the variable.
            up: The upper bound of the variable.
        """
        return pulp.LpVariable(name=name, lowBound=low, upBound=up, cat="Continuous")

    def binary(self, name: str) -> pulp.LpVariable:
        """Creates a new binary linear programming variable.

        Args:
            name: The name of the variable.
        """
        return pulp.LpVariable(name=name, cat="Binary")

    def sum(
        self, vector: list[pulp.LpAffineExpression | float]
    ) -> pulp.LpAffineExpression:
        """Sums a list of linear expressions.

        Args:
            vector: list of `LpAffineExpression` objects to sum.
        """
        return pulp.lpSum(vector)

    def constrain(self, constraint: pulp.LpConstraint, name: str | None = None) -> None:
        """Create a linear program constrain.

        Args:
            constraint: equality or inequality expression.
            name: optional name to give to the constraint.
        """
        self.prob.addConstraint(constraint, name)

    def objective(self, objective: pulp.LpAffineExpression) -> None:
        """Sets the linear program objective function.

        Args:
            objective: cost function to optimize.
        """
        self.prob.setObjective(objective)

    def solve(
        self, verbose: int | bool, allow_infeasible: bool = False
    ) -> OptimizationStatus:
        """Solve the optimization problem.

        Args:
            verbose: a flag indicating how verbose the output should be.  0 for no output.
            allow_infeasible: whether an infeasible solution should raise an error.
        """
        set_logging_level(logger, level=verbose)

        logger.debug(
            f"optimizer.solve: variables={len(self.variables())}, constraints={len(self.constraints())}"
        )
        self.assert_no_duplicate_variables()

        self.solver.solve(self.prob)
        status = self.status()
        objective = pulp.value(self.prob.objective)
        logger.info(
            f"optimizer.solve: {status=}, {objective=}",
        )
        feasible = status == "Optimal"
        if not allow_infeasible:
            assert feasible, "Infeasible simulation!"
            assert objective is not None, f"Solver failure - objective is {objective}!"

        return OptimizationStatus(status=status, feasible=feasible, objective=objective)

    def assert_no_duplicate_variables(self) -> None:
        """Check there are no duplicate variable names in the optimization problem."""
        variables = self.variables()
        assert isinstance(variables, list)
        names = [v.name for v in variables]
        assert (
            len(names) == len(set(names))
        ), f"duplicate variables detected - {len([x for x in names if names.count(x) >= 2])} of {len(names)}\n{sorted(set([x for x in names if names.count(x) >= 2]))}"

    def status(self) -> str:
        """Return the status of the optimization problem."""
        return str(pulp.LpStatus[self.prob.status])

    def constraints(self) -> dict[str, pulp.LpConstraint]:
        """Constraints of the optimization problem."""
        return self.prob.constraints

    def variables(
        self, dict: bool = False
    ) -> list[pulp.LpVariable] | dict[str, pulp.LpVariable]:
        """Variables of the optimization problem."""
        if dict is False:
            return self.prob.variables()
        else:
            return {v.name: v for v in self.prob.variables()}

    def constrain_max(
        self, continuous: pulp.LpVariable, binary: pulp.LpVariable | int, max: float
    ) -> None:
        """Constrain the maximum value of a continuous variable.

        Args:
            continuous: a continuous variable, linked to the binary variable.
            binary: a binary variable, linked to the continuous variable.
            max: the allowed maximum value.
        """
        self.constrain(continuous - binary * max <= 0)

    def constrain_min(
        self, continuous: pulp.LpVariable, binary: pulp.LpVariable, min: float
    ) -> None:
        """Constrain the minimum value of a continuous variable.

        Args:
            continuous: a continuous variable, linked to the binary variable.
            binary: a binary variable, linked to the continuous variable.
            max: the allowed minimum value.
        """
        self.constrain(-continuous + binary * min <= 0)

    def min_two_variables(
        self, name: str, a: pulp.LpVariable, b: pulp.LpVariable, M: float
    ) -> pulp.LpVariable:
        """Create a variable that is the minimum of two other variables.

        The variables can be either both linear program variables, or one can be a float.  Both cannot be floats.

        Args:
            name: Name of the maximum variables.
            a: A continuous variable or float.
            b: A continuous variable or float.
            M: The big-M parameter.
        """
        if isinstance(a, float):
            assert not isinstance(b, float)

        assert "min" in name
        cv = self.continuous(name=f"{name}-continuous", low=None, up=None)
        cb = self.binary(name=f"{name}-binary")

        self.constrain(a >= cv)
        self.constrain(b >= cv)
        self.constrain(cv >= a - M * (1 - cb))
        self.constrain(cv >= b - M * cb)
        return cv

    def max_two_variables(
        self,
        name: str,
        a: pulp.LpVariable | float,
        b: pulp.LpVariable | float,
        M: float,
    ) -> pulp.LpVariable:
        """Create a variable that is the maximum of two other variables.

        The variables can be either both linear program variables, or one can be a float.  Both cannot be floats.

        Args:
            name: Name of the maximum variables.
            a: A continuous variable or float.
            b: A continuous variable or float.
            M: The big-M parameter.
        """
        if isinstance(a, float):
            assert not isinstance(b, float)

        assert "max" in name
        cv = self.continuous(name=f"{name}-continuous", low=None, up=None)
        cb = self.binary(name=f"{name}-binary")

        self.constrain(cv >= a)
        self.constrain(cv >= b)
        self.constrain(a + M * cb >= cv)
        self.constrain(b + M * (1 - cb) >= cv)
        return cv

    def min_many_variables(
        self, name: str, variables: list[pulp.LpVariable | float], M: float
    ) -> pulp.LpVariable:
        """Create a variable that is the minimum of many other variables.

        All the variables should be greater than or equal to zero.

        Args:
            name: Name of the new variable.
            variables: A list of continuous variables or floats.
            M: The big-M parameter.
        """
        assert "min_many" in name
        min_var = self.continuous(name=f"{name}-continuous", low=None, up=None)
        binary_vars = [self.binary(f"{name}-binary-{i}") for i in range(len(variables))]

        for v in variables:
            if isinstance(v, pulp.LpVariable):
                assert v.lowBound >= 0
            else:
                assert v >= 0

        for var, binary_var in zip(variables, binary_vars):
            self.constrain(min_var <= var)
            self.constrain(var - M * binary_var <= min_var)

        self.constrain(self.sum(binary_vars) <= len(variables) - 1)
        return min_var

    def max_many_variables(
        self, name: str, variables: list[pulp.LpVariable | float], M: float
    ) -> pulp.LpVariable:
        """Create a variable that is the maximum of many other variables.

        The variables can be either linear program variables or floats.

        All the variables should be greater than or equal to zero.

        Args:
            name: Name of the maximum variables.
            variables: A list of continuous variables or floats.
            M: The big-M parameter.
        """
        assert "max_many" in name
        max_var = self.continuous(name=f"{name}-continuous", low=None, up=None)
        binary_vars = [self.binary(f"{name}-binary-{i}") for i in range(len(variables))]

        for v in variables:
            if isinstance(v, pulp.LpVariable):
                assert v.lowBound >= 0
            else:
                assert v >= 0

        for var, binary_var in zip(variables, binary_vars):
            self.constrain(max_var >= var)
            if isinstance(var, pulp.LpVariable):
                self.constrain(var + M * (1 - binary_var) >= max_var)
        return max_var

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
