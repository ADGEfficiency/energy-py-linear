"""
Interface to the `pulp` optimization library to solve linear programming problems.

The `Optimizer` allows creating linear constraints, variables, and objectives, along with a linear program solver.
"""
import typing

import pulp


class Optimizer:
    """
    Solver for linear programs. Interfaces with `pulp`.

    Attributes:
        prob: problem to be optimized.
        solver: solver to use for solving the optimization problem.
    """

    def __init__(self) -> None:
        """Initialize an Optimizer."""
        self.prob = pulp.LpProblem("prob", pulp.LpMinimize)
        self.solver = pulp.PULP_CBC_CMD(msg=0)

    def continuous(
        self, name: str, low: float = 0, up: typing.Optional[float] = None
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

    def sum(self, vector: list[pulp.LpAffineExpression]) -> pulp.LpAffineExpression:
        """Sums a list of linear expressions.

        Args:
            vector: list of `LpAffineExpression` objects to sum.
        """
        return pulp.lpSum(vector)

    def constrain(
        self, constraint: pulp.LpConstraint, name: typing.Optional[str] = None
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
        self, verbose: int = 0, allow_infeasible: bool = False
    ) -> tuple[str, bool]:
        """Solve the optimization problem.

        Args:
            verbose: a flag indicating how verbose the output should be.  0 for no output.
            allow_infeasible: whether an infeasible solution should raise an error.

        Returns:
            status: optimization status like `Optimial` or `Infeasible`
            feasible: whether optimization was feasible or not.
        """
        self.assert_no_duplicate_variables()
        self.solver.solve(self.prob)

        status = self.status()
        if verbose > 0:
            print(f"status is {status}")

        feasible = status == "Optimal"
        if not allow_infeasible:
            assert feasible

        return status, feasible

    def assert_no_duplicate_variables(self) -> None:
        """Check there are no duplicate variable names in the optimization problem."""
        variables = self.variables()
        names = [v.name for v in variables]
        assert len(names) == len(
            set(names)
        ), f"duplicate variables detected - {[x for x in names if names.count(x) >= 2]}"

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
        self, continuous: pulp.LpVariable, binary: pulp.LpVariable, max: float
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

    def value(self, variable: typing.Union[float, pulp.LpVariable]) -> float:
        """Return the value of a linear program variable.

        Args:
            variable: either a pupl variable or already numeric.
        """
        if isinstance(variable, pulp.LpVariable):
            return variable.value()
        else:
            return float(variable)
