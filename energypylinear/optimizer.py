import typing

import pulp


class Pulp:
    def __init__(self):
        self.prob = pulp.LpProblem("prob", pulp.LpMinimize)
        self.solver = pulp.PULP_CBC_CMD(msg=1)

    def continuous(self, name: str, low: float = 0, up: typing.Optional[float] = None):
        return pulp.LpVariable(name=name, lowBound=low, upBound=up, cat="Continuous")

    def binary(self, name: str):
        return pulp.LpVariable(name=name, cat="Binary")

    def sum(self, vector: list[pulp.LpAffineExpression]):
        return pulp.lpSum(vector)

    def constrain(
        self, constraint: pulp.LpConstraint, name: typing.Optional[str] = None
    ):
        return self.prob.addConstraint(constraint, name)

    def objective(self, objective):
        return self.prob.setObjective(objective)

    def solve(self):
        self.assert_no_duplicate_variables()
        self.solver.solve(self.prob)
        return self.status()

    def assert_no_duplicate_variables(self):
        variables = self.variables()
        names = [v.name for v in variables]
        assert len(names) == len(
            set(names)
        ), f"duplicate variables detected - {[x for x in names if names.count(x) >= 2]}"

    def status(self):
        return pulp.LpStatus[self.prob.status]

    def constraints(self):
        return self.prob.constraints

    def variables(self):
        return self.prob.variables()

    def constrain_max(
        self, continuous: pulp.LpVariable, binary: pulp.LpVariable, max: float
    ) -> pulp.LpConstraint:
        return continuous - binary * max <= 0

    def constrain_min(
        self, continuous: pulp.LpVariable, binary: pulp.LpVariable, min: float
    ) -> pulp.LpConstraint:
        return -continuous + binary * min <= 0
