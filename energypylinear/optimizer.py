import typing

import pulp


class Pulp:
    def __init__(self):
        self.prob = pulp.LpProblem()
        self.solver = pulp.PULP_CBC_CMD(msg=0)

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
        self.solver.solve(self.prob)
        return self.status()

    def status(self):
        return pulp.LpStatus[self.prob.status]

    def constraints(self):
        return self.prob.constraints

    def constrain_max(
        self, continuous: pulp.LpVariable, binary: pulp.LpVariable, max: float
    ) -> pulp.LpConstraint:
        return continuous - binary * max <= 0

    def constrain_min(
        self, continuous: pulp.LpVariable, binary: pulp.LpVariable, max: float
    ) -> pulp.LpConstraint:
        return -continuous + binary * max <= 0
