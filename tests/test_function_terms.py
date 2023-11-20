"""Tests the use of FunctionTerms in a custom objective function."""
import hypothesis
import numpy as np
import pulp

import energypylinear as epl

float_args = {
    "allow_infinity": False,
    "allow_nan": False,
    "min_value": -100000,
    "max_value": 100000,
}
gap_args = {
    "allow_infinity": False,
    "allow_nan": False,
    "min_value": 0.1,
    "max_value": 10000,
}

atol = 1e-4

settings = {
    "print_blob": True,
    "max_examples": 5000,
    "verbosity": hypothesis.Verbosity.verbose,
}


def coerce_variables(
    a: float,
    b: float,
    a_gap: float,
    b_gap: float,
    a_is_float: bool,
    b_is_float: bool,
    opt: "epl.Optimizer",
) -> tuple[float | pulp.LpVariable, float | pulp.LpVariable]:
    """Helper function to transform hypothesis parameters."""
    if a_is_float:
        av = a
    else:
        av = opt.continuous("a", low=a, up=a + a_gap)

    if b_is_float and not a_is_float:
        bv = b
    else:
        bv = opt.continuous("b", low=b, up=b + b_gap)

    if isinstance(av, float):
        assert isinstance(bv, pulp.LpVariable)
    if isinstance(bv, float):
        assert isinstance(av, pulp.LpVariable)

    return av, bv


@hypothesis.settings(**settings)
@hypothesis.given(
    a=hypothesis.strategies.floats(**float_args),
    b=hypothesis.strategies.floats(**float_args),
    a_gap=hypothesis.strategies.floats(**gap_args),
    b_gap=hypothesis.strategies.floats(**gap_args),
    a_is_float=hypothesis.strategies.booleans(),
    b_is_float=hypothesis.strategies.booleans(),
)
def test_max_two_variables(
    a: float, b: float, a_gap: float, b_gap: float, a_is_float: bool, b_is_float: bool
) -> None:
    """Tests that we can constrain a variable to be the maximum of two other variables."""
    opt = epl.Optimizer()
    av, bv = coerce_variables(a, b, a_gap, b_gap, a_is_float, b_is_float, opt)
    cv = opt.max_two_variables(
        "max-a-b", av, bv, M=max(abs(a) + a_gap, abs(b) + b_gap) * 2.0
    )
    opt.objective(av + bv)
    opt.solve(verbose=3)
    np.testing.assert_allclose(max(a, b), cv.value(), atol=atol)


@hypothesis.settings(**settings)
@hypothesis.given(
    a=hypothesis.strategies.floats(**float_args),
    b=hypothesis.strategies.floats(**float_args),
    a_gap=hypothesis.strategies.floats(**gap_args),
    b_gap=hypothesis.strategies.floats(**gap_args),
    a_is_float=hypothesis.strategies.booleans(),
    b_is_float=hypothesis.strategies.booleans(),
)
def test_min_two_variables(
    a: float, b: float, a_gap: float, b_gap: float, a_is_float: bool, b_is_float: bool
) -> None:
    """Tests that we can constrain a variable to be the maximum of two other variables."""
    opt = epl.Optimizer()
    av, bv = coerce_variables(a, b, a_gap, b_gap, a_is_float, b_is_float, opt)
    cv = opt.min_two_variables(
        "min-a-b", av, bv, M=max(abs(a) + a_gap, abs(b) + b_gap) * 2.0
    )
    opt.objective(av + bv)
    opt.solve(verbose=3)
    np.testing.assert_allclose(min(a, b), cv.value(), atol=atol)
