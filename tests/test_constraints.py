"""Tests the constraints on minimum and maximum variables."""
import random

import hypothesis
import numpy as np
import pulp
import pytest

import energypylinear as epl

atol = 1e-4
settings = hypothesis.settings(
    print_blob=True,
    max_examples=1000,
    report_multiple_bugs=False,
)


def coerce_variables(
    a: float,
    b: float,
    a_gap: float,
    b_gap: float,
    a_is_float: bool,
    b_is_float: bool,
    opt: "epl.Optimizer",
) -> tuple[float | pulp.LpVariable, float | pulp.LpVariable]:
    """Helper function to transform hypothesis parameters.

    a_is_float | b_is_float | av        | bv
    True       | True       | float     | float
    True       | False      | float     | lpvar
    False      | True       | lpvar     | float
    False      | False      | lpvar     | lpvar
    """
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


@pytest.mark.parametrize(
    "a_is_float, b_is_float, expected_av_type, expected_bv_type",
    [
        (True, True, "float", "lpvar"),
        (True, False, "float", "lpvar"),
        (False, True, "lpvar", "float"),
        (False, False, "lpvar", "lpvar"),
    ],
)
def test_coerce_variables(
    a_is_float: bool, b_is_float: bool, expected_av_type: str, expected_bv_type: str
) -> None:
    """Test the coerce variables helper.

    Args:
        a_is_float: Whether a is a float or LpVariable.
        b_is_float: Whether b is a float or LpVariable.
        expected_av_type: Expected type of av.
        expected_bv_type: Expected type of bv.
    """
    a, b, a_gap, b_gap = 1.0, 2.0, 0.5, 0.5
    opt = epl.Optimizer()
    av, bv = coerce_variables(a, b, a_gap, b_gap, a_is_float, b_is_float, opt)

    if expected_av_type == "float":
        assert isinstance(av, float)
    else:
        assert isinstance(av, pulp.LpVariable)

    if expected_bv_type == "float":
        assert isinstance(bv, float)
    else:
        assert isinstance(bv, pulp.LpVariable)


@hypothesis.settings(settings)
@hypothesis.given(
    a=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=-100000, max_value=100000
    ),
    b=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=-100000, max_value=100000
    ),
    a_gap=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=0.1, max_value=10000
    ),
    b_gap=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=0.1, max_value=10000
    ),
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


@hypothesis.settings(settings)
@hypothesis.given(
    a=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=-100000, max_value=100000
    ),
    b=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=-100000, max_value=100000
    ),
    a_gap=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=0.1, max_value=10000
    ),
    b_gap=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=0.1, max_value=10000
    ),
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


@hypothesis.settings(settings)
@hypothesis.given(
    n=hypothesis.strategies.integers(min_value=5, max_value=100),
)
def test_max_many_variables(n: int) -> None:
    """Tests that we can constrain a variable to be the maximum of many other variables."""
    opt = epl.Optimizer()
    lp_vars = [opt.continuous(name=f"v{i}", low=0) for i in range(n)]
    maxes = [random.random() * 100 for _ in lp_vars]
    for m, v in zip(maxes, lp_vars):
        opt.constrain(v == m)

    max_var = opt.max_many_variables("max-many", lp_vars, M=max(maxes))
    opt.objective(opt.sum(lp_vars))
    opt.solve(verbose=0)
    np.testing.assert_allclose(max(maxes), max_var.value(), atol=atol)


@hypothesis.settings(settings)
@hypothesis.given(
    n=hypothesis.strategies.integers(min_value=5, max_value=100),
)
def test_min_many_variables(n: int) -> None:
    """Tests that we can constrain a variable to be the minimum of many other variables."""
    opt = epl.Optimizer()
    lp_vars = [opt.continuous(name=f"v{i}", low=0) for i in range(n)]
    mins = [random.random() * 100 for _ in lp_vars]
    for m, v in zip(mins, lp_vars):
        opt.constrain(v == m)
    min_var = opt.min_many_variables("min-many", lp_vars, M=max(mins) * 100)
    opt.objective(opt.sum(lp_vars))
    opt.solve(verbose=0)
    print(mins)
    np.testing.assert_allclose(min(mins), min_var.value(), atol=atol)
