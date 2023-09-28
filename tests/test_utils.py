"""Test utility functions."""
import numpy as np

from energypylinear.utils import repeat_to_match_length


def test_repeat_to_match_length() -> None:
    """Tests the repeat_to_match_length function."""
    assert all(
        repeat_to_match_length(np.array([1.0, 2.0, 3.0]), np.zeros(5))
        == np.array([1.0, 2.0, 3.0, 1.0, 2.0])
    )
