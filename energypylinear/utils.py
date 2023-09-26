"""Utility functions."""
import numpy as np


def repeat_to_match_length(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Repeats an array to match the length of another array."""
    quotient, remainder = divmod(len(b), len(a))
    return np.concatenate([np.tile(a, quotient), a[:remainder]])


def check_array_lengths(results: dict[str, list]) -> None:
    """Check that all lists in the results dictionary have the same length.

    Args:
        results (dict[str, list]):
            Dictionary containing lists whose lengths need to be checked.

    Raises:
        AssertionError: If lists in the dictionary have different lengths.
    """
    lens = []
    dbg = []
    for k, v in results.items():
        lens.append(len(v))
        dbg.append((k, len(v)))
    assert len(set(lens)) == 1, f"{len(set(lens))} {dbg}"
