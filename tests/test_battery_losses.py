import pytest
import energypylinear
import numpy as np
#  cfg, actions, expected_losses
test_cases = (
    #  full charge for two steps, full discharge for two steps
    (
        {
            'initial_charge': 0.0,
            'power': 2.0,
            'capacity': 100,
            'episode_length': 4,
            'efficiency': 0.9
        },
        [1.0, 1.0, -1.0, -1.0],
        [0.0, 0.0, 2.0 * (1-0.95) / 2, 2.0*(1-0.95)/2]
    ),
)

