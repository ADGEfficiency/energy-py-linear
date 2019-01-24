import pytest
import pandas as pd
import numpy as np

import energypylinear


def test_view_output():
    """Test that we output the expected view."""
    prices = [10, 20, 30, 40, 10, 50]

    model = energypylinear.Battery(
        power=2, capacity=4
    )

    info = model.optimize(
        prices=prices, initial_charge=0
    )

    expected_header = ['Import [MW]', 'Export [MW]', 'Gross [MW]', 'Net [MW]',
                       'Losses [MW]', 'Charge [MWh]', 'Prices [$/MWh]',
                       'Forecast [$/MWh]', 'Actual [$/5min]',
                       'Forecast [$/5min]']

    results = pd.DataFrame(info, columns=info[0].keys())
    for header in expected_header:
        assert(header in results.columns)

    result_rows = results.values.tolist()
    assert result_rows[0] == [2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 10.0, 10.0, 1.6666666666666667, 1.6666666666666667]
    assert result_rows[1] == [2.0, 0.0, 2.0, 2.0, 0.0, 0.16666667, 20.0, 20.0, 3.3333333333333335, 3.3333333333333335]
    assert result_rows[2] == [0.0, 1.6363636, -1.6363636, -1.47272724, 0.16363636, 0.33333333, 30.0, 30.0, -3.6818180999999996, -3.6818180999999996]
    assert result_rows[3] == [0.0, 2.0, -2.0, -1.8, 0.2, 0.18333333, 40.0, 40.0, -6.0, -6.0]
    assert result_rows[4] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0]
    assert np.isnan(result_rows[5]).any()
    assert result_rows[5][5] == 0.0
    assert result_rows[5][6] == 50.0
    assert result_rows[5][7] == 50.0
