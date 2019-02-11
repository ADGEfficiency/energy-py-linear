import pytest
import io
import csv

import energypylinear


def test_view_output():
    """Test that we output the expected view for a simple problem."""
    prices = [10, 20, 10]

    model = energypylinear.Battery(
        power=2, capacity=4, efficiency=1.0
    )

    info = model.optimize(
        prices=prices, initial_charge=0, timestep='1hr'
    )
    expected_header = ['Import [MW]', 'Export [MW]', 'Gross [MW]', 'Net [MW]',
                       'Losses [MW]', 'Charge [MWh]', 'Prices [$/MWh]',
                       'Forecast [$/MWh]', 'Actual [$/1hr]',
                       'Forecast [$/1hr]']

    csvfile = io.StringIO()
    fieldnames = info[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for row in info:
        writer.writerow(row)

    csv_rows = csvfile.getvalue().split('\r\n')
    for header in expected_header:
        assert(header in csv_rows[0])

    #  order of the csv columns is different than from expected_header
    actual_header = [col for col in csv_rows[0].split(',')]
    row_one = [col for col in csv_rows[1].split(',')]
    row_two = [col for col in csv_rows[2].split(',')]

    assert float(row_one[actual_header.index('Import [MW]')]) == 2.0
    assert float(row_one[actual_header.index('Export [MW]')]) == 0.0
    assert float(row_one[actual_header.index('Prices [$/MWh]')]) == 10

    assert float(row_two[actual_header.index('Import [MW]')]) == 0.0
    assert float(row_two[actual_header.index('Export [MW]')]) == 2.0
    assert float(row_two[actual_header.index('Prices [$/MWh]')]) == 20

if __name__ == '__main__':
    test_view_output()
