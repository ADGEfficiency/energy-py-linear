"""Tests for `epl.interval_data`."""
import numpy as np
import pandas as pd
import pytest

import energypylinear as epl


def test_interval_data() -> None:
    """Test we can load prices and carbon intensities correctly."""
    electricity_prices = [20, 40, -100, 50]
    gas_prices = [20, 40, -100, 50]
    carbon_intensities = [0.1, 0.4, 0.3, 0.9]

    id = epl.interval_data.IntervalData(
        electricity_prices=electricity_prices,
        gas_prices=gas_prices,
        electricity_carbon_intensities=carbon_intensities,
    )
    assert all(np.array(id.electricity_prices) == np.array(electricity_prices))
    assert all(np.array(id.gas_prices) == np.array(gas_prices))
    assert all(
        np.array(id.electricity_carbon_intensities) == np.array(carbon_intensities)
    )

    id = epl.interval_data.IntervalData(
        electricity_prices=electricity_prices,
        gas_prices=40.0,
        electricity_carbon_intensities=0.05,
    )
    assert all(np.array(id.electricity_prices) == electricity_prices)
    assert all(np.array(id.gas_prices) == np.array([40, 40, 40, 40]))
    assert all(
        np.array(id.electricity_carbon_intensities)
        == np.array([0.05, 0.05, 0.05, 0.05])
    )
    assert all(id.high_temperature_load_mwh == np.array([0, 0, 0, 0]))
    assert id.idx == [0, 1, 2, 3]

    #  TODO actually test the dataframe itself
    id.to_dataframe()


def test_interval_data_numpy_arrays() -> None:
    """Test we can load data from numpy array and pandas series"""
    electricity_prices = pd.Series(np.array([20, 40, -100, 50]))
    gas_prices = np.array([20, 40, -100, 50])
    carbon_intensities = np.array([0.1, 0.4, 0.3, 0.9])

    epl.interval_data.IntervalData(
        electricity_prices=electricity_prices,
        gas_prices=gas_prices,
        electricity_carbon_intensities=carbon_intensities,
    )


def test_interval_data_no_electricity_prices() -> None:
    """Test we fail when we don't supply electricity prices."""
    carbon_intensities = [0.1, 0.4, 0.3, 0.9]

    with pytest.raises(Exception):
        epl.interval_data.IntervalData(
            electricity_carbon_intensities=carbon_intensities,
        )
