"""Tests for the Renewable Generator asset."""
import hypothesis
import numpy as np
import pytest

import energypylinear as epl
from energypylinear.debug import debug_asset


def test_optimization_price() -> None:
    """Test optimization of the renewable generator for price."""
    electricity_prices = [-100.0, -1, 1, 100]

    #  test that we only dispatch the asset when prices are positive
    asset = epl.RenewableGenerator(
        electric_generation_mwh=[50, 50, 50, 50],
        name="wind",
        electric_generation_lower_bound_pct=0.0,
        electricity_prices=electricity_prices,
    )
    simulation = asset.optimize()
    results = simulation.results
    expected = [0.0, 0.0, 50, 50]
    np.testing.assert_array_equal(results["site-export_power_mwh"], expected)
    np.testing.assert_array_equal(results["wind-electric_generation_mwh"], expected)

    #  test that when we disallow the asset to turn down, it always generates
    asset = epl.RenewableGenerator(
        electric_generation_mwh=[50, 50, 50, 50],
        name="wind",
        electric_generation_lower_bound_pct=1.0,
        electricity_prices=electricity_prices,
    )
    simulation = asset.optimize()
    results = simulation.results
    expected = [50, 50, 50, 50]
    np.testing.assert_array_equal(results["site-export_power_mwh"], expected)
    np.testing.assert_array_equal(results["wind-electric_generation_mwh"], expected)


def test_interval_data():

    epl.assets.renewable_generator.RenewableGeneratorIntervalData(
        electric_generation_mwh=[1.0, 2.0]
    )

    idata = epl.assets.renewable_generator.RenewableGeneratorIntervalData(
        electric_generation_mwh=2.0
    )
    assert idata.electric_generation_mwh == [2.0]

    with pytest.raises(Exception):
        epl.assets.renewable_generator.RenewableGeneratorIntervalData()
