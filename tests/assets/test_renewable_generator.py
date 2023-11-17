"""Tests for the Renewable Generator asset."""
import hypothesis
import numpy as np
import pytest

import energypylinear as epl


def test_optimization_price() -> None:
    """Test optimization of the renewable generator for price."""
    electricity_prices = [-100.0, -1, 1, 100]

    #  test that we only dispatch the asset when prices are positive
    expected = [0.0, 0.0, 50, 50]
    asset = epl.RenewableGenerator(
        electric_generation_mwh=[50, 50, 50, 50],
        name="wind",
        electric_generation_lower_bound_pct=0.0,
        electricity_prices=electricity_prices,
    )
    simulation = asset.optimize()
    results = simulation.results
    np.testing.assert_array_equal(results["site-export_power_mwh"], expected)
    np.testing.assert_array_equal(results["wind-electric_generation_mwh"], expected)

    #  test when we only allow 50% turndown
    expected = [25, 25, 50, 50]
    asset = epl.RenewableGenerator(
        electric_generation_mwh=[50, 50, 50, 50],
        name="wind",
        electric_generation_lower_bound_pct=0.5,
        electricity_prices=electricity_prices,
    )
    simulation = asset.optimize()
    results = simulation.results
    np.testing.assert_array_equal(results["site-export_power_mwh"], expected)
    np.testing.assert_array_equal(results["wind-electric_generation_mwh"], expected)

    #  test that when we disallow the asset to turn down, it always generates
    expected = [50, 50, 50, 50]
    asset = epl.RenewableGenerator(
        electric_generation_mwh=[50, 50, 50, 50],
        name="wind",
        electric_generation_lower_bound_pct=1.0,
        electricity_prices=electricity_prices,
    )
    simulation = asset.optimize()
    results = simulation.results
    np.testing.assert_array_equal(results["site-export_power_mwh"], expected)
    np.testing.assert_array_equal(results["wind-electric_generation_mwh"], expected)


def test_optimization_carbon() -> None:
    """Test optimization of the renewable generator for carbon."""
    electricity_carbon_intensities = [-1.0, 0.1, 1.0]

    #  test when we allow asset to turn down
    expected = [0.0, 20, 30]
    asset = epl.RenewableGenerator(
        electric_generation_mwh=[10, 20, 30, 40],
        name="wind",
        electric_generation_lower_bound_pct=0.0,
        electricity_carbon_intensities=electricity_carbon_intensities,
    )
    simulation = asset.optimize(objective="carbon")
    results = simulation.results

    assert results.shape[0] == 3
    np.testing.assert_array_equal(results["site-export_power_mwh"], expected)
    np.testing.assert_array_equal(results["wind-electric_generation_mwh"], expected)

    #  test when we don't allow asset to turn down
    expected = [10.0, 20, 30]
    asset = epl.RenewableGenerator(
        electric_generation_mwh=[10, 20, 30, 40],
        name="wind",
        electric_generation_lower_bound_pct=1.0,
        electricity_carbon_intensities=electricity_carbon_intensities,
    )
    simulation = asset.optimize(objective="carbon")
    results = simulation.results

    assert results.shape[0] == 3
    np.testing.assert_array_equal(results["site-export_power_mwh"], expected)
    np.testing.assert_array_equal(results["wind-electric_generation_mwh"], expected)


def test_interval_data() -> None:
    """Tests the epl.RenewableGenerator and epl.Site interval data."""

    #  the happy paths
    epl.assets.renewable_generator.RenewableGeneratorIntervalData(
        electric_generation_mwh=[1.0, 2.0]
    )
    epl.assets.renewable_generator.RenewableGeneratorIntervalData(
        electric_generation_mwh=np.array([1.0, 2.0])
    )

    #  test that we transform a float into a list
    #  this is so the repeating to length will work correctly in epl.Site
    idata = epl.assets.renewable_generator.RenewableGeneratorIntervalData(
        electric_generation_mwh=2.0
    )
    assert idata.electric_generation_mwh == [2.0]

    # test that we fail with no data
    with pytest.raises(Exception):
        epl.assets.renewable_generator.RenewableGeneratorIntervalData()  # type: ignore

    # test that we fail with negative values
    with pytest.raises(Exception):
        epl.assets.renewable_generator.RenewableGeneratorIntervalData(
            electric_generation_mwh=[-10, 10]
        )


@hypothesis.settings(print_blob=True, deadline=None)
@hypothesis.given(
    idx_length=hypothesis.strategies.integers(min_value=10, max_value=24),
    prices_mu=hypothesis.strategies.floats(min_value=-1000, max_value=1000),
    prices_std=hypothesis.strategies.floats(min_value=0.1, max_value=1000),
    prices_offset=hypothesis.strategies.floats(min_value=-250, max_value=250),
    electric_generation_lower_bound_pct=hypothesis.strategies.floats(
        min_value=0, max_value=1.0
    ),
)
def test_hypothesis(
    idx_length: int,
    prices_mu: float,
    prices_std: float,
    prices_offset: float,
    electric_generation_lower_bound_pct: float,
) -> None:
    """Test optimization with hypothesis."""
    electricity_prices = (
        np.random.normal(prices_mu, prices_std, idx_length) + prices_offset
    )
    electric_generation_mwh = np.clip(
        np.random.normal(prices_mu, prices_std, idx_length) + prices_offset,
        a_min=0,
        a_max=None,
    )
    asset = epl.RenewableGenerator(
        electricity_prices=electricity_prices,
        electric_generation_mwh=electric_generation_mwh,
        electric_generation_lower_bound_pct=electric_generation_lower_bound_pct,
    )
    asset.optimize(verbose=False)
