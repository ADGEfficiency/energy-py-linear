"""Test Battery asset."""

import time

import hypothesis
import numpy as np
import pandas as pd
import pytest

import energypylinear as epl

#  maybe can move to defaults / constants
tol = 1e-5


@pytest.mark.parametrize(
    "electricity_prices, initial_charge_mwh, expected_dispatch",
    [
        ([10, 10, 10], 0, [0, 0, 0]),
        ([20, 10, 10], 6, [-4, -2, 0]),
        ([10, 50, 10, 5000, 10], 0, [4, -4, 4, -4, 0]),
    ],
)
def test_price_optimization(
    electricity_prices: list[float],
    initial_charge_mwh: float,
    expected_dispatch: list[float],
) -> None:
    """Test optimization for price."""
    power_mw = 4
    capacity_mwh = 6
    efficiency = 1.0
    freq_mins = 60
    asset = epl.Battery(
        power_mw=power_mw,
        capacity_mwh=capacity_mwh,
        efficiency_pct=efficiency,
        electricity_prices=np.array(electricity_prices),
        freq_mins=freq_mins,
        initial_charge_mwh=initial_charge_mwh,
        final_charge_mwh=0,
    )
    simulation = asset.optimize(
        verbose=False,
    )
    charge = simulation.results["battery-electric_charge_mwh"].values
    discharge = simulation.results["battery-electric_discharge_mwh"].values
    dispatch = charge - discharge
    np.testing.assert_almost_equal(dispatch, expected_dispatch)


@pytest.mark.parametrize(
    "carbon_intensities, initial_charge_mwh, expected_dispatch",
    [
        ([0.1, 0.8, -0.1], 3, [1, -4, 3]),
        ([0.9, 0.1, 0.2, 0.9], 6, [-4, 4, 0, 0]),
        ([0.9, 0.1, 0.2, 0.9], 0, [0, 4, 0, -4]),
    ],
)
def test_carbon_optimization(
    carbon_intensities: list[float],
    initial_charge_mwh: float,
    expected_dispatch: list[float],
) -> None:
    """Test optimization for carbon."""
    power_mw = 4
    capacity_mwh = 6
    efficiency = 1.0
    freq_mins = 60
    prices = np.zeros_like(carbon_intensities)
    asset = epl.Battery(
        power_mw=power_mw,
        capacity_mwh=capacity_mwh,
        efficiency_pct=efficiency,
        electricity_prices=prices,
        electricity_carbon_intensities=carbon_intensities,
        freq_mins=freq_mins,
        initial_charge_mwh=initial_charge_mwh,
    )
    simulation = asset.optimize(
        objective="carbon",
    )
    charge = simulation.results["battery-electric_charge_mwh"].values
    discharge = simulation.results["battery-electric_discharge_mwh"].values
    dispatch = charge - discharge
    np.testing.assert_almost_equal(dispatch, expected_dispatch)


def test_simultaneous_charge_discharge() -> None:
    """Test that we don't ever simultaneously charge and discharge."""
    electricity_prices = np.random.normal(100, 1000, 1024)
    power_mw = 400
    capacity_mwh = 100
    initial_charge_mwh = 0
    final_charge_mwh = capacity_mwh

    asset = epl.Battery(
        power_mw=power_mw,
        capacity_mwh=capacity_mwh,
        electricity_prices=electricity_prices,
        initial_charge_mwh=initial_charge_mwh,
        final_charge_mwh=final_charge_mwh,
    )
    asset.optimize(
        verbose=False,
    )


def check_no_simultaneous(
    df: pd.DataFrame, left_col: str, right_col: str
) -> tuple[bool, pd.DataFrame]:
    """Checks that we don't do two things at once."""
    checks = (
        ((df[left_col] > 0) & (df[right_col] == 0))
        | ((df[right_col] > 0) & (df[left_col] == 0))
        | ((df[left_col] == 0) & (df[right_col] == 0))
    )
    return (
        checks.all(),
        df.loc[
            ~checks,
            [left_col, right_col],
        ],
    )


def test_check_no_simultaneous() -> None:
    """Tests the helper function that checks we don't do two things at once."""
    df = pd.DataFrame(
        {
            "battery-electric_charge_mwh": [10, 0, 0, 0],
            "battery-electric_discharge_mwh": [0, 20, 30, 0],
        }
    )
    name = "battery"
    check, errors = check_no_simultaneous(
        df, f"{name}-electric_charge_mwh", f"{name}-electric_discharge_mwh"
    )
    assert check

    df = pd.DataFrame(
        {
            "battery-electric_charge_mwh": [10, 10, 30],
            "battery-electric_discharge_mwh": [0, 20, 30],
        }
    )
    check, errors = check_no_simultaneous(
        df, f"{name}-electric_charge_mwh", f"{name}-electric_discharge_mwh"
    )
    assert not check


@hypothesis.settings(
    print_blob=True,
    max_examples=250,
    verbosity=hypothesis.Verbosity.verbose,
    deadline=20000,
)
@hypothesis.given(
    idx_length=hypothesis.strategies.integers(min_value=10, max_value=24),
    power_mw=hypothesis.strategies.floats(min_value=0.1, max_value=100),
    capacity_mwh=hypothesis.strategies.floats(min_value=0.1, max_value=100),
    initial_charge_mwh=hypothesis.strategies.floats(min_value=0.0, max_value=100),
    efficiency=hypothesis.strategies.floats(min_value=0.1, max_value=1.0),
    prices_mu=hypothesis.strategies.floats(min_value=-1000, max_value=1000),
    prices_std=hypothesis.strategies.floats(min_value=1.0, max_value=1000),
    prices_offset=hypothesis.strategies.floats(min_value=-250, max_value=250),
)
def test_hypothesis(
    idx_length: int,
    power_mw: float,
    capacity_mwh: float,
    efficiency: float,
    initial_charge_mwh: float,
    prices_mu: float,
    prices_std: float,
    prices_offset: float,
) -> None:
    """Test optimization with hypothesis."""
    freq_mins = 30
    electricity_prices = (
        np.random.normal(prices_mu, prices_std, idx_length) + prices_offset
    )

    #  possible to enter into infeasible situations where the final charge is
    #  more than the battery can produce in the intervals
    #  this is a problem with the tests, not the application
    final_charge_mwh = None

    asset = epl.Battery(
        power_mw=power_mw,
        capacity_mwh=capacity_mwh,
        efficiency_pct=efficiency,
        electricity_prices=electricity_prices,
        freq_mins=freq_mins,
        initial_charge_mwh=initial_charge_mwh,
        final_charge_mwh=final_charge_mwh,
    )

    simulation = asset.optimize(
        verbose=False,
        optimizer_config=epl.OptimizerConfig(relative_tolerance=0.01, timeout=18),
    )

    freq = epl.Freq(freq_mins)

    #  check we don't exceed the battery rating
    assert all(
        simulation.results["battery-electric_charge_mwh"]
        <= freq.mw_to_mwh(power_mw) + tol
    )
    assert all(
        simulation.results["battery-electric_discharge_mwh"]
        <= freq.mw_to_mwh(power_mw) + tol
    )

    #  check charge & discharge are always positive
    assert all(simulation.results["battery-electric_charge_mwh"] >= 0 - tol)
    assert all(simulation.results["battery-electric_discharge_mwh"] >= 0 - tol)

    #  check we don't exceed battery capacity
    name = "battery"
    for var in ["electric_initial_charge_mwh", "electric_final_charge_mwh"]:
        assert all(simulation.results[f"{name}-{var}"] <= capacity_mwh + tol)
        assert all(simulation.results[f"{name}-{var}"] >= 0 - tol)

    #  check we set initial and final charge correctly
    np.testing.assert_almost_equal(
        simulation.results[f"{name}-electric_initial_charge_mwh"].iloc[0],
        asset.cfg.initial_charge_mwh,
        decimal=4,
    )
    assert isinstance(asset.cfg.final_charge_mwh, float)
    np.testing.assert_almost_equal(
        simulation.results[f"{name}-electric_final_charge_mwh"].iloc[-1],
        asset.cfg.final_charge_mwh,
        decimal=4,
    )

    #  check losses are a percentage of our charge
    mask = simulation.results[f"{name}-electric_charge_mwh"] > 0
    subset = simulation.results[mask]
    np.testing.assert_almost_equal(
        subset[f"{name}-electric_loss_mwh"].values,
        (1 - efficiency) * subset[f"{name}-electric_charge_mwh"].values,
        decimal=4,
    )

    check, errors = check_no_simultaneous(
        simulation.results,
        f"{name}-electric_charge_mwh",
        f"{name}-electric_discharge_mwh",
    )
    assert check, errors

    #  check losses are always zero when we discharge
    mask = simulation.results[f"{name}-electric_discharge_mwh"] > 0
    subset = simulation.results[mask]
    assert all(subset[f"{name}-electric_loss_mwh"] == 0)
    np.testing.assert_allclose(subset[f"{name}-electric_loss_mwh"], 0)


def test_import_export_prices() -> None:
    """Test the use of export electricity prices in the battery model."""

    # # test that when we have no export value, we never arbitrage
    electricity_prices = np.clip(np.random.normal(100, 1000, 128), a_min=0, a_max=None)
    export_electricity_prices = 0.0

    power_mw = 2
    capacity_mwh = 4
    initial_charge_mwh = 0
    final_charge_mwh = 0

    asset = epl.Battery(
        power_mw=power_mw,
        capacity_mwh=capacity_mwh,
        electricity_prices=electricity_prices,
        export_electricity_prices=export_electricity_prices,
        initial_charge_mwh=initial_charge_mwh,
        final_charge_mwh=final_charge_mwh,
    )
    simulation = asset.optimize(verbose=False)
    assert simulation.results["battery-electric_charge_mwh"].sum() == 0

    # test that as we increase export prices, we use the battery more
    battery_usage = []

    for export_price_delta in range(-30, 60, 10):
        tic = time.perf_counter()
        asset = epl.Battery(
            power_mw=power_mw,
            capacity_mwh=capacity_mwh,
            electricity_prices=electricity_prices,
            export_electricity_prices=electricity_prices + export_price_delta,
            initial_charge_mwh=initial_charge_mwh,
            final_charge_mwh=final_charge_mwh,
        )
        simulation = asset.optimize(
            verbose=False,
            optimizer_config=epl.OptimizerConfig(relative_tolerance=0, timeout=60 * 2),
        )
        battery_usage.append(simulation.results["battery-electric_charge_mwh"].sum())
        print(f"{export_price_delta=}, time={time.perf_counter() - tic} sec")
    assert np.all(np.diff(battery_usage) >= -tol)


def test_no_simultaneous_import_export() -> None:
    """Test that we never import and export at the same time."""
    electricity_prices = np.random.normal(100, 1000, 1024)
    export_electricity_prices = np.random.normal(100, 1000, 1024)
    asset = epl.Battery(
        electricity_prices=electricity_prices,
        export_electricity_prices=export_electricity_prices,
    )
    simulation = asset.optimize()
    results = simulation.results

    check_no_simultaneous(results, "site-import_power_mwh", "site-export_power_mwh")


@hypothesis.settings(
    print_blob=True,
    max_examples=25,
    verbosity=hypothesis.Verbosity.verbose,
    deadline=20000,
)
@hypothesis.given(
    charge_mw=hypothesis.strategies.floats(min_value=1.0, max_value=100),
    discharge_mw=hypothesis.strategies.floats(min_value=1.0, max_value=100),
)
def test_different_charge_discharge_rates(
    charge_mw: float, discharge_mw: float
) -> None:
    """Test that we can charge and discharge our battery at different rates."""
    prices = np.random.uniform(-1000, 1000, 512)
    capacity = 100
    asset = epl.Battery(
        power_mw=charge_mw,
        discharge_power_mw=discharge_mw,
        capacity_mwh=capacity,
        initial_charge_mwh=0,
        final_charge_mwh=capacity,
        efficiency_pct=1.0,
        electricity_prices=prices,
    )
    simulation = asset.optimize()
    np.testing.assert_allclose(
        simulation.results["battery-electric_charge_mwh"].max(), charge_mw
    )
    np.testing.assert_allclose(
        simulation.results["battery-electric_discharge_mwh"].max(), discharge_mw
    )
