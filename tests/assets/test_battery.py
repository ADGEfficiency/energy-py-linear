"""Test Battery asset."""

import hypothesis
import numpy as np
import pytest

import energypylinear as epl
from energypylinear.flags import Flags

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
def test_battery_price(
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
def test_battery_carbon(
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



def test_battery_charge_discharge_binary_variables() -> None:
    """Test optimization with hypothesis."""
    electricity_prices = (
        np.random.normal(100, 1000, 1024)
    )
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
    simulation = asset.optimize(
        flags=Flags(include_charge_discharge_binary_variables=True)
    )

    name = asset.cfg.name

    #  check losses are a percentage of our charge
    mask = simulation.results[f"{name}-electric_charge_mwh"] > 0
    subset = simulation.results[mask]
    np.testing.assert_almost_equal(
        subset[f"{name}-electric_loss_mwh"].values,
        (1 - asset.cfg.efficiency_pct) * subset[f"{name}-electric_charge_mwh"].values,
        decimal=4,
    )

    #  check losses are always zero when we discharge
    mask = simulation.results[f"{name}-electric_discharge_mwh"] > 0
    subset = simulation.results[mask]
    assert all(subset[f"{name}-electric_loss_mwh"] == 0)


@hypothesis.settings(
    print_blob=True,
    max_examples=100,
    verbosity=hypothesis.Verbosity.verbose,
    deadline=2000,
)
@hypothesis.given(
    idx_length=hypothesis.strategies.integers(min_value=10, max_value=24),
    power_mw=hypothesis.strategies.floats(min_value=0.1, max_value=100),
    capacity_mwh=hypothesis.strategies.floats(min_value=0.1, max_value=100),
    initial_charge_mwh=hypothesis.strategies.floats(min_value=0.0, max_value=100),
    efficiency=hypothesis.strategies.floats(min_value=0.1, max_value=1.0),
    prices_mu=hypothesis.strategies.floats(min_value=-1000, max_value=1000),
    prices_std=hypothesis.strategies.floats(min_value=0.1, max_value=1000),
    prices_offset=hypothesis.strategies.floats(min_value=-250, max_value=250),
)
def test_battery_hypothesis(
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

    #  this flag is needed for the check that losses are zero
    #  when we discharge
    flags = Flags(include_charge_discharge_binary_variables=True)
    simulation = asset.optimize(flags=flags)

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
    for var in ["initial_charge_mwh", "final_charge_mwh"]:
        assert all(simulation.results[f"{name}-{var}"] <= capacity_mwh + tol)
        assert all(simulation.results[f"{name}-{var}"] >= 0 - tol)

    #  check we set initial and final charge correctly
    np.testing.assert_almost_equal(
        simulation.results[f"{name}-initial_charge_mwh"].iloc[0],
        asset.cfg.initial_charge_mwh,
        decimal=4,
    )
    np.testing.assert_almost_equal(
        simulation.results[f"{name}-final_charge_mwh"].iloc[-1],
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

    #  check losses are always zero when we discharge
    #  this requires flag `include_charge_discharge_binary_variables`
    #  TODO - separate out this check into a separate test
    #  to keep the binary variables out of this longer test
    mask = simulation.results[f"{name}-electric_discharge_mwh"] > 0
    subset = simulation.results[mask]
    assert all(subset[f"{name}-electric_loss_mwh"] == 0)
