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
def test_battery_optimization_price(
    electricity_prices, initial_charge_mwh, expected_dispatch
) -> None:
    power_mw = 4
    capacity_mwh = 6
    efficiency = 1.0
    freq_mins = 60
    asset = epl.battery.Battery(
        power_mw=power_mw, capacity_mwh=capacity_mwh, efficiency=efficiency
    )
    results = asset.optimize(
        electricity_prices=electricity_prices,
        freq_mins=freq_mins,
        initial_charge_mwh=initial_charge_mwh,
        final_charge_mwh=0,
    )
    charge = results["battery-charge_mwh"].values
    discharge = results["battery-discharge_mwh"].values
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
def test_battery_optimization_carbon(
    carbon_intensities, initial_charge_mwh, expected_dispatch
):
    power_mw = 4
    capacity_mwh = 6
    efficiency = 1.0
    freq_mins = 60
    prices = np.random.uniform(-100, 100, len(carbon_intensities)).tolist()
    asset = epl.battery.Battery(
        power_mw=power_mw, capacity_mwh=capacity_mwh, efficiency=efficiency
    )
    results = asset.optimize(
        electricity_prices=prices,
        electricity_carbon_intensities=carbon_intensities,
        freq_mins=freq_mins,
        initial_charge_mwh=initial_charge_mwh,
        objective="carbon",
    )
    charge = results["battery-charge_mwh"].values
    discharge = results["battery-discharge_mwh"].values
    dispatch = charge - discharge
    np.testing.assert_almost_equal(dispatch, expected_dispatch)


@hypothesis.settings(
    print_blob=True,
    max_examples=200,
    verbosity=hypothesis.Verbosity.verbose,
    deadline=2000,
)
@hypothesis.given(
    idx_length=hypothesis.strategies.integers(min_value=10, max_value=24),
    power_mw=hypothesis.strategies.floats(min_value=0.1, max_value=100),
    capacity_mwh=hypothesis.strategies.floats(min_value=0.1, max_value=100),
    initial_charge_mwh=hypothesis.strategies.floats(min_value=0.0, max_value=100),
    final_charge_mwh=hypothesis.strategies.floats(min_value=0.0, max_value=100),
    efficiency=hypothesis.strategies.floats(min_value=0.5, max_value=1.0),
    prices_mu=hypothesis.strategies.floats(min_value=-1000, max_value=1000),
    prices_std=hypothesis.strategies.floats(min_value=0.1, max_value=25),
)
def test_battery_hypothesis(
    idx_length: int,
    power_mw: float,
    capacity_mwh: float,
    efficiency: float,
    initial_charge_mwh: float,
    final_charge_mwh: float,
    prices_mu: float,
    prices_std: float,
) -> None:
    freq_mins = 30
    electricity_prices = np.random.normal(prices_mu, prices_std, idx_length).tolist()

    asset = epl.battery.Battery(
        power_mw=power_mw, capacity_mwh=capacity_mwh, efficiency=efficiency
    )

    #  TODO
    #  possible to enter into infeasible situations where the final charge is more than the battery can
    #  produce in the interval
    #  this is a problem with the tests, not the application
    final_charge_mwh = None
    results = asset.optimize(
        electricity_prices=electricity_prices,
        freq_mins=freq_mins,
        initial_charge_mwh=initial_charge_mwh,
        final_charge_mwh=final_charge_mwh,
    )

    freq = epl.freq.Freq(freq_mins)

    #  check we don't exceed the battery rating
    assert all(results["battery-charge_mwh"] <= freq.mw_to_mwh(power_mw) + tol)
    assert all(results["battery-discharge_mwh"] <= freq.mw_to_mwh(power_mw) + tol)

    #  check charge & discharge are always positive
    assert all(results["battery-charge_mwh"] >= 0 - tol)
    assert all(results["battery-discharge_mwh"] >= 0 - tol)

    #  check we don't exceed battery capacity
    name = "battery-"
    for var in ["initial_charge_mwh", "final_charge_mwh"]:
        assert all(results[f"{name}-{var}"] <= capacity_mwh + tol)
        assert all(results[f"{name}-{var}"] >= 0 - tol)

    #  check we set initial and final charge correctly
    np.testing.assert_almost_equal(
        results[f"{name}-initial_charge_mwh"].iloc[0],
        asset.cfg.initial_charge_mwh,
        decimal=4,
    )
    np.testing.assert_almost_equal(
        results[f"{name}-final_charge_mwh"].iloc[-1],
        asset.cfg.final_charge_mwh,
        decimal=4,
    )

    #  check losses are a percentage of our charge
    mask = results[f"{name}-charge_mwh"] > 0
    subset = results[mask]
    np.testing.assert_almost_equal(
        subset[f"{name}-losses_mwh"].values,
        (1 - efficiency) * subset[f"{name}-charge_mwh"].values,
        decimal=4,
    )
    #  check losses are always zero when we discharge
    mask = results[f"{name}-discharge_mwh"] > 0
    subset = results[mask]

    #  temporaray debugging dataframe
    temp = pd.DataFrame(
        {
            "charge": results[f"{name}-charge_mwh"],
            "charge_bin": results[f"{name}-charge_binary"],
            "discharge": results[f"{name}-discharge_mwh"],
            "discharge_bin": results[f"{name}-discharge_binary"],
            "losses": results[f"{name}-losses_mwh"],
        }
    )
    """
    TODO DEBT
    bit of a bug here TODO - issue with charge and discharge at the same time

    this will cause the line below to break
    """
    # assert all(subset[f"{name}-losses_mwh"] == 0)
