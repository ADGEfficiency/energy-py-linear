"""Tests for `epl.plots`.

These tests don't test the plot contents - only that we can plot & save something.
"""
import numpy as np
import pytest

import energypylinear as epl
from energypylinear.flags import Flags


def test_battery_plot(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Test we can plot the battery chart."""
    path = tmp_path_factory.mktemp("figs")
    asset = epl.Battery(
        power_mw=2, capacity_mwh=4, electricity_prices=np.random.normal(100, 10, 10)
    )
    simulation = asset.optimize()

    assert not (path / "battery.png").exists()
    asset.plot(simulation, path=path)
    assert (path / "battery.png").exists()
    asset.plot(simulation, path=path / "battery-custom.png")
    assert (path / "battery-custom.png").exists()


def test_evs_plot(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Test we can plot the EVs chart."""
    path = tmp_path_factory.mktemp("figs")
    ds = epl.data_generation.generate_random_ev_input_data(10, 5, 3)
    asset = epl.EVs(
        **ds,
        charge_event_efficiency=1.0,
        charger_turndown=0.0,
    )
    results = asset.optimize(
        flags=Flags(
            allow_evs_discharge=False,
            fail_on_spill_asset_use=True,
            allow_infeasible=False,
        ),
    )
    assert not (path / "evs.png").exists()
    asset.plot(results, path=path)
    assert (path / "evs.png").exists()
    asset.plot(results, path=path / "evs-custom.png")
    assert (path / "evs-custom.png").exists()


def test_chp_plot(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Test we can plot the CHP chart."""
    path = tmp_path_factory.mktemp("figs")

    prices = np.random.uniform(-1000, 1000, 24).tolist()
    ht_load = np.random.uniform(0, 100, 24).tolist()
    lt_load = np.random.uniform(0, 100, 24).tolist()

    asset = epl.CHP(
        electric_power_max_mw=100,
        electric_power_min_mw=50,
        electric_efficiency_pct=0.3,
        high_temperature_efficiency_pct=0.5,
        electricity_prices=prices,
        gas_prices=20,
        high_temperature_load_mwh=ht_load,
        low_temperature_load_mwh=lt_load,
        freq_mins=60,
    )
    results = asset.optimize()

    assert not (path / "chp.png").exists()
    asset.plot(results, path=path)
    assert (path / "chp.png").exists()
    asset.plot(results, path=path / "chp-custom.png")
    assert (path / "chp-custom.png").exists()


def test_heat_pump_plot(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Test we can plot the CHP chart."""
    path = tmp_path_factory.mktemp("figs")

    prices = np.random.uniform(-1000, 1000, 24).tolist()
    ht_load = np.random.uniform(0, 100, 24).tolist()
    lt_load = np.random.uniform(0, 100, 24).tolist()
    lt_gen = np.random.uniform(0, 100, 24).tolist()

    asset = epl.HeatPump(
        10,
        2.0,
        electricity_prices=prices,
        gas_prices=20,
        high_temperature_load_mwh=ht_load,
        low_temperature_load_mwh=lt_load,
        low_temperature_generation_mwh=lt_gen,
        freq_mins=60,
    )
    results = asset.optimize()

    assert not (path / "heat-pump.png").exists()
    asset.plot(results, path=path)
    assert (path / "heat-pump.png").exists()
