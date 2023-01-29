"""these tests don't test the plot contents - only that we can plot & save something"""
import numpy as np
import pytest

import energypylinear as epl


def test_battery_plot(tmp_path_factory: pytest.TempPathFactory) -> None:
    path = tmp_path_factory.mktemp("figs")
    asset = epl.battery.Battery(power_mw=2, capacity_mwh=4)
    electricity_prices = np.random.normal(100, 10, 10).tolist()
    results = asset.optimize(electricity_prices=electricity_prices)

    assert not (path / "battery.png").exists()
    asset.plot(results, path=path)
    assert (path / "battery.png").exists()
    asset.plot(results, path=path / "battery-custom.png")
    assert (path / "battery-custom.png").exists()


def test_evs_plot(tmp_path_factory: pytest.TempPathFactory) -> None:
    path = tmp_path_factory.mktemp("figs")
    ds = epl.data_generation.generate_random_ev_input_data(10, 2, 3)
    asset = epl.evs.EVs(
        charger_mws=ds["charger_mws"].tolist(),
    )
    ds.pop("charger_mws")
    results = asset.optimize(
        **ds,
    )
    assert not (path / "evs.png").exists()
    asset.plot(results, path=path)
    assert (path / "evs.png").exists()
    asset.plot(results, path=path / "evs-custom.png")
    assert (path / "evs-custom.png").exists()


def test_chp_plot(tmp_path_factory: pytest.TempPathFactory) -> None:
    path = tmp_path_factory.mktemp("figs")

    prices = np.random.uniform(-1000, 1000, 24).tolist()
    ht_load = np.random.uniform(0, 100, 24).tolist()
    lt_load = np.random.uniform(0, 100, 24).tolist()

    asset = epl.chp.Generator(
        electric_power_max_mw=100,
        electric_power_min_mw=50,
        electric_efficiency_pct=0.3,
        high_temperature_efficiency_pct=0.5,
    )
    results = asset.optimize(
        electricity_prices=prices,
        gas_prices=20,
        high_temperature_load_mwh=ht_load,
        low_temperature_load_mwh=lt_load,
        freq_mins=60,
    )

    assert not (path / "chp.png").exists()
    asset.plot(results, path=path)
    assert (path / "chp.png").exists()
    asset.plot(results, path=path / "chp-custom.png")
    assert (path / "chp-custom.png").exists()
