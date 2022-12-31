"""these tests don't test the plot contents - only that we can plot & save something"""
import numpy as np
import pytest

import energypylinear as epl


def test_battery_plot(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("figs")
    asset = epl.battery.Battery(power_mw=2, capacity_mwh=4)
    electricity_prices = np.random.normal(100, 10, 10).tolist()
    results = asset.optimize(electricity_prices=electricity_prices)

    assert not (path / "battery.png").exists()
    asset.plot(results, path=path)
    assert (path / "battery.png").exists()
    asset.plot(results, path=path / "battery-custom.png")
    assert (path / "battery-custom.png").exists()


def test_evs_plot(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("figs")
    ds = epl.data_generation.generate_random_ev_input_data(10, 2, 3)
    asset = epl.evs.EVs(
        charger_mws=ds["charger_mws"].tolist(),
    )
    ds.pop("charger_mws")
    results = asset.optimize(**ds, return_interval_data=True)

    assert not (path / "evs.png").exists()
    asset.plot(*results, path=path)
    assert (path / "evs.png").exists()
    asset.plot(*results, path=path / "evs-custom.png")
    assert (path / "evs-custom.png").exists()
