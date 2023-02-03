import numpy as np
from _pytest.capture import CaptureFixture

import energypylinear as epl
from energypylinear.defaults import defaults


def test_spill_validation():
    with pytest.raises(AssertionError as err:
        sc = epl.assets.spill.SiteConfig(name='not-valid')


def test_chp_gas_turbine_price(capsys: CaptureFixture) -> None:
    asset = epl.chp.Generator(
        electric_power_max_mw=100,
        electric_power_min_mw=50,
        electric_efficiency_pct=0.3,
        high_temperature_efficiency_pct=0.5,
    )
    """
    - high electricity price, low heat demand
    - expect generator to run full load and dump heat to low temperature
    """
    results = asset.optimize(
        electricity_prices=[
            1000,
        ],
        gas_prices=20,
        high_temperature_load_mwh=[
            20,
        ],
        freq_mins=60,
    )
    results = results.simulation
    #  https://docs.pytest.org/en/7.1.x/how-to/capture-stdout-stderr.html
    capture = capsys.readouterr()
    assert "Spill Occurred" in capture.out
    row = results.iloc[0, :]
    assert row["generator-electric_generation_mwh"] == 100

    """
    - low electricity price, low heat demand
    - expect all heat demand met from boiler
    """
    results = asset.optimize(
        electricity_prices=[-100],
        gas_prices=20,
        high_temperature_load_mwh=[20],
        freq_mins=60,
    )
    results = results.simulation
    #  https://docs.pytest.org/en/7.1.x/how-to/capture-stdout-stderr.html
    capture = capsys.readouterr()
    assert "Spill Occurred" not in capture.out
