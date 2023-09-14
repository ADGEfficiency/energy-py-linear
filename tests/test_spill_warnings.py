"""Tests for the spill warnings."""

import pydantic
import pytest
from _pytest.capture import CaptureFixture

import energypylinear as epl
from energypylinear.flags import Flags


def test_spill_validation() -> None:
    """Check we fail for incorrectly named spill asset."""
    with pytest.raises(pydantic.ValidationError):
        epl.assets.spill.SpillConfig(name="not-valid")
    epl.assets.spill.SpillConfig(name="valid-spill")


def test_chp_spill(capsys: CaptureFixture) -> None:
    """Check spill with gas turbine."""
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
        gas_prices=20.0,
        high_temperature_load_mwh=[
            20,
        ],
        freq_mins=60,
    )
    simulation = results.simulation
    capture = capsys.readouterr()
    assert "warn_spills" in capture.out

    # now check we fail when we want to
    with pytest.raises(ValueError):
        flags = Flags(fail_on_spill_asset_use=True)
        results = asset.optimize(
            electricity_prices=[
                1000,
            ],
            gas_prices=20,
            high_temperature_load_mwh=[
                20,
            ],
            freq_mins=60,
            flags=flags,
        )

    row = simulation.iloc[0, :]
    assert row["generator-electric_generation_mwh"] == 100

    """
    - low electricity price, low heat demand
    - expect all heat demand met from boiler
    """
    asset.optimize(
        electricity_prices=[-100],
        gas_prices=20,
        high_temperature_load_mwh=[20],
        freq_mins=60,
    )
    capture = capsys.readouterr()
    assert "warn_spills" not in capture.out


def test_evs_spill() -> None:
    """Test EV spills."""

    asset = epl.EVs(
        chargers_power_mw=[100],
        charge_events_capacity_mwh=[50, 100, 30],
        charger_turndown=0.0,
        charge_event_efficiency=1.0,
    )
    results = asset.optimize(
        electricity_prices=[-100, 50],
        charge_events=[
            [1, 0],
            [1, 0],
            [1, 0],
        ],
    )
    assert results.simulation["total-spills_mwh"].sum() > 0
