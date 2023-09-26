"""Test the site API."""
import random

import numpy as np
import pytest

import energypylinear as epl
from energypylinear.assets.site import validate_interval_data
from energypylinear.data_generation import generate_random_ev_input_data
from energypylinear.debug import debug_simulation
from energypylinear.defaults import defaults


def test_site() -> None:
    """Tests the epl.Site() API"""
    site = epl.Site(
        assets=[
            epl.Battery(
                power_mw=2, capacity_mwh=4, efficiency_pct=0.9, name="small-battery"
            ),
            epl.Battery(
                power_mw=8, capacity_mwh=1, efficiency_pct=0.8, name="big-battery"
            ),
            epl.CHP(
                electric_power_max_mw=50,
                electric_efficiency_pct=0.3,
                high_temperature_efficiency_pct=0.5,
            ),
            epl.CHP(
                electric_power_max_mw=50,
                electric_efficiency_pct=0.4,
                high_temperature_efficiency_pct=0.4,
                name="gas-engine-chp",
            ),
            epl.Boiler(high_temperature_generation_max_mw=100),
            epl.Spill(),
            epl.Valve(),
        ],
        electricity_prices=[100, 1000, -20, 40, 50],
    )

    simulation = site.optimize()

    """
    first interval we both charge and generate max electricity
    second interval we discharge and generate
    """
    np.testing.assert_array_almost_equal(
        simulation.results["site-import_power_mwh"],
        [0.0, 0, 4.6, 0.2222, 0.0],
        decimal=defaults.decimal_tolerance,
    )
    np.testing.assert_array_almost_equal(
        simulation.results["site-export_power_mwh"],
        [96.75, 102.8, 0.0, 0.0, 53.0],
        decimal=defaults.decimal_tolerance,
    )


@pytest.mark.parametrize("seed", range(10))
def test_sites(seed: int) -> None:
    """Tests various hardcoded combinations of assets."""
    ds = generate_random_ev_input_data(48, n_chargers=3, charge_length=3, seed=seed)
    assets = [
        epl.Battery(
            power_mw=2,
            capacity_mwh=4,
            efficiency_pct=0.9,
        ),
        epl.Battery(
            power_mw=8,
            capacity_mwh=1,
            efficiency_pct=0.8,
            name="battery2",
        ),
        epl.CHP(
            electric_power_max_mw=100,
            electric_efficiency_pct=0.3,
            high_temperature_efficiency_pct=0.5,
            name="chp1",
        ),
        epl.CHP(
            electric_power_max_mw=50,
            electric_efficiency_pct=0.4,
            high_temperature_efficiency_pct=0.4,
            name="chp2",
        ),
        epl.EVs(
            **ds,
            charge_event_efficiency=0.8,
            charger_turndown=0.0,
            name="evs1",
        ),
        epl.EVs(
            **ds,
            charge_event_efficiency=1.0,
            charger_turndown=0.4,
            name="evs2",
        ),
        epl.HeatPump(name="heat-pump1"),
        epl.HeatPump(name="heat-pump2"),
        epl.RenewableGenerator(
            electric_generation_mwh=np.random.uniform(0, 100, 48), name="wind"
        ),
        epl.RenewableGenerator(
            electric_generation_mwh=np.random.uniform(0, 100, 48), name="solar"
        ),
    ]

    n_assets = random.randint(len(assets), len(assets))
    sampled_assets = random.sample(assets, n_assets)
    site = epl.Site(assets=sampled_assets, electricity_prices=ds["electricity_prices"])
    simulation = site.optimize(verbose=True)
    debug_simulation(simulation.results)


def test_interval_data() -> None:
    """Test site interval data."""

    #  test we use electricity_prices to create the index
    electricity_prices = [1.0, 2.0]
    idata = epl.assets.site.SiteIntervalData(electricity_prices=electricity_prices)
    assert len(idata.idx) == 2

    #  test we use electricity_carbon_intensities to create the index
    electricity_prices = [1.0, 2.0]
    electricity_carbon_intensities = [1.0, 2.0, 3.0]
    idata = epl.assets.site.SiteIntervalData(
        electricity_carbon_intensities=electricity_carbon_intensities
    )
    assert len(idata.idx) == 3

    #  test that things work correctly when the assets have the same length data as the site index
    site = epl.Site(assets=[], electricity_carbon_intensities=[1.0, 2.0])
    assets = [epl.RenewableGenerator(electric_generation_mwh=[1.0, 2.0])]
    validate_interval_data(assets, site)

    #  test that things work correctly when the assets have different length data as the site index
    with pytest.raises(AssertionError):
        site = epl.Site(assets=[], electricity_carbon_intensities=[1.0, 2.0])
        assets = [epl.RenewableGenerator(electric_generation_mwh=2.0)]
        validate_interval_data(assets, site, repeat_interval_data=False)

    #  test that things work correctly when the assets have different length data as the site index
    site = epl.Site(assets=[], electricity_carbon_intensities=[1.0, 2.0])
    assets = [epl.RenewableGenerator(electric_generation_mwh=2.0, name="solar")]
    validate_interval_data(assets, site, repeat_interval_data=True)

    asset = assets[0]
    assert asset.cfg.name == "solar"
    assert isinstance(asset.cfg.interval_data.electric_generation_mwh, np.ndarray)
    assert all(asset.cfg.interval_data.electric_generation_mwh == [2.0, 2.0])

    #  test that we fail when we don't supply electricity prices or carbon
    with pytest.raises(ValueError):
        epl.assets.site.SiteIntervalData()

    #  test the validation routes
    epl.assets.site.SiteIntervalData(
        electricity_prices=[10, 10],
        high_temperature_load_mwh=5.0,
        high_temperature_generation_mwh=[5, 5],
    )
