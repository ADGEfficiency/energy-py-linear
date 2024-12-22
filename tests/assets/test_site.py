"""Test the site API."""
import random

import numpy as np
import pydantic_core
import pytest

import energypylinear as epl
from energypylinear.assets.site import SiteIntervalData, validate_interval_data
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
        electricity_prices=[100, 1000, -20, 40, 45],
    )

    simulation = site.optimize()
    simulation.results.to_csv("temp.csv")
    np.testing.assert_array_almost_equal(
        simulation.results["site-import_power_mwh"],
        [0.0, 0, 3.0, 0.2222, 0.0],
        decimal=defaults.decimal_tolerance,
    )
    np.testing.assert_array_almost_equal(
        simulation.results["site-export_power_mwh"],
        [97.0, 102.6, 0.0, 0.0, 2.8],
        decimal=defaults.decimal_tolerance,
    )
    accounts = epl.get_accounts(simulation.results)
    np.testing.assert_allclose(simulation.status.objective, accounts.profit * -1)


@pytest.mark.parametrize("seed", range(24))
def test_sites(seed: int) -> None:
    """Tests various random combinations of assets."""
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
        epl.HeatPump(cop=2.0, electric_power_mw=5.0),
        epl.HeatPump(name="heat-pump2"),
        epl.RenewableGenerator(
            electric_generation_mwh=np.random.uniform(0, 100, 48), name="wind"
        ),
        epl.RenewableGenerator(
            electric_generation_mwh=np.random.uniform(0, 100, 48), name="solar"
        ),
        epl.Spill(),
    ]

    n_assets = random.randint(len(assets), len(assets))
    sampled_assets = random.sample(assets, n_assets)
    site = epl.Site(
        assets=sampled_assets,
        electricity_prices=ds["electricity_prices"],
        gas_prices=30,
        electric_load_mwh=30,
        high_temperature_load_mwh=100,
        low_temperature_load_mwh=100,
        low_temperature_generation_mwh=100,
    )
    simulation = site.optimize(verbose=True)
    debug_simulation(simulation.results)

    # can only check the objective versus the accounts if we don't have any spill assets
    if not [isinstance(a, epl.Spill) for a in sampled_assets]:
        accounts = epl.get_accounts(simulation.results)
        np.testing.assert_allclose(simulation.status.objective, accounts.profit * -1)


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

    #  test that things fail when the assets have different length data as the site index
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
        low_temperature_generation_mwh=[5, 5],
    )


def test_site_freq_mins() -> None:
    """Tests different freq mins in the site."""

    mins = np.array([5, 15, 30, 60])
    for freq_mins in mins:
        generation = np.full_like(mins, 5)
        load = np.full_like(mins, 10)
        prices = np.full_like(mins, 100)
        assets = [epl.RenewableGenerator(electric_generation_mwh=generation)]
        site = epl.Site(
            assets=assets,
            electric_load_mwh=load,
            electricity_prices=prices,
            freq_mins=freq_mins,
        )
        simulation = site.optimize()
        np.testing.assert_array_equal(
            simulation.results["site-import_power_mwh"], 10 - 5
        )


def test_site_interval_data_export_prices() -> None:
    """Test we validate export electricity prices in the SiteIntervalData."""

    # when we don't specify export prices, they are the same as import prices
    id = SiteIntervalData(
        electricity_prices=[10, 20, 30], export_electricity_prices=None
    )
    assert isinstance(id.export_electricity_prices, np.ndarray)
    np.testing.assert_array_equal(id.export_electricity_prices, [10, 20, 30])

    # when we specify export prices as a constant, it works as expected
    id = SiteIntervalData(electricity_prices=[10, 20, 30], export_electricity_prices=70)
    assert isinstance(id.export_electricity_prices, np.ndarray)
    np.testing.assert_array_equal(id.export_electricity_prices, [70, 70, 70])

    id = SiteIntervalData(electricity_prices=[10, 20, 30], export_electricity_prices=0)
    assert isinstance(id.export_electricity_prices, np.ndarray)
    np.testing.assert_array_equal(id.export_electricity_prices, [0, 0, 0])

    # when we specify export prices as interval data, it works as expected
    id = SiteIntervalData(
        electricity_prices=[10, 20, 30], export_electricity_prices=[30, 20, 10]
    )
    assert isinstance(id.export_electricity_prices, np.ndarray)
    np.testing.assert_array_equal(id.export_electricity_prices, [30, 20, 10])

    # when we specify export prices as interval data, it fails as expected
    with pytest.raises(pydantic_core._pydantic_core.ValidationError):
        id = SiteIntervalData(
            electricity_prices=[10, 20, 30], export_electricity_prices=[30, 20]
        )

    with pytest.raises(pydantic_core._pydantic_core.ValidationError):
        id = SiteIntervalData(
            electricity_prices=[10, 20, 30], export_electricity_prices=[30, 20, 30, 40]
        )


def test_allow_infeasible() -> None:
    """Tests the allow infeasible flag."""
    site = epl.Site(
        assets=[],
        electricity_prices=[100, 1000, -20, 40, 50],
        electric_load_mwh=1000,
        import_limit_mw=0,
    )
    with pytest.raises(AssertionError, match="Infeasible simulation!"):
        site.optimize(flags=epl.Flags(allow_infeasible=False))

    site.optimize(flags=epl.Flags(allow_infeasible=True))


@pytest.mark.parametrize("import_limit_mw", [0, 10, 20, 30])
@pytest.mark.parametrize("export_limit_mw", [0, 5, 10, 15])
def test_import_export_limits(import_limit_mw: float, export_limit_mw: float) -> None:
    """Test the site import & export limits."""
    site = epl.Site(
        assets=[
            epl.Battery(
                power_mw=120, capacity_mwh=120, efficiency_pct=1.0, name="big-battery"
            ),
        ],
        electricity_prices=[-1000, 1000, -1000, 1000, -1000, 1000],
        import_limit_mw=import_limit_mw,
        export_limit_mw=export_limit_mw,
        freq_mins=60,
    )

    simulation = site.optimize()
    assert simulation.feasible
    assert simulation.results["site-import_power_mwh"].max() <= import_limit_mw
    assert simulation.results["site-export_power_mwh"].max() <= export_limit_mw
