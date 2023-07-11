"""Test the site API."""
import random

import numpy as np

import energypylinear as epl
from energypylinear.data_generation import generate_random_ev_input_data
from energypylinear.defaults import defaults


def test_site() -> None:
    """Tests the epl.Site() API"""
    site = epl.Site(
        assets=[
            epl.Battery(
                power_mw=2, capacity_mwh=4, efficiency=0.9, name="small-battery"
            ),
            epl.Battery(power_mw=8, capacity_mwh=1, efficiency=0.8, name="big-battery"),
            epl.Generator(
                electric_power_max_mw=50,
                electric_efficiency_pct=0.3,
                high_temperature_efficiency_pct=0.5,
            ),
            epl.Generator(
                electric_power_max_mw=50,
                electric_efficiency_pct=0.4,
                high_temperature_efficiency_pct=0.4,
                name="gas-engine-generator",
            ),
            epl.Boiler(high_temperature_generation_max_mw=100),
        ]
    )

    results = site.optimize(electricity_prices=[100, 1000, -20, 40, 50])
    print(results.simulation)
    print(results.simulation.columns)
    simulation = results.simulation

    """
    first interval we both charge and generate max electricity
    second interval we discharge and generate
    """
    np.testing.assert_array_almost_equal(
        simulation["site-import_power_mwh"],
        [0, 0, 4.6, 0.2222, 0.0],
        decimal=defaults.decimal_tolerance,
    )
    np.testing.assert_array_almost_equal(
        simulation["site-export_power_mwh"],
        ([100.0, 100.0, 0.0, 0.0, 53.0]),
        decimal=defaults.decimal_tolerance,
    )


def test_sites() -> None:
    """Tests various hardcoded combinations of assets."""
    for _ in range(10):
        ds = generate_random_ev_input_data(10, n_chargers=3, charge_length=3)
        assets = [
            epl.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9),
            epl.Battery(power_mw=8, capacity_mwh=1, efficiency=0.8, name="battery2"),
            epl.Generator(
                electric_power_max_mw=100,
                electric_efficiency_pct=0.3,
                high_temperature_efficiency_pct=0.5,
                name="generator1",
            ),
            epl.Generator(
                electric_power_max_mw=50,
                electric_efficiency_pct=0.4,
                high_temperature_efficiency_pct=0.4,
                name="generator2",
            ),
            epl.EVs(
                chargers_power_mw=ds["charger_mws"],
                charge_events_capacity_mwh=ds["charge_events_capacity_mwh"].tolist(),
                charge_events=ds["charge_events"],
                charge_event_efficiency=0.8,
                charger_turndown=0.0,
                name="evs1",
            ),
            epl.EVs(
                chargers_power_mw=ds["charger_mws"],
                charge_events_capacity_mwh=ds["charge_events_capacity_mwh"].tolist(),
                charge_events=ds["charge_events"],
                charge_event_efficiency=1.0,
                charger_turndown=0.4,
                name="evs2",
            ),
        ]

        ds.pop("charger_mws")
        ds.pop("charge_events_capacity_mwh")
        n_assets = random.randint(len(assets), len(assets))

        sampled_assets = random.sample(assets, n_assets)
        print(len(sampled_assets))
        site = epl.Site(assets=sampled_assets)
        site.optimize(**ds, verbose=True)
