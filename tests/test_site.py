"""Test the site API."""
import numpy as np

import energypylinear as epl
from energypylinear.data_generation import generate_random_ev_input_data
from energypylinear.defaults import defaults


def test_site() -> None:
    """Tests the epl.Site() API"""
    site = epl.Site(
        assets=[
            epl.Battery(),
            epl.Battery(name="fast-battery"),
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
    this is a bit of a regression test

    first interval we both charge and generate max electricity
    second interval we discharge and generate
    """
    np.testing.assert_array_almost_equal(
        simulation["site-import_power_mwh"],
        [0, 0, 4.0, 0.44444, 0.0],
        decimal=defaults.decimal_tolerance,
    )
    np.testing.assert_array_almost_equal(
        simulation["site-export_power_mwh"],
        [96.0, 103.6, 0, 0, 4.0],
        decimal=defaults.decimal_tolerance,
    )


def test_sites() -> None:
    """Tests various hardcoded combinations of assets."""

    interval_data = generate_random_ev_input_data(10, n_chargers=3, charge_length=3)
    site = epl.Site(
        assets=[
            epl.Battery(),
            epl.Generator(
                electric_power_max_mw=100,
                electric_efficiency_pct=0.3,
                high_temperature_efficiency_pct=0.5,
            ),
            epl.EVs(charger_mws=interval_data["charger_mws"]),
        ]
    )
    interval_data.pop("charger_mws")
    site.optimize(**interval_data)


# import hypothesis
# import hypothesis.strategies as st

# # Define the strategy for sampling from the input list
# asset_strategy = st.one_of(
#     st.builds(
#         epl.Battery,
#         power_mw=st.floats(0.5, 100),
#         capacity_mwh=st.floats(0.5, 100),
#     ),
#     st.builds(
#         epl.Generator,
#         electric_power_max_mw=st.integers(1, 100),
#         electric_efficiency_pct=st.floats(0, 1),
#         high_temperature_efficiency_pct=st.floats(0, 1),
#     ),
#     st.just(
#         epl.EVs(
#             charger_mws=[5, 10, 15],
#         )
#         # charger_mws=st.lists(
#         #     st.sampled_from([5, 10, 20]),
#         #     min_size=1,
#         #     max_size=3,
#         # ),
#     ),
# )

# # Define the main strategy for lists of assets
# assets_strategy = st.lists(asset_strategy, min_size=1)

# # Use the given decorator with the assets_strategy
# @hypothesis.settings(
#     print_blob=True,
#     max_examples=200,
#     verbosity=hypothesis.Verbosity.verbose,
#     deadline=2000,
# )
# @hypothesis.given(assets=assets_strategy)
# def test_site_hypothesis(assets: list) -> None:
#     site = epl.Site(assets)
#     print(assets)

#     from energypylinear.data_generation import generate_random_ev_input_data

#     interval_data = generate_random_ev_input_data(10, n_chargers=3, charge_length=3)
#     interval_data.pop("charger_mws")
#     site.optimize(**interval_data)
