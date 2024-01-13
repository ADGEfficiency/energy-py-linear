"""Tests the use of FunctionTerms in a custom objective function."""
import hypothesis
import numpy as np

import energypylinear as epl

settings = hypothesis.settings(
    print_blob=True,
    max_examples=1000,
    report_multiple_bugs=True,
)


def test_min_two_variables_export_threshold() -> None:
    """Test the use of the minimum of two variables function term.

    At an incentive for the site to export at the minimum of a fixed value.

    Exporting at less than this value only makes sense when the electricity price is negative enough.

    """
    electricity_prices = np.array([-1000, -750, -250, -100, 0, 10, 100, 1000])
    export_threshold_mwh = 5
    electric_load_mwh = 0
    gas_prices = 20
    export_charge = -500

    electric_efficiency = 0.5
    chp_size = 50

    assets = [
        epl.CHP(
            electric_efficiency_pct=electric_efficiency,
            electric_power_max_mw=chp_size,
        )
    ]

    site = epl.Site(
        assets=assets,
        gas_prices=gas_prices,
        electricity_prices=electricity_prices,
        electric_load_mwh=electric_load_mwh,
    )
    terms = [
        {
            "asset_type": "site",
            "variable": "export_power_mwh",
            "interval_data": "electricity_prices",
            "coefficient": -1,
        },
        {
            "asset_type": "*",
            "variable": "gas_consumption_mwh",
            "interval_data": "gas_prices",
        },
        {
            "type": "complex",
            "function": "min_two_variables",
            "a": {
                "asset_type": "site",
                "variable": "export_power_mwh",
            },
            "b": export_threshold_mwh,
            "coefficient": export_charge,
            "M": (
                electric_load_mwh
                + assets[0].cfg.electric_power_max_mw
                + export_threshold_mwh
            )
            * 1,
        },
    ]

    simulation = site.optimize(
        verbose=4,
        objective={"terms": terms},
    )
    print(simulation.results[["site-export_power_mwh"]])

    electricity_cost = gas_prices / electric_efficiency
    electricity_profit = electricity_prices - electricity_cost
    full_export = electricity_profit > 0
    minimum_export = electricity_profit - export_charge > 0

    """
    expect three modes

    1. no export - when prices are very negative, and the minimum export incentive outweighs the revenue from exporting
    2. minimum export - when the negative prices do not outweighh the minimum export incentive
    3. full export - when prices are high
    """
    if sum(full_export) > 0:
        np.testing.assert_allclose(
            simulation.results["site-export_power_mwh"][full_export], chp_size
        )

    if sum(minimum_export) > 0:
        np.testing.assert_allclose(
            simulation.results["site-export_power_mwh"][minimum_export & ~full_export],
            export_threshold_mwh,
        )
    if sum(~minimum_export) > 0:
        np.testing.assert_allclose(
            simulation.results["site-export_power_mwh"][~minimum_export & ~full_export],
            0.0,
        )


@hypothesis.settings(settings, deadline=1000)
@hypothesis.given(
    import_charge=hypothesis.strategies.floats(
        min_value=0, max_value=1000, allow_nan=False, allow_infinity=False
    ),
    electric_efficiency=hypothesis.strategies.floats(
        min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
@hypothesis.example(import_charge=1000, electric_efficiency=0.5)
def test_maximum_two_variables_export_tariff(
    import_charge: float,
    electric_efficiency: float,
) -> None:
    """Test the use of the maximum of two variables function term.

    A site where we have a charge for importing, based on the maximum of site import and a fixed value."""

    electric_load_mwh = 10
    import_threshold_mwh = 5
    gas_prices = 20
    assets = [
        epl.CHP(
            electric_efficiency_pct=electric_efficiency,
            electric_power_max_mw=10,
        )
    ]
    site = epl.Site(
        assets=assets,
        gas_prices=gas_prices,
        electricity_prices=[0.0],
        electric_load_mwh=electric_load_mwh,
    )
    terms = [
        {
            "asset_type": "site",
            "variable": "import_power_mwh",
            "interval_data": "electricity_prices",
        },
        {
            "asset_type": "site",
            "variable": "export_power_mwh",
            "interval_data": "electricity_prices",
            "coefficient": -1,
        },
        {
            "asset_type": "*",
            "variable": "gas_consumption_mwh",
            "interval_data": "gas_prices",
        },
        {
            "type": "complex",
            "function": "max_two_variables",
            "a": {
                "asset_type": "site",
                "variable": "import_power_mwh",
            },
            "b": import_threshold_mwh,
            "coefficient": import_charge,
            "M": (
                electric_load_mwh
                + assets[0].cfg.electric_power_max_mw
                + import_threshold_mwh
            )
            * 10,
        },
    ]
    simulation = site.optimize(
        verbose=4,
        objective={"terms": terms},
    )

    cost_to_generate_on_site = gas_prices / electric_efficiency

    if import_charge > cost_to_generate_on_site:
        expected_import_mwh = 5.0
    else:
        expected_import_mwh = 10.0

    actual_import = simulation.results["site-import_power_mwh"].iloc[0]
    print(
        f"{import_charge=} {cost_to_generate_on_site=} {expected_import_mwh=} {actual_import=}"
    )
    print(simulation.results[["chp-electric_generation_mwh", "site-import_power_mwh"]])

    np.testing.assert_allclose(
        simulation.results["site-import_power_mwh"].iloc[0], expected_import_mwh
    )
    print("")


def test_minimum_multiple_generators() -> None:
    """Test that the sum of multiple generators is above a certain value.

    Can't be done currently - TODO

    Would require summing all the asset types with one of the two_variables constraints

    interesting alternative to broadcasting...

    probably move this back into todo.md
    """
    pass


def test_maximum_import_all_intervals_no_assets() -> None:
    """Test that we can add a charge to the maximum import over all intervals with no assets.
    todo
    - run a simulation with no max import charge, no assets
    - run a simulation with a max import charge, no assets
    - test that the difference in the objective is the import charge only
    """

    electric_load_mwh = [10.0, 50, 10]
    electricity_prices = [10.0, 10, 10]
    gas_prices = 20
    site = epl.Site(
        assets=[],
        gas_prices=gas_prices,
        electricity_prices=electricity_prices,
        electric_load_mwh=electric_load_mwh,
    )
    no_charge_terms = [
        {
            "asset_type": "site",
            "variable": "import_power_mwh",
            "interval_data": "electricity_prices",
        },
        {
            "asset_type": "site",
            "variable": "export_power_mwh",
            "interval_data": "electricity_prices",
            "coefficient": -1,
        },
        {
            "asset_type": "*",
            "variable": "gas_consumption_mwh",
            "interval_data": "gas_prices",
        },
    ]
    simulation = site.optimize(
        verbose=0,
        objective={"terms": no_charge_terms},
    )
    no_charge_obj = simulation.status.objective

    network_charge = 200
    for minimum_network_demand in [20, max(electric_load_mwh) + 10]:
        charge_terms = [
            {
                "asset_type": "site",
                "variable": "import_power_mwh",
                "interval_data": "electricity_prices",
            },
            {
                "asset_type": "site",
                "variable": "export_power_mwh",
                "interval_data": "electricity_prices",
                "coefficient": -1,
            },
            {
                "asset_type": "*",
                "variable": "gas_consumption_mwh",
                "interval_data": "gas_prices",
            },
            {
                "function": "max_many_variables",
                "variables": {
                    "asset_type": "site",
                    "variable": "import_power_mwh",
                },
                "constant": minimum_network_demand,
                "coefficient": network_charge,
                "M": 1000,
            },
        ]
        simulation = site.optimize(
            verbose=0,
            objective={"terms": charge_terms},
        )
        charge_obj = simulation.status.objective
        assert (
            charge_obj - no_charge_obj
            == max(
                electric_load_mwh
                + [
                    minimum_network_demand,
                ]
            )
            * network_charge
        )


def test_maximum_import_all_intervals_chp() -> None:
    """Test that we can add a charge to the maximum import over all intervals with a CHP asset."""
    # run a simulation with no max import charge
    # with a CHP that never runs - zero value electricity
    electric_load_mwh = [10.0, 50, 10]
    electricity_prices = [0.0, 0.0, 0.0]
    gas_prices = 20

    chp_size = 10
    site = epl.Site(
        assets=[
            epl.CHP(
                electric_efficiency_pct=1.0,
                electric_power_max_mw=chp_size,
                electric_power_min_mw=chp_size,
            )
        ],
        gas_prices=gas_prices,
        electricity_prices=electricity_prices,
        electric_load_mwh=electric_load_mwh,
    )
    no_charge_terms = [
        {
            "asset_type": "site",
            "variable": "import_power_mwh",
            "interval_data": "electricity_prices",
        },
        {
            "asset_type": "site",
            "variable": "export_power_mwh",
            "interval_data": "electricity_prices",
            "coefficient": -1,
        },
        {
            "asset_type": "*",
            "variable": "gas_consumption_mwh",
            "interval_data": "gas_prices",
        },
    ]
    simulation = site.optimize(
        verbose=0,
        objective={"terms": no_charge_terms},
    )
    no_charge_obj = simulation.status.objective
    np.testing.assert_allclose(
        simulation.results["chp-electric_generation_mwh"].sum(), 0.0
    )

    # run a simulation with a max import charge, with a CHP that will run
    network_charge = 200
    minimum_network_demand = 40.0
    charge_terms = [
        {
            "asset_type": "site",
            "variable": "import_power_mwh",
            "interval_data": "electricity_prices",
        },
        {
            "asset_type": "site",
            "variable": "export_power_mwh",
            "interval_data": "electricity_prices",
            "coefficient": -1,
        },
        {
            "asset_type": "*",
            "variable": "gas_consumption_mwh",
            "interval_data": "gas_prices",
        },
        {
            "function": "max_many_variables",
            "variables": {
                "asset_type": "site",
                "variable": "import_power_mwh",
            },
            "constant": minimum_network_demand,
            "coefficient": network_charge,
            "M": 1000,
        },
    ]
    simulation = site.optimize(
        verbose=0,
        objective={"terms": charge_terms},
    )
    charge_obj = simulation.status.objective
    assert (
        charge_obj - no_charge_obj
        == (
            max(
                electric_load_mwh
                + [
                    minimum_network_demand,
                ]
            )
            - chp_size
        )
        * network_charge
        + chp_size * gas_prices
    )

    #  TODO difference in objective
    # test that the difference in the objective is the import charge - gas consumption

    # expect CHP to run at peak electric demand - minimum network demand
    peak_demand = np.max(electric_load_mwh)
    peak_demand_idx = np.argmax(electric_load_mwh)
    chp_operation = np.zeros_like(electric_load_mwh)
    chp_operation[peak_demand_idx] = peak_demand - minimum_network_demand
    np.testing.assert_allclose(
        chp_operation, simulation.results["chp-electric_generation_mwh"]
    )


def test_minimum_export_all_intervals_chp() -> None:
    """
    Test that our CHP asset is incentivised to export a minimum amount.

    todo

    - run simulation with no minimum export charge
    - run simulation with a minimum export charge, assert that export is at least X for all intervals
    """

    # run a simulation with no max import charge
    # with a CHP that never runs - zero value electricity
    electric_load_mwh = 0.0
    electricity_prices = [30, 25, 30.0]
    gas_prices = 20

    chp_size = 10
    site = epl.Site(
        assets=[
            epl.CHP(
                electric_efficiency_pct=1.0,
                electric_power_max_mw=chp_size,
                electric_power_min_mw=chp_size,
            )
        ],
        gas_prices=gas_prices,
        electricity_prices=electricity_prices,
        electric_load_mwh=electric_load_mwh,
    )
    no_charge_terms = [
        {
            "asset_type": "site",
            "variable": "import_power_mwh",
            "interval_data": "electricity_prices",
        },
        {
            "asset_type": "site",
            "variable": "export_power_mwh",
            "interval_data": "electricity_prices",
            "coefficient": -1,
        },
        {
            "asset_type": "*",
            "variable": "gas_consumption_mwh",
            "interval_data": "gas_prices",
        },
    ]
    simulation = site.optimize(
        verbose=0,
        objective={"terms": no_charge_terms},
    )
    np.testing.assert_allclose(
        simulation.results["chp-electric_generation_mwh"],
        np.full_like(electricity_prices, chp_size),
    )

    # run a simulation with a charge on the maximum of export and a constant
    # when the export is above this constant, an extra charge kicks in

    maximum_export_charge = 200
    maximum_export = 5.0
    charge_terms = [
        {
            "asset_type": "site",
            "variable": "import_power_mwh",
            "interval_data": "electricity_prices",
        },
        {
            "asset_type": "site",
            "variable": "export_power_mwh",
            "interval_data": "electricity_prices",
            "coefficient": -1,
        },
        {
            "asset_type": "*",
            "variable": "gas_consumption_mwh",
            "interval_data": "gas_prices",
        },
        {
            "function": "min_many_variables",
            "variables": {
                "asset_type": "chp",
                "variable": "electric_generation_mwh",
            },
            "constant": maximum_export,
            "coefficient": maximum_export_charge,
            "M": 1000,
        },
    ]
    simulation = site.optimize(
        verbose=0,
        objective={"terms": charge_terms},
    )
    np.testing.assert_allclose(
        simulation.results["chp-electric_generation_mwh"], [10.0, 0.0, 10.0]
    )

    # TODO test the objective values???


def test_filter_assets() -> None:
    """Test that the filter on `*` works correctly."""

    electric_load_mwh = 0.0
    electricity_prices = [-50.0, -25, -50]
    gas_prices = 20

    # run a simulation where we incentivise all generators to not run
    site = epl.Site(
        assets=[
            epl.CHP(
                electric_efficiency_pct=1.0,
                electric_power_max_mw=20,
                electric_power_min_mw=0,
            ),
            epl.RenewableGenerator(
                electric_generation_mwh=[30, 30, 30],
                electric_generation_lower_bound_pct=0.0,
            ),
        ],
        gas_prices=gas_prices,
        electricity_prices=electricity_prices,
        electric_load_mwh=electric_load_mwh,
    )
    no_incentive = site.optimize(
        objective={
            "terms": [
                {
                    "asset_type": "site",
                    "variable": "import_power_mwh",
                    "interval_data": "electricity_prices",
                },
                {
                    "asset_type": "site",
                    "variable": "export_power_mwh",
                    "interval_data": "electricity_prices",
                    "coefficient": -1,
                },
                {
                    "asset_type": "*",
                    "variable": "gas_consumption_mwh",
                    "interval_data": "gas_prices",
                },
            ]
        }
    )
    np.testing.assert_allclose(
        no_incentive.results["total-electric_generation_mwh"], [0, 0, 0]
    )

    # run a simulation where we incentivise a CHP to run at a minimum load
    chp_incentive = site.optimize(
        objective={
            "terms": [
                {
                    "asset_type": "site",
                    "variable": "import_power_mwh",
                    "interval_data": "electricity_prices",
                },
                {
                    "asset_type": "site",
                    "variable": "export_power_mwh",
                    "interval_data": "electricity_prices",
                    "coefficient": -1,
                },
                {
                    "asset_type": "*",
                    "variable": "gas_consumption_mwh",
                    "interval_data": "gas_prices",
                },
                {
                    "type": "complex",
                    "function": "min_many_variables",
                    "variables": {
                        "asset_name": "chp",
                        "variable": "electric_generation_mwh",
                    },
                    "constant": 15,
                    "coefficient": -2000,
                    "M": 1000,
                },
            ]
        }
    )
    np.testing.assert_allclose(
        chp_incentive.results["chp-electric_generation_mwh"], [15, 15, 15]
    )
    np.testing.assert_allclose(
        chp_incentive.results["total-electric_generation_mwh"], [15, 15, 15]
    )

    # run a simulation where we incentivise both generators to run at a minimum load
    gen_incentive = site.optimize(
        objective={
            "terms": [
                {
                    "asset_type": "site",
                    "variable": "import_power_mwh",
                    "interval_data": "electricity_prices",
                },
                {
                    "asset_type": "site",
                    "variable": "export_power_mwh",
                    "interval_data": "electricity_prices",
                    "coefficient": -1,
                },
                {
                    "asset_type": "*",
                    "variable": "gas_consumption_mwh",
                    "interval_data": "gas_prices",
                },
                {
                    "type": "complex",
                    "function": "min_many_variables",
                    "variables": {
                        "asset_type": "*",
                        "variable": "electric_generation_mwh",
                    },
                    "constant": 15,
                    "coefficient": -2000,
                    "M": 1000,
                },
            ]
        }
    )

    np.testing.assert_allclose(
        gen_incentive.results["total-electric_generation_mwh"], [30, 30, 30]
    )
    np.testing.assert_allclose(
        gen_incentive.results["renewable-generator-electric_generation_mwh"],
        [15, 15, 15],
    )
    np.testing.assert_allclose(
        gen_incentive.results["chp-electric_generation_mwh"], [15, 15, 15]
    )
