"""Tests the use of FunctionTerms in a custom objective function."""
import hypothesis
import numpy as np
import pulp

import energypylinear as epl

atol = 1e-4
settings = hypothesis.settings(
    print_blob=True,
    max_examples=1000,
    report_multiple_bugs=False,
)


def coerce_variables(
    a: float,
    b: float,
    a_gap: float,
    b_gap: float,
    a_is_float: bool,
    b_is_float: bool,
    opt: "epl.Optimizer",
) -> tuple[float | pulp.LpVariable, float | pulp.LpVariable]:
    """Helper function to transform hypothesis parameters."""
    if a_is_float:
        av = a
    else:
        av = opt.continuous("a", low=a, up=a + a_gap)

    if b_is_float and not a_is_float:
        bv = b
    else:
        bv = opt.continuous("b", low=b, up=b + b_gap)

    if isinstance(av, float):
        assert isinstance(bv, pulp.LpVariable)
    if isinstance(bv, float):
        assert isinstance(av, pulp.LpVariable)

    return av, bv


@hypothesis.settings(settings)
@hypothesis.given(
    a=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=-100000, max_value=100000
    ),
    b=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=-100000, max_value=100000
    ),
    a_gap=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=0.1, max_value=10000
    ),
    b_gap=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=0.1, max_value=10000
    ),
    a_is_float=hypothesis.strategies.booleans(),
    b_is_float=hypothesis.strategies.booleans(),
)
def test_max_two_variables(
    a: float, b: float, a_gap: float, b_gap: float, a_is_float: bool, b_is_float: bool
) -> None:
    """Tests that we can constrain a variable to be the maximum of two other variables."""
    opt = epl.Optimizer()
    av, bv = coerce_variables(a, b, a_gap, b_gap, a_is_float, b_is_float, opt)
    cv = opt.max_two_variables(
        "max-a-b", av, bv, M=max(abs(a) + a_gap, abs(b) + b_gap) * 2.0
    )
    opt.objective(av + bv)
    opt.solve(verbose=3)
    np.testing.assert_allclose(max(a, b), cv.value(), atol=atol)


@hypothesis.settings(settings)
@hypothesis.given(
    a=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=-100000, max_value=100000
    ),
    b=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=-100000, max_value=100000
    ),
    a_gap=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=0.1, max_value=10000
    ),
    b_gap=hypothesis.strategies.floats(
        allow_infinity=False, allow_nan=False, min_value=0.1, max_value=10000
    ),
    a_is_float=hypothesis.strategies.booleans(),
    b_is_float=hypothesis.strategies.booleans(),
)
def test_min_two_variables(
    a: float, b: float, a_gap: float, b_gap: float, a_is_float: bool, b_is_float: bool
) -> None:
    """Tests that we can constrain a variable to be the maximum of two other variables."""
    opt = epl.Optimizer()
    av, bv = coerce_variables(a, b, a_gap, b_gap, a_is_float, b_is_float, opt)
    cv = opt.min_two_variables(
        "min-a-b", av, bv, M=max(abs(a) + a_gap, abs(b) + b_gap) * 2.0
    )
    opt.objective(av + bv)
    opt.solve(verbose=3)
    np.testing.assert_allclose(min(a, b), cv.value(), atol=atol)


# @hypothesis.settings(
#     settings,
#     deadline=2000,
# )
# @hypothesis.given(
#     electricity_prices=hypothesis.strategies.lists(
#         hypothesis.strategies.floats(
#             min_value=2, max_value=1000, allow_nan=False, allow_infinity=False
#         ),
#         min_size=10,
#         max_size=10,
#     ),
#     wind_mwh=hypothesis.strategies.floats(
#         min_value=0.1, max_value=1000, allow_nan=False, allow_infinity=False
#     ),
#     export_threshold_mwh=hypothesis.strategies.floats(
#         min_value=0, max_value=1000, allow_nan=False, allow_infinity=False
#     ),
#     export_network_charge=hypothesis.strategies.floats(
#         min_value=0, max_value=1000, allow_nan=False, allow_infinity=False
#     ),
# )
# def test_function_term_export_tariff(
#     electricity_prices: list[float],
#     wind_mwh: float,
#     export_threshold_mwh: float,
#     export_network_charge: float,
# ) -> None:
#     """Test that we can use a function term in a custom objective.

#     This tests the example of having a network tariff that is the minimum of two things:

#     1. a constant value (like 10 MW),
#     2. the site import.

#     This example is a bit odd - we apply a charge to the export when it is above a certain value.
#     """
#     assets: list[epl.Asset] = [
#         epl.RenewableGenerator(
#             electric_generation_mwh=wind_mwh,
#             name="wind",
#             electric_generation_lower_bound_pct=0.0,
#         ),
#     ]
#     site = epl.Site(
#         assets=assets,
#         electricity_prices=electricity_prices,
#     )
#     terms = [
#         {
#             "asset_type": "site",
#             "variable": "import_power_mwh",
#             "interval_data": "electricity_prices",
#         },
#         {
#             "asset_type": "site",
#             "variable": "export_power_mwh",
#             "interval_data": "electricity_prices",
#             "coefficient": -1,
#         },
#         {
#             "type": "function",
#             "function": "max_two_variables",
#             "a": {
#                 "asset_type": "site",
#                 "variable": "export_power_mwh",
#             },
#             "b": export_threshold_mwh,
#             "coefficient": export_network_charge,
#             "M": wind_mwh * 10 if wind_mwh > 0 else 10,
#         },
#     ]
#     simulation = site.optimize(
#         verbose=4,
#         objective={"terms": terms},
#     )

#     full_export = np.array(electricity_prices) > export_network_charge

#     if sum(full_export) > 0:
#         np.testing.assert_allclose(
#             simulation.results["site-export_power_mwh"][full_export], wind_mwh
#         )
#     if sum(~full_export) > 0:
#         np.testing.assert_allclose(
#             simulation.results["site-export_power_mwh"][~full_export],
#             min(export_threshold_mwh, wind_mwh),
#         )


def test_minimum_two_variables() -> None:
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
            electric_efficiency_pct=electric_efficiency, electric_power_max_mw=chp_size
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
            "type": "function",
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

    1. no export - when prices are very negative, and the minmum export incentive outweighs the revenue from exporting
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
        epl.CHP(electric_efficiency_pct=electric_efficiency, electric_power_max_mw=10)
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
            "type": "function",
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

    Would require summing all the asset types - interesting alternative to broadcasting...
    """
    pass
