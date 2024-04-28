"""Tests the implementation of custom constraints."""

import numpy as np

"""

Want to test bunch of combinations

- LHS constant, RHS asset type = "*", interval_data = None, agg = None
- LHS constant, RHS asset type = "*", interval_data = None, agg = sum

- LHS asset_name, interal_data = something, agg = None, RHS constant
- LHS asset_name, interal_data = something, agg = sum, RHS constant

- LHS asset_name, interal_data = something, agg = None, RHS asset type = "*", interval_data = None, agg = None
- LHS asset_name, interal_data = something, agg = None, RHS asset type = "*", interval_data = None, agg = sum

Also test that we can do
- LHS asset_name, interal_data = something, agg = None, LHS asset type = "*", interval_data = None, agg = None AND RHS asset_name, interal_data = something, agg = None, RHS asset type = "*", interval_data = None, agg = sum

Raise value error if LHS and RHS both constant
"""

import energypylinear as epl
import pytest


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        ([10], [20]),
        ([10 + 20], [20 + 30]),
    ],
)
def test_no_all_constants(lhs: tuple, rhs: tuple) -> None:
    with pytest.raises(ValueError):
        epl.Constraint(lhs=lhs, rhs=rhs, sense="le")


n_intervals_combinations_test = 48
# fmt: off
test_custom_constraints_params = [
    # LHS constant, RHS asset type "*", no interval data, no aggregation
    (
        [
            (10, epl.ConstraintTerm(asset_type="*", variable="gas_consumption_mwh"), "ge", None)
        ],
        n_intervals_combinations_test,

    ),
    # # LHS constant, RHS asset type "*", no interval data, aggregation sum
    (
        [
            (
                epl.ConstraintTerm(asset_type="*", variable="gas_consumption_mwh"),
                10,
                "le",
                "sum",
            )
        ],
        1
    ),
    # # LHS asset_name, interval data, no aggregation, RHS constant
    # (
    #     [epl.ConstraintTerm(asset_name="test_asset", variable="var", interval_data="data")],
    #     [20],
    #     "eq",
    # ),
    # # LHS asset_name, interval data, aggregation sum, RHS constant
    # (
    #     [epl.ConstraintTerm(asset_name="test_asset", variable="var", interval_data="data", aggregation="sum", )],
    #     [30],
    #     "eq",
    # ),
    # # LHS asset_name, interval data, no aggregation, RHS asset type "*", no interval data, no aggregation
    # (
    #     [epl.ConstraintTerm(asset_name="test_asset", variable="var", interval_data="data")],
    #     [epl.ConstraintTerm(asset_type="*", variable="var")],
    #     "ge",
    # ),
    # # LHS asset_name, interval data, no aggregation, RHS asset type "*", no interval data, aggregation sum
    # (
    #     [epl.ConstraintTerm(asset_name="test_asset", variable="var", interval_data="data")],
    #     [epl.ConstraintTerm(asset_type="*", variable="var", aggregation="sum")],
    #     "ge",
    # ),
    # # Combination of different types on LHS and RHS
    # (
    #     [epl.ConstraintTerm(asset_name="lhs_asset", variable="var", interval_data="data")],
    #     [epl.ConstraintTerm(asset_type="*", variable="var", aggregation="sum")],
    #     "ge",
    # ),
    # # Multiple constraint terms on both sides
    # (
    #     [epl.ConstraintTerm(asset_name="lhs_asset", variable="var", interval_data="data"), 40],
    #     [epl.ConstraintTerm(asset_type="*", variable="var", aggregation="sum"), 30],
    #     "ge",
    # ),
    # (
    #     [
    #         epl.ConstraintTerm(asset_name="lhs_asset", variable="var", interval_data="data"),
    #         epl.ConstraintTerm(asset_name="lhs_asset", variable="var", interval_data="data"),
    #     ],
    #     [
    #         epl.ConstraintTerm(asset_name="lhs_asset", variable="var", interval_data="data"),
    #         epl.ConstraintTerm(asset_name="lhs_asset", variable="var", interval_data="data"),
    #     ],
    #     "ge",
    # ),
]
# fmt: on


@pytest.mark.parametrize(
    "custom_constraints, expected_n_extra_constraints",
    test_custom_constraints_params,
)
def test_custom_constraint_combinations(
    custom_constraints,
    expected_n_extra_constraints,
):
    """
    Tests many combinations of custom constraint terms:
    - contants,
    - asset types,
    - asset names,
    - interval data,
    - aggregations,
    - senses.

    Only checks that we can create and optimize with these constraints, not that we get
    the correct result. Correctness is tested in other tests in this file.

    """
    electricity_prices = np.random.uniform(-100, 100, n_intervals_combinations_test)

    constraints = []
    for constraint in custom_constraints:
        lhs, rhs, sense, aggregation = constraint
        constraints.append(
            epl.Constraint(lhs=lhs, rhs=rhs, sense=sense, aggregation=aggregation)
        )

    no_constraint_asset = epl.Battery(electricity_prices=electricity_prices)
    no_constraint_asset.optimize()

    asset = epl.Battery(
        electricity_prices=np.random.uniform(-100, 100, 48), constraints=constraints
    )
    asset.optimize()
    n_extra_constraints = len(asset.site.optimizer.constraints()) - len(
        no_constraint_asset.site.optimizer.constraints()
    )

    n_agg_constraints = len([c for c in constraints if c.aggregation == "sum"])
    n_non_agg_constraints = len(constraints) - n_agg_constraints

    assert (
        n_extra_constraints
        == 3 * expected_n_extra_constraints * n_non_agg_constraints + n_agg_constraints
    )

    # now test with two assets
    asset_one = epl.Battery(name="battery-eins")
    asset_second = epl.Battery(name="battery-zwei")

    # now do with dictionaries
    constraints = [c.dict() for c in constraints]
    no_constraint_site = epl.Site(
        assets=[asset_one, asset_second, epl.Spill()],
        electricity_prices=np.random.uniform(-100, 100, 48),
    )
    no_constraint_site.optimize()

    site = epl.Site(
        assets=[asset_one, asset_second, epl.Spill()],
        electricity_prices=np.random.uniform(-100, 100, 48),
        constraints=constraints,
    )
    site.optimize()
    n_extra_constraints = len(site.optimizer.constraints()) - len(
        no_constraint_site.optimizer.constraints()
    )
    assert (
        n_extra_constraints
        == 4 * expected_n_extra_constraints * n_non_agg_constraints + n_agg_constraints
    )


def test_battery_cycle_constraint() -> None:
    # test with the epl objects
    asset = epl.Battery(
        power_mw=1,
        capacity_mwh=2,
        efficiency_pct=0.98,
        constraints=[
            epl.Constraint(
                lhs=[
                    epl.ConstraintTerm(
                        asset_type="battery", variable="electric_charge_mwh"
                    ),
                    epl.ConstraintTerm(
                        asset_type="battery", variable="electric_discharge_mwh"
                    ),
                ],
                rhs=2 * 15,
                sense="le",
            )
        ],
    )

    # test with dictionaries, invert the LHS & RHS
    asset = epl.Battery(
        power_mw=1,
        capacity_mwh=2,
        efficiency_pct=0.98,
        constraints=[
            {
                "lhs": 2 * 15,
                "rhs": [
                    {"asset_type": "battery", "variable": "electric_charge_mwh"},
                    {"asset_type": "battery", "variable": "electric_discharge_mwh"},
                ],
                "sense": "ge",
            }
        ],
    )

    # TODO - add a test with a coefficient of 2.0, to allow twice as many cycles
