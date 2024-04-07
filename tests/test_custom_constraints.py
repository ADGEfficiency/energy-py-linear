"""Tests the implementation of custom constraints."""

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


# fmt: off
test_custom_constraints_params = [
    # LHS constant, RHS asset type "*", no interval data, no aggregation
    (
        [10],
        [epl.ConstraintTerm(asset_type="*", variable="var")],
        "le"
    ),
    # LHS constant, RHS asset type "*", no interval data, aggregation sum
    (
        [10],
        [epl.ConstraintTerm(asset_type="*", variable="var", aggregation="sum")],
        "le"
    ),
    # LHS asset_name, interval data, no aggregation, RHS constant
    (
        [epl.ConstraintTerm(asset_name="test_asset", variable="var", interval_data="data")],
        [20],
        "eq",
    ),
    # LHS asset_name, interval data, aggregation sum, RHS constant
    (
        [epl.ConstraintTerm(asset_name="test_asset", variable="var", interval_data="data", aggregation="sum", )],
        [30],
        "eq",
    ),
    # LHS asset_name, interval data, no aggregation, RHS asset type "*", no interval data, no aggregation
    (
        [epl.ConstraintTerm(asset_name="test_asset", variable="var", interval_data="data")],
        [epl.ConstraintTerm(asset_type="*", variable="var")],
        "ge",
    ),
    # LHS asset_name, interval data, no aggregation, RHS asset type "*", no interval data, aggregation sum
    (
        [epl.ConstraintTerm(asset_name="test_asset", variable="var", interval_data="data")],
        [epl.ConstraintTerm(asset_type="*", variable="var", aggregation="sum")],
        "ge",
    ),
    # Combination of different types on LHS and RHS
    (
        [epl.ConstraintTerm(asset_name="lhs_asset", variable="var", interval_data="data")],
        [epl.ConstraintTerm(asset_type="*", variable="var", aggregation="sum")],
        "ge",
    ),
    # Multiple constraint terms on both sides
    (
        [epl.ConstraintTerm(asset_name="lhs_asset", variable="var", interval_data="data"), 40],
        [epl.ConstraintTerm(asset_type="*", variable="var", aggregation="sum"), 30],
        "ge",
    ),
    (
        [
            epl.ConstraintTerm(asset_name="lhs_asset", variable="var", interval_data="data"),
            epl.ConstraintTerm(asset_name="lhs_asset", variable="var", interval_data="data"),
        ],
        [
            epl.ConstraintTerm(asset_name="lhs_asset", variable="var", interval_data="data"),
            epl.ConstraintTerm(asset_name="lhs_asset", variable="var", interval_data="data"),
        ],
        "ge",
    ),
]
# fmt: on


@pytest.mark.parametrize("lhs, rhs, sense", test_custom_constraints_params)
def test_custom_constraints(lhss, rhss, sense):
    constraints = []
    for lhs, rhs in zip(lhss, rhss):
        constraints.append(epl.Constraint(lhs=lhs, rhs=rhs, sense=sense))

    asset = epl.Battery(constraints=constraints)
    asset.optimize()

    site = epl.Site(assets=[asset], constraints=constraints)
    site.optimize()

    # now do with dictionaries
    constraints = []
    for lhs, rhs in zip(lhss, rhss):
        if isinstance(lhs, epl.ConstraintTerm):
            lhs = lhs.dict()

        if isinstance(rhs, epl.ConstraintTerm):
            rhs = rhs.dict()

        constraints.append({"lhs": lhs, "rhs": rhs, "sense": sense})

    asset = epl.Battery(constraints=constraints)
    asset.optimize()

    site = epl.Site(assets=[asset], constraints=constraints)
    site.optimize()


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
