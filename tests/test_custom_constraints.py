"""Tests the implementation of custom constraints."""

import numpy as np
import pytest

import energypylinear as epl


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        (10, 20),
        ([10 + 20], [20 + 30]),
    ],
)
def test_no_all_constants(
    lhs: float | epl.ConstraintTerm | dict | list[float | epl.ConstraintTerm | dict],
    rhs: float | epl.ConstraintTerm | dict | list[float | epl.ConstraintTerm | dict],
) -> None:
    """Tests that we raise errors when we have all floats on both sides of a constraint."""
    with pytest.raises(ValueError):
        epl.Constraint(lhs=lhs, rhs=rhs, sense="le")


n_intervals_combinations_test = 48
test_custom_constraints_params = [
    (
        [
            (
                10,
                epl.ConstraintTerm(asset_type="*", variable="gas_consumption_mwh"),
                "ge",
                None,
            )
        ],
        n_intervals_combinations_test,
    ),
    (
        [
            (
                epl.ConstraintTerm(asset_type="*", variable="gas_consumption_mwh"),
                10,
                "le",
                "sum",
            )
        ],
        1,
    ),
    (
        [
            (
                epl.ConstraintTerm(
                    asset_name="battery",
                    variable="electric_charge_mwh",
                    interval_data="electricity_prices",
                ),
                20,
                "le",
                None,
            )
        ],
        n_intervals_combinations_test,
    ),
    (
        [
            (
                epl.ConstraintTerm(
                    asset_name="battery",
                    variable="electric_charge_mwh",
                    interval_data="electricity_prices",
                ),
                30,
                "le",
                "sum",
            )
        ],
        1,
    ),
    (
        [
            (
                epl.ConstraintTerm(
                    asset_name="battery",
                    variable="electric_charge_mwh",
                    interval_data="electricity_prices",
                ),
                epl.ConstraintTerm(asset_type="*", variable="electric_generation_mwh"),
                "ge",
                None,
            )
        ],
        n_intervals_combinations_test,
    ),
    (
        [
            (
                epl.ConstraintTerm(
                    asset_name="battery",
                    variable="electric_charge_mwh",
                    interval_data="electricity_prices",
                ),
                epl.ConstraintTerm(asset_type="*", variable="electric_generation_mwh"),
                "ge",
                None,
            )
        ],
        n_intervals_combinations_test,
    ),
    (
        [
            (
                epl.ConstraintTerm(
                    asset_name="battery",
                    variable="electric_charge_mwh",
                    interval_data="electricity_prices",
                ),
                epl.ConstraintTerm(asset_type="*", variable="electric_generation_mwh"),
                "le",
                "sum",
            )
        ],
        1,
    ),
    (
        [
            (
                [
                    epl.ConstraintTerm(
                        asset_name="battery",
                        variable="electric_generation_mwh",
                        interval_data="electricity_prices",
                    ),
                    epl.ConstraintTerm(
                        asset_type="*", variable="electric_generation_mwh"
                    ),
                ],
                [
                    epl.ConstraintTerm(
                        asset_name="battery",
                        variable="electric_generation_mwh",
                        interval_data="electricity_prices",
                    ),
                    epl.ConstraintTerm(
                        asset_type="*", variable="electric_generation_mwh"
                    ),
                ],
                "eq",
                "sum",
            )
        ],
        1,
    ),
    (
        [
            (
                epl.ConstraintTerm(
                    asset_name="battery",
                    variable="electric_generation_mwh",
                    interval_data="electricity_prices",
                ),
                epl.ConstraintTerm(asset_type="*", variable="electric_generation_mwh"),
                "eq",
                "sum",
            ),
            (
                epl.ConstraintTerm(
                    asset_name="battery",
                    variable="electric_generation_mwh",
                    interval_data="electricity_prices",
                ),
                epl.ConstraintTerm(asset_type="*", variable="electric_generation_mwh"),
                "eq",
                None,
            ),
        ],
        n_intervals_combinations_test + 1,
    ),
]


@pytest.mark.parametrize(
    "custom_constraints, expected_n_extra_constraints", test_custom_constraints_params
)
def test_custom_constraint_combinations(
    custom_constraints: list,
    expected_n_extra_constraints: int,
) -> None:
    """Tests many combinations of custom constraint terms:
    - contants,
    - asset types,
    - asset names,
    - interval data,
    - aggregations,
    - senses.

    Only checks that we can create and optimize with these constraints, not that we get
    the correct result. Correctness is tested in other tests in this file."""
    electricity_prices = np.random.uniform(-100, 100, n_intervals_combinations_test)

    constraints = []
    for constraint in custom_constraints:
        lhs, rhs, sense, aggregation = constraint
        constraints.append(
            epl.Constraint(
                lhs=lhs, rhs=rhs, sense=sense, interval_aggregation=aggregation
            )
        )

    no_constraint_asset = epl.Battery(electricity_prices=electricity_prices)
    no_constraint_asset.optimize(verbose=3)

    asset = epl.Battery(
        electricity_prices=np.random.uniform(-100, 100, 48), constraints=constraints
    )
    asset.optimize(verbose=3)
    n_extra_constraints = len(asset.site.optimizer.constraints()) - len(
        no_constraint_asset.site.optimizer.constraints()
    )

    assert n_extra_constraints == expected_n_extra_constraints

    # now test with two assets
    asset_one = epl.Battery(name="battery")
    asset_second = epl.Battery(name="battery-zwei")

    # now do with dictionaries
    constraints_dicts = [c.dict() for c in constraints]
    no_constraint_site = epl.Site(
        assets=[asset_one, asset_second, epl.Spill()],
        electricity_prices=np.random.uniform(-100, 100, 48),
    )
    no_constraint_site.optimize(verbose=3)

    site = epl.Site(
        assets=[asset_one, asset_second, epl.Spill()],
        electricity_prices=np.random.uniform(-100, 100, 48),
        constraints=constraints_dicts,
    )
    site.optimize(verbose=3)
    n_extra_constraints = len(site.optimizer.constraints()) - len(
        no_constraint_site.optimizer.constraints()
    )
    assert n_extra_constraints == expected_n_extra_constraints


def test_battery_cycle_constraint() -> None:
    """Test example where we constrain the cycles of one battery."""
    electricity_prices = np.random.normal(0.0, 10000, 128)
    cycle_limit = 2.0

    # optimize a battery with no constraints
    asset = epl.Battery(
        power_mw=1,
        capacity_mwh=2,
        efficiency_pct=0.98,
        electricity_prices=electricity_prices,
    )
    simulation = asset.optimize(verbose=3)
    total_battery_use_no_constraints = simulation.results.sum()[
        ["total-electric_charge_mwh", "total-electric_discharge_mwh"]
    ].sum()

    # optimize a battery with a constraint on number of cycles
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
                rhs=cycle_limit,
                sense="le",
                interval_aggregation="sum",
            )
        ],
        electricity_prices=electricity_prices,
    )
    simulation = asset.optimize(
        verbose=3,
    )
    total_battery_use_with_low_constraint = simulation.results.sum()[
        ["total-electric_charge_mwh", "total-electric_discharge_mwh"]
    ].sum()
    assert total_battery_use_with_low_constraint < total_battery_use_no_constraints
    np.testing.assert_allclose(total_battery_use_with_low_constraint, cycle_limit)

    # now test with dictionaries, and invert the LHS & RHS
    asset = epl.Battery(
        power_mw=1,
        capacity_mwh=2,
        efficiency_pct=0.98,
        constraints=[
            {
                "rhs": [
                    {"asset_type": "battery", "variable": "electric_charge_mwh"},
                    {"asset_type": "battery", "variable": "electric_discharge_mwh"},
                ],
                "lhs": cycle_limit,
                "sense": "ge",
                "interval_aggregation": "sum",
            }
        ],
        electricity_prices=electricity_prices,
    )
    simulation = asset.optimize(
        verbose=3,
    )
    total_battery_use_with_low_constraint = simulation.results.sum()[
        ["total-electric_charge_mwh", "total-electric_discharge_mwh"]
    ].sum()
    assert total_battery_use_with_low_constraint < total_battery_use_no_constraints
    np.testing.assert_allclose(total_battery_use_with_low_constraint, cycle_limit)

    # now test with a higher coefficient of 2.0, to allow twice as many cycles
    asset = epl.Battery(
        power_mw=1,
        capacity_mwh=2,
        efficiency_pct=0.98,
        constraints=[
            {
                "rhs": [
                    {"asset_type": "battery", "variable": "electric_charge_mwh"},
                    {"asset_type": "battery", "variable": "electric_discharge_mwh"},
                ],
                "lhs": cycle_limit * 2,
                "sense": "ge",
                "interval_aggregation": "sum",
            }
        ],
        electricity_prices=electricity_prices,
    )
    simulation = asset.optimize(
        verbose=3,
    )
    total_battery_use_with_low_constraint = simulation.results.sum()[
        ["total-electric_charge_mwh", "total-electric_discharge_mwh"]
    ].sum()
    assert total_battery_use_with_low_constraint < total_battery_use_no_constraints
    np.testing.assert_allclose(total_battery_use_with_low_constraint, cycle_limit * 2)

    # TODO - could test the non-sum aggregation here
    # would limit battery use in each interval


def test_battery_cycle_constraint_multiple_batteries() -> None:
    """Test example where we constrain the cycles of two batteries."""
    electricity_prices = np.random.normal(0.0, 10000, 128)
    cycle_limit = 2.0

    # site with two batteries, constrain both
    battery_names = ["battery", "battery-zwei"]
    site = epl.Site(
        assets=[
            epl.Battery(power_mw=1, capacity_mwh=2, efficiency_pct=0.98, name=name)
            for name in battery_names
        ],
        electricity_prices=electricity_prices,
        constraints=[
            epl.Constraint(
                lhs=[
                    epl.ConstraintTerm(asset_name=name, variable="electric_charge_mwh"),
                    epl.ConstraintTerm(
                        asset_name=name, variable="electric_discharge_mwh"
                    ),
                ],
                rhs=cycle_limit,
                sense="le",
                interval_aggregation="sum",
            )
            for name in battery_names
        ],
    )
    simulation = site.optimize(verbose=3)
    for name in battery_names:
        np.testing.assert_allclose(
            simulation.results.sum()[
                [f"{name}-electric_charge_mwh", f"{name}-electric_discharge_mwh"]
            ].sum(),
            cycle_limit,
        )

    # site with two batteries, constrain only one
    battery_names = ["battery", "battery-zwei"]
    site = epl.Site(
        assets=[
            epl.Battery(power_mw=1, capacity_mwh=2, efficiency_pct=0.98, name=name)
            for name in battery_names
        ],
        electricity_prices=electricity_prices,
        constraints=[
            epl.Constraint(
                lhs=[
                    epl.ConstraintTerm(
                        asset_name="battery-zwei", variable="electric_charge_mwh"
                    ),
                    epl.ConstraintTerm(
                        asset_name="battery-zwei", variable="electric_discharge_mwh"
                    ),
                ],
                rhs=cycle_limit,
                sense="le",
                interval_aggregation="sum",
            )
        ],
    )
    simulation = site.optimize(verbose=3)

    # unconstrained battery is not limited
    name = "battery"
    np.testing.assert_array_less(
        cycle_limit,
        simulation.results.sum()[
            [f"{name}-electric_charge_mwh", f"{name}-electric_discharge_mwh"]
        ].sum(),
    )

    # constrained battery is limited
    name = "battery-zwei"
    np.testing.assert_allclose(
        simulation.results.sum()[
            [f"{name}-electric_charge_mwh", f"{name}-electric_discharge_mwh"]
        ].sum(),
        cycle_limit,
    )


def test_limit_sum_generation_in_each_interval() -> None:
    """Test that we can constrain the sum of two generators within each interval."""

    idx_len = 4
    generator_size = 100
    solar_gen = [10.0, 20, 30, 40]

    # optimize unconstrained
    site = epl.Site(
        assets=[
            epl.RenewableGenerator(electric_generation_mwh=solar_gen, name="solar"),
            epl.CHP(electric_power_max_mw=generator_size, electric_efficiency_pct=0.5),
        ],
        electricity_prices=np.full(idx_len, 400),
    )
    simulation = site.optimize(verbose=3)
    totals = simulation.results.sum()
    assert totals["chp-electric_generation_mwh"].sum() == idx_len * generator_size
    assert totals["solar-electric_generation_mwh"].sum() == sum(solar_gen)

    # optimize with a constraint on the sum of the generators
    site = epl.Site(
        assets=[
            epl.RenewableGenerator(
                electric_generation_mwh=solar_gen,
                name="solar",
                electric_generation_lower_bound_pct=0.0,
            ),
            epl.CHP(electric_power_max_mw=generator_size, electric_efficiency_pct=0.5),
        ],
        electricity_prices=np.full(idx_len, 400),
        constraints=[
            epl.Constraint(
                lhs=epl.ConstraintTerm(
                    variable="electric_generation_mwh", asset_type="*"
                ),
                rhs=25,
                sense="le",
            )
        ],
    )
    simulation = site.optimize(verbose=3)
    np.testing.assert_allclose(
        simulation.results["chp-electric_generation_mwh"], [15, 5, 0, 0]
    )
    np.testing.assert_allclose(
        simulation.results["solar-electric_generation_mwh"], [10, 20, 25, 25]
    )
