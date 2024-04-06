import energypylinear as epl


def test_battery_cycle_constraint() -> None:
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
            )
        ],
    )

    asset = epl.Battery(
        power_mw=1,
        capacity_mwh=2,
        efficiency_pct=0.98,
        constraints=[
            {
                "lhs": [
                    {"asset_type": "battery", "variable": "electric_charge_mwh"},
                    {"asset_type": "battery", "variable": "electric_discharge_mwh"},
                ],
                "rhs": 2 * 15,
            }
        ],
    )
