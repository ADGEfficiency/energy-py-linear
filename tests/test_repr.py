"""Keeping coverage happy by testing __repr__ with print."""
import energypylinear as epl


def test_repr() -> None:
    """Test we can print our things."""
    things = [
        epl.Battery(),
        epl.EVs(chargers_power_mw=[10], charge_events_capacity_mwh=[10]),
        epl.Site(),
        epl.assets.spill.Spill(),
        epl.assets.valve.Valve(),
        epl.assets.boiler.Boiler(),
        epl.interval_data.IntervalData(electricity_prices=[10]),
        epl.Optimizer(),
        epl.accounting.accounting.Account(cost=0, emissions=0),
        epl.accounting.accounting.Accounts(
            electricity=epl.accounting.accounting.ElectricityAccount(
                import_cost=0,
                export_cost=0,
                cost=0,
                import_emissions=0,
                export_emissions=0,
                emissions=0,
            ),
            gas=epl.accounting.accounting.ElectricityAccount(
                import_cost=0,
                export_cost=0,
                cost=0,
                import_emissions=0,
                export_emissions=0,
                emissions=0,
            ),
            profit=0,
            cost=0,
            emissions=0,
        ),
        epl.interval_data.IntervalVars(),
    ]
    for th in things:
        repr(th)
        str(th)
        print(th)
