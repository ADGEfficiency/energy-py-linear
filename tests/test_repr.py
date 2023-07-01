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
    ]
    for th in things:
        print(th)
