"""Keeping coverage happy by testing __repr__ with print."""
import energypylinear as epl


def test_repr() -> None:
    """Test we can print our things."""
    things = [
        epl.Battery(),
        epl.Site(),
        epl.assets.spill.Spill(),
        epl.assets.valve.Valve(),
        epl.assets.boiler.Boiler(),
        epl.interval_data.IntervalData(electricity_prices=[10]),
        epl.Optimizer(),
    ]
    for th in things:
        print(th)