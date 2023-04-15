import energypylinear as epl


def test_repr() -> None:
    things = [
        epl.Battery(),
        epl.Site(),
        epl.assets.spill.Spill(),
        epl.assets.valve.Valve(),
        epl.interval_data.IntervalData(electricity_prices=[10]),
    ]
    for th in things:
        print(th)
