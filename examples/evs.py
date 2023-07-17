"""Example of smart charging of electric vehicles for economic and carbon optimization."""
import energypylinear as epl

if __name__ == "__main__":
    #  interval data
    electricity_prices: list[float] = [-100, 50, 30, 50, 40]
    charge_events = [
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0],
    ]
    charge_event_mwh: list[float] = [50, 100, 30, 40]

    #  ev model
    asset = epl.EVs(
        chargers_power_mw=[100, 100],
        charge_events_capacity_mwh=charge_event_mwh,
    )
    results = asset.optimize(
        electricity_prices=electricity_prices,
        charge_events=charge_events,
    )
    asset.plot(results, "evs.png")
