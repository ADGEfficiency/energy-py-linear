"""Example of using a battery for economic and carbon optimization."""
import energypylinear as epl

if __name__ == "__main__":
    #  input interval data
    electricity_prices = [100, 50, 200, -100, 0, 200, 100, -100]
    carbon_intensities = [0.1, 0.2, 0.1, 0.15, 0.01, 0.7, 0.5, 0.01]

    #  battery model
    asset = epl.Battery(power_mw=2, capacity_mwh=4, efficiency=0.9)

    #  optimize for money
    results = asset.optimize(
        electricity_prices=electricity_prices,
        electricity_carbon_intensities=carbon_intensities,
        freq_mins=60,
        initial_charge_mwh=1,
        final_charge_mwh=3,
    )
    asset.plot(results, "battery.png")

    #  optimize for carbon
    results = asset.optimize(
        electricity_prices=electricity_prices,
        electricity_carbon_intensities=carbon_intensities,
        freq_mins=60,
        initial_charge_mwh=1,
        final_charge_mwh=3,
        objective="carbon",
    )
    asset.plot(results, "battery-carbon.png")
