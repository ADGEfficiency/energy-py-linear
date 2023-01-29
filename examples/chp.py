"""example of using a CHP for economic and carbon optimization
CHP = Combined Heat & Power.
"""
import energypylinear as epl

if __name__ == "__main__":
    #  input interval data
    electricity_prices = [100, 50, 200, -100, 0, 200, 100, -100]
    carbon_intensities = [0.1, 0.2, 0.1, 0.15, 0.01, 0.7, 0.5, 0.01]

    #  100 MWe gas turbine
    asset = epl.chp.Generator(
        electric_power_max_mw=100,
        electric_power_min_mw=50,
        electric_efficiency_pct=0.3,
        high_temperature_efficiency_pct=0.5,
    )

    #  optimize for money
    results = asset.optimize(
        electricity_prices=electricity_prices,
        electricity_carbon_intensities=carbon_intensities,
        freq_mins=60,
    )
    asset.plot(results, "gas-turbine.png")

    #  100 MWe gas engine
    asset = epl.chp.Generator(
        electric_power_max_mw=100,
        electric_power_min_mw=10,
        electric_efficiency_pct=0.4,
        high_temperature_efficiency_pct=0.2,
        low_temperature_efficiency_pct=0.2,
    )
    #  optimize for carbon
    results = asset.optimize(
        electricity_prices=electricity_prices,
        electricity_carbon_intensities=carbon_intensities,
        freq_mins=60,
        objective="carbon",
    )
    asset.plot(results, "gas-engine-carbon.png")
