import energypylinear as epl

assets = [
    epl.Battery(
        power_mw=20,
        capacity_mwh=40,
        efficiency=0.9,
    ),
    epl.Battery(power_mw=40, capacity_mwh=10, efficiency=0.8, name="battery-2"),
    epl.chp.Generator(
        electric_power_max_mw=100,
        electric_power_min_mw=50,
        electric_efficiency_pct=0.3,
        high_temperature_efficiency_pct=0.5,
    ),
]

site = epl.Site(assets=assets)
data = site.optimize(electricity_prices=[100, 50, 200, -100, 0, 200, 100, -100])
print(data)
