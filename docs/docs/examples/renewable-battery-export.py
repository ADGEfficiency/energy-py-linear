"""Example of a site with a renewable generator and a battery exporting to the grid."""

import energypylinear as epl

assets = [
    epl.Battery(power_mw=10, capacity_mwh=20, efficiency_pct=0.9),
    epl.RenewableGenerator(
        electric_generation_mwh=[10, 20, 30, 20, 10],
        electric_generation_lower_bound_pct=0.5,
        name="solar",
    ),
]

site = epl.Site(
    assets=assets,
    electricity_carbon_intensities=[0.5, -0.5, 0.5, 0.5, -0.5],
    export_limit_mw=25,
)

simulation = site.optimize(objective="carbon")
