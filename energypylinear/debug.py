from rich import print


def debug_simulation(simulation):
    """be great to debug the balances here as well"""

    print("[red]DEBUG[/red]")
    debug = [
        "site-import_power_mwh",
        "site-export_power_mwh",
        "electricity_carbon_intensities",
    ]
    print(simulation[debug])

    def debug_column(simulation, col):
        cols = [c for c in simulation.columns if col in c]
        subset = simulation[cols]
        subset.columns = [c.replace(col, "") for c in subset.columns]
        print(col)
        print(subset)

    debug_column(simulation, "electric_charge_mwh")
    debug_column(simulation, "electric_discharge_mwh")
    debug_column(simulation, "initial_soc_mwh")
    debug_column(simulation, "final_soc_mwh")
    debug_column(simulation, "electric_loss_mwh")
