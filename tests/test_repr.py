"""Keeping coverage happy by testing __repr__ with print."""
import numpy as np

import energypylinear as epl


def test_repr() -> None:
    """Test we can print our things."""
    things = [
        epl.Battery(),
        epl.EVs(chargers_power_mw=[10], charge_events_capacity_mwh=[10]),
        epl.assets.evs.EVsArrayOneInterval(
            i=0,
            cfg=epl.assets.evs.EVsConfig(
                name="evs",
                charger_cfgs=np.array([0]),
                spill_charger_cfgs=np.array([0]),
                charge_event_cfgs=np.array([0]),
            ),
            initial_soc_mwh=np.array([0]),
            final_soc_mwh=np.array([0]),
            electric_charge_mwh=np.array([0]),
            electric_charge_binary=np.array([0]),
            electric_discharge_mwh=np.array([0]),
            electric_discharge_binary=np.array([0]),
            electric_loss_mwh=np.array([0]),
        ),
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
