"""Keeping coverage happy by testing __repr__ with print."""
import numpy as np

import energypylinear as epl


def test_repr() -> None:
    """Test we can print our things."""

    ds = epl.data_generation.generate_random_ev_input_data(
        48, n_chargers=3, charge_length=3, n_charge_events=12, seed=42
    )
    site = epl.Site(assets=[], electricity_prices=np.array([0, 0]))
    things = [
        site,
        site.cfg,
        epl.HeatPump(electric_power_mw=1.0, cop=3),
        epl.Battery(),
        epl.CHP(),
        epl.Boiler(),
        epl.EVs(**ds),
        epl.RenewableGenerator(electric_generation_mwh=[10]),
        epl.assets.evs.EVOneInterval(
            i=0,
            charge_event_idx=0,
            charge_event_cfg=epl.assets.evs.ChargeEventConfig(
                name="ce", capacity_mwh=10, efficiency_pct=0.5
            ),
            charger_idx=0,
            charger_cfg=epl.assets.evs.ChargerConfig(
                name="ca", power_min_mw=0, power_max_mw=0
            ),
            cfg=epl.assets.evs.EVsConfig(
                name="evs",
                charger_cfgs=np.array([0]),
                spill_charger_cfgs=np.array([0]),
                charge_event_cfgs=np.array([0]),
                freq_mins=0,
                charge_events=np.array([[0], [0]]),
            ),
            initial_soc_mwh=0.0,
            final_soc_mwh=0.0,
            electric_charge_mwh=0.0,
            electric_charge_binary=0,
            electric_discharge_mwh=0.0,
            electric_discharge_binary=0,
            electric_loss_mwh=0.0,
        ),
        epl.assets.evs.EVsArrayOneInterval(
            i=0,
            cfg=epl.assets.evs.EVsConfig(
                name="evs",
                charger_cfgs=np.array([0]),
                spill_charger_cfgs=np.array([0]),
                charge_event_cfgs=np.array([0]),
                freq_mins=0,
                charge_events=np.array([[0], [0]]),
            ),
            initial_soc_mwh=np.array([0]),
            final_soc_mwh=np.array([0]),
            electric_charge_mwh=np.array([0]),
            electric_charge_binary=np.array([0]),
            electric_discharge_mwh=np.array([0]),
            electric_discharge_binary=np.array([0]),
            electric_loss_mwh=np.array([0]),
            charge_event_idxs=np.array([0]),
            charger_idxs=np.array([0]),
        ),
        epl.Spill(),
        epl.Valve(),
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
            gas=epl.accounting.accounting.GasAccount(
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
