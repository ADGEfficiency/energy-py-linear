import collections

import numpy as np
import pandas as pd

from energypylinear.data import IntervalData
from energypylinear.optimizer import Optimizer


def extract_results(interval_data: IntervalData, vars: dict) -> pd.DataFrame:

    optimizer = Optimizer()
    results = collections.defaultdict(list)
    for i in interval_data.idx:
        site = vars["sites"][i]

        results["import_power_mwh"].append(site.import_power_mwh.value())
        results["export_power_mwh"].append(site.export_power_mwh.value())

        spill = vars["spills"][i]
        for attr in [
            "electric_generation_mwh",
            "high_temperature_generation_mwh",
            "electric_load_mwh",
            "low_temperature_load_mwh",
        ]:
            name = f"{spill.cfg.name}"
            results[f"{name}-{attr}"].append(getattr(spill, attr).value())

        if len(vars["batteries"]) > 0:
            for battery in vars["batteries"][i]:
                name = f"{battery.cfg.name}"
                for attr in [
                    "charge_mwh",
                    "discharge_mwh",
                    "losses_mwh",
                    "initial_charge_mwh",
                    "final_charge_mwh",
                    # "efficiency_pct",  TODO this is a float
                ]:
                    results[f"{name}-{attr}"].append(getattr(battery, attr).value())

        if len(vars["generators"]) > 0:
            for generator in vars["generators"][i]:
                name = f"{generator.cfg.name}"
                for attr in [
                    "electric_generation_mwh",
                    "gas_consumption_mwh",
                    "high_temperature_generation_mwh",
                ]:
                    results[f"{name}-{attr}"].append(getattr(generator, attr).value())

        if len(vars["boilers"]):
            boilers = vars["boilers"][i]
            for boiler in boilers:
                name = f"{boiler.cfg.name}"
                for attr in ["high_temperature_generation_mwh", "gas_consumption_mwh"]:
                    results[f"{name}-{attr}"].append(getattr(boiler, attr).value())

        if len(vars["evs-array"]):
            evs = vars["evs-array"][i]
            """
            for each timestep (i)
                for each charger
                    select all charge events
            """
            for charger_idx, charger_cfg in enumerate(evs.charger_cfgs[0, 0, :]):
                for attr in ["charge_mwh", "charge_binary"]:
                    results[f"charger-{charger_cfg.name}-{attr}"].append(
                        #   getting attribute on a EVsArrayOneInterval object
                        sum(
                            [
                                optimizer.value(x)
                                for x in getattr(evs, attr)[0, :, charger_idx]
                            ]
                        )
                    )
            """
            for each timestep (i)
                for each charge event
                    select all chargers

            """
            for charge_event_idx, _ in enumerate(interval_data.evs.charge_event_mwh):
                #  same attribute access as above - TODO refactor
                results[f"charge-event-{charge_event_idx}-charge_mwh"].append(
                    sum(
                        [
                            optimizer.value(x)
                            for x in getattr(evs, "charge_mwh")[0, charge_event_idx, :]
                        ]
                    )
                )

        if len(vars["spill-evs-array"]):
            evs = vars["spill-evs-array"][i]
            for charger_idx, charger_cfg in enumerate(evs.charger_cfgs[0, 0, :]):
                for attr in ["charge_mwh", "charge_binary"]:
                    results[f"spill-charger-{charger_cfg.name}-{attr}"].append(
                        sum(
                            [
                                optimizer.value(x)
                                for x in getattr(evs, attr)[0, :, charger_idx]
                            ]
                        )
                    )

    #  add totals
    #  can I do this without pandas??
    results = pd.DataFrame(results)
    for col in [
        "electric_generation_mwh",
        "gas_consumption_mwh",
        "high_temperature_generation_mwh",
    ]:
        cols = [c for c in results.columns if (col in c)]
        results[f"total-{col}"] = results[cols].sum(axis=1)

    #  add balances + check them - TODO
    validate_results(interval_data, results)

    #  add warnings on the spill generator + boiler use
    return results


def validate_results(interval_data: IntervalData, results: pd.DataFrame) -> None:
    """
    validations TODO
    - don't import / export in same interval
    - column names (types?)
    - validate against interval data - lengths
    """

    cols = [
        "import_power_mwh",
        "export_power_mwh",
    ]
    for c in cols:
        assert c in results.columns

    #  check that the sum across charge_event_mwh and charge_event_usage is the same
    charge_event_mwh = interval_data.evs.charge_event_mwh
    cols = [
        c
        for c in results.columns
        if c.startswith("charge-event-") and c.endswith("-charge_mwh")
    ]
    cols += [
        c
        for c in results.columns
        if c.startswith("spill-") and c.endswith("-charge_mwh")
    ]
    charge_event_usage_mwh = results[cols].sum(axis=0).values
    # np.testing.assert_almost_equal(charge_event_mwh, charge_event_usage_mwh, decimal=5)
