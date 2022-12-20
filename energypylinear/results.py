import collections

import numpy as np
import pandas as pd

from energypylinear.data import IntervalData
from energypylinear.optimizer import Optimizer


def extract_results(interval_data: IntervalData, vars: dict) -> pd.DataFrame:
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
                    results[f"{charger_cfg.name}-{attr}"].append(
                        sum([x.value() for x in getattr(evs, attr)[0, :, charger_idx]])
                    )

            #  want to grab all the charge_mwh for each charge event
            for charge_event_idx, _ in enumerate(evs.charger_cfgs[0, :, 0]):
                for attr in ["charge_mwh"]:
                    results[f"charge-event-{charge_event_idx}-{attr}"].append(
                        sum(
                            [
                                x.value()
                                for x in getattr(evs, attr)[0, charge_event_idx, :]
                            ]
                        )
                    )

            evs = vars["spill-evs-array"][i]
            for charger_idx, charger_cfg in enumerate(evs.charger_cfgs[0, 0, :]):
                for attr in ["charge_mwh", "charge_binary"]:
                    results[f"{charger_cfg.name}-{attr}"].append(
                        sum([x.value() for x in getattr(evs, attr)[0, :, charger_idx]])
                    )
            for charge_event_idx, _ in enumerate(evs.charger_cfgs[0, :, 0]):
                for attr in ["charge_mwh"]:
                    results[f"spill-charge-event-{charge_event_idx}-{attr}"].append(
                        sum(
                            [
                                x.value()
                                for x in getattr(evs, attr)[0, charge_event_idx, :]
                            ]
                        )
                    )

    results = pd.DataFrame(results)

    #  add totals for charge events across both the spill and normal chargers
    if len(vars["evs-array"]):
        for charge_event_idx, _ in enumerate(interval_data.evs.charge_event_mwh):
            results[f"charge-event-{charge_event_idx}-total-charge_mwh"] = results[
                [
                    f"charge-event-{charge_event_idx}-charge_mwh",
                    f"spill-charge-event-{charge_event_idx}-charge_mwh",
                ]
            ].sum(axis=1)

    #  add totals
    #  can I do this without pandas??
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

    if interval_data.evs:
        for charge_event_idx, charge_event_mwh in enumerate(
            interval_data.evs.charge_event_mwh
        ):
            np.testing.assert_almost_equal(
                results[f"charge-event-{charge_event_idx}-total-charge_mwh"].sum(),
                charge_event_mwh,
                decimal=5,
            )
        """
        want to check
        - only one charger -> one charge event each interval
        """
        cols = [
            c
            for c in results.columns
            if c.startswith("charger-")
            and c.endswith("-charge_binary")
            and "spill" not in c
        ]
        subset = results[cols]
        assert (subset <= 1).all().all()
