import collections

import numpy as np
import pandas as pd
import pandera as pa
import pydantic
from rich import print

import energypylinear as epl
from energypylinear.defaults import defaults
from energypylinear.flags import Flags
from energypylinear.interval_data import IntervalData
from energypylinear.optimizer import Optimizer

optimizer = Optimizer()

def warn_spills(simulation: pd.DataFrame, flags: Flags) -> bool:
    """Prints warnings if we have spilled."""
    #  add warnings on the use of any spill asset
    spill_columns = [c for c in simulation.columns if "spill" in c]
    #  filter out binary columns - TODO separate loop while dev
    spill_columns = [c for c in spill_columns if "charge_binary" not in c]
    spill_results = simulation[spill_columns]
    assert isinstance(spill_results, pd.DataFrame)
    spill_occured = spill_results.sum().sum() > 0.0

    spills = spill_results.sum(axis=0).to_dict()
    spills = {k: v for k, v in spills.items() if v > 0}
    if spill_occured and flags.fail_on_spill_asset_use:
        spill_message = f"""
        Spill Occurred!
        {len(spills)} of {spill_results.shape[1]} spill columns
        {spills}
        """
        raise ValueError(spill_message)
    elif spill_occured:
        spill_message = f"""
        [red]Spill Occurred![/]
        {len(spills)} of {spill_results.shape[1]} spill columns
        {spills}
        """
        print(spill_message)
    return spill_occured


