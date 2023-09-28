"""Warnings for results module."""
import pandas as pd

from energypylinear.flags import Flags
from energypylinear.logger import logger
from energypylinear.optimizer import Optimizer

optimizer = Optimizer()


def warn_spills(simulation: pd.DataFrame, flags: Flags, verbose: bool = True) -> bool:
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
    elif spill_occured and verbose:
        logger.warning(
            "warn_spills",
            n_spills=len(spills),
            spill_columns=spill_results.shape[1],
            spills=spills,
        )
    return spill_occured
