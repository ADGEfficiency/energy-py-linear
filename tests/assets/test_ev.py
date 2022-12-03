import typing

import numpy as np
import pydantic

from energypylinear.defaults import defaults
from energypylinear.freq import Freq
from energypylinear.optimizer import Pulp as Optimizer


class ChargerConfig(pydantic.BaseModel):
    name: str
    power_mw: float


class EVOneInterval(pydantic.BaseModel):
    #  could use `electric_charge_mwh`
    #  or event electric_load_mwh
    charge_mwh: np.ndarray
    charge_binary: np.ndarray

    class Config:
        arbitrary_types_allowed = True


def ev_one_interval(
    optimizer: Optimizer, chargers, charge_event, i, freq: Freq
) -> EVOneInterval:
    n_chargers = len(chargers)
    n_charge_events = charge_event.shape[1]

    charge_mwh = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    charge_binary = np.zeros((1, n_charge_events, n_chargers)).astype(object)

    for charge_event_idx in range(n_charge_events):
        for charger_idx, charger in enumerate(chargers):
            name = f"charge-event:{charge_event_idx},i:{i},{charger.name}"

            charge_mwh[0, charge_event_idx, charger_idx] = optimizer.continuous(
                f"charge_mwh,{name}", low=0, up=freq.mw_to_mwh(charger.power_mw)
            )
            charge_binary[0, charge_event_idx, charger_idx] = optimizer.binary(
                f"charge_binary,{name}"
            )

    return EVOneInterval(charge_mwh=charge_mwh, charge_binary=charge_binary)


def stack_ev(vars, attr):
    evs = np.concatenate(
        [getattr(v, attr) for v in vars["evs"]],
        axis=0,
    )
    spill = np.concatenate(
        [getattr(v, attr) for v in vars["spill-evs"]],
        axis=0,
    )
    return np.concatenate([evs, spill], axis=2)


def constrain_within_interval(
    optimizer: Optimizer,
    evs: list[EVOneInterval],
    charge_event,
    freq: Freq,
    chargers: np.ndarray,
    i: int,
    add_single_charger_or_event_constraints: bool = True,
) -> None:
    n_charge_events = charge_event.shape[1]
    n_chargers = chargers.shape[0]

    for charge_event_idx in range(n_charge_events):
        for charger_idx, charger in enumerate(chargers):
            #  add the connection between the contiunous and binary variables
            continuous = evs.charge_mwh[0, charge_event_idx, charger_idx]
            binary = evs.charge_binary[0, charge_event_idx, charger_idx]

            optimizer.constrain_max(
                continuous, binary, freq.mw_to_mwh(charger.power_mw)
            )
            #  only let the binary be positive when the charge_event is positive
            #  this forces the charger to only charge during a charge event
            optimizer.constrain(binary <= charge_event[i, charge_event_idx])

    #  required to handle the spill charger case
    #  where we don't want to limit it
    if add_single_charger_or_event_constraints:
        #  constrain to only one charger per charging event
        #  sum across all chargers for one charge event <= 1
        for charge_event_idx in range(n_charge_events):
            optimizer.constrain(
                optimizer.sum(evs.charge_binary[0, charge_event_idx, :]) <= 1
            )
        #  constrain to only one charge event per charger
        #  sum across all charge events for one charger <= 1
        for charger_idx in range(n_chargers):
            optimizer.constrain(
                optimizer.sum(evs.charge_binary[0, :, charger_idx]) <= 1
            )

    #  TODO perhaps could do these in the loop above?


# def test_ev_one_interval():

chargers = np.array(
    [
        ChargerConfig(name="charger-0", power_mw=2),
        ChargerConfig(name="charger-1", power_mw=4),
    ]
)

spill_chargers = np.array(
    [ChargerConfig(name="charger-spill", power_mw=defaults.spill_charge_max_mw)]
)

#  transpose to have time as first dimension
charge_event = np.array(
    [
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1],
    ]
).T
charge_event_mwh = np.array([50, 100, 25])
idx = np.arange(charge_event.shape[0])

#  TODO move to interval data validation
assert idx.shape[0] == charge_event.shape[0]
assert charge_event.shape[1] == charge_event_mwh.shape[0]


def constrain_after_intervals(
    optimizer: Optimizer, vars: dict, charge_event_mwh
) -> None:
    stacked_charge_mwh = stack_ev(vars, "charge_mwh")

    #  how would this constraint change with bi-directional?
    #  I think alot!
    #  perhaps better to just constrain the state of charge at the end of
    #  the charge event
    #  but this would require tracking the ev SOC
    #  which you would need to do as well
    #  TODO - revisit with bi-directional charging
    for charge_event_idx, required_charge_mwh in enumerate(charge_event_mwh):
        optimizer.constrain(
            optimizer.sum(stacked_charge_mwh[:, charge_event_idx, :].flatten().tolist())
            == required_charge_mwh
        )


import collections

freq = Freq(30)
optimizer = Optimizer()

vars = collections.defaultdict(list)
for i in idx:
    vars["evs"].append(ev_one_interval(optimizer, chargers, charge_event, i, freq))
    vars["spill-evs"].append(
        ev_one_interval(
            optimizer,
            spill_chargers,
            charge_event,
            i,
            freq,
        )
    )

    constrain_within_interval(
        optimizer,
        vars["evs"][i],
        charge_event,
        freq,
        chargers,
        i,
    )
    constrain_within_interval(
        optimizer,
        vars["spill-evs"][i],
        charge_event,
        freq,
        spill_chargers,
        i,
        add_single_charger_or_event_constraints=False,
    )

constrain_after_intervals(optimizer, vars, charge_event_mwh)

#  check the stack worked correctly
#  TODO move after interval data refactor
stacked_charge_mwh = stack_ev(vars, "charge_mwh")
assert stacked_charge_mwh.shape[0] == len(idx)
assert stacked_charge_mwh.shape[1] == charge_event.shape[1]
assert stacked_charge_mwh.shape[2] == chargers.shape[0] + spill_chargers.shape[0]
out = stacked_charge_mwh
