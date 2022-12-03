"""
Electric vehicle models

Both smart and on-demand charging

- On demand charging is modelled by reducing the charge event length,
- Smart charging / queuing is modelled by having a large charge event length.

In future do V2G

- V2G / bi-directional charging modelled by allowing charge_mwh to be positive and negative
- bi-directional charging would require tracking the capacity of the car, which would require an additional input of `Car` with a capacity

# Input Data

chargers

- power_mw

charge_event_mwh

- list of floats
- n charge events
- mwh of charge required for that charge event

charge_events

- 2D array of binary integers
- n charge events * m intervals
- 0 = charger disconnected, 1 = charger connected

cars

- for bi-directional only
- capacity_mwh
- leave to future

# Variables

charge_mwh

- actual charging in each interval
- 2D array of floats, same shape as charge_event_binary
- n charge events * n_chargers * m intervals
- should I try to make this 3D?

charge_binary

- same shape as charge_mwh
- n charge events * n_chargers * m intervals
- linked to the charge_mwh via a min and/or max constraint

Example - 2 chargers, 3 charge events, 4 intervals

charger_0, charge_event_0, [0, 0, 0, 0]
charger_0, charge_event_1, [0, 0, 0, 0]
charger_0, charge_event_2, [0, 0, 0, 0]
charger_1, charge_event_0, [0, 0, 0, 0]
charger_1, charge_event_1, [0, 0, 0, 0]
charger_1, charge_event_2, [0, 0, 0, 0]

# Constraints

constrain each charge event to get all the charge it needs
- energy balance constraint
- charge_mwh.sum(time) == charge_event_mwh

one charger can only be attached to one charge event at a time

one charge event can only have one attached charger at a time

only one charger can be used for one charge event across all times
- this prevents chargers switching during charge events

# Objective

minimum energy cost
- energy cost = charge_mwh * electricity_price

incentivize early charging
- want to reduce the cost to charge earlier on
- could just reduce prices earlier in the intervals???
- perhaps quantify as `value_of_time` ?
- interval_length * value_per_interval
- if charged 30 min earlier -> some benefit

could perhaps just add something like the below to each interval:
- [5, 10, 15, 20, 25, 30]
- and multiply it by the binary variable?
- 5 = `value_of_interval`

could maybe test by reversing the value_of_interval
"""
import typing

import numpy as np
import pydantic

import energypylinear as epl
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
    evs: EVOneInterval,
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


class EVs:
    def __init__(self, charger_mws: list[float]):

        self.charger_cfgs = np.array(
            [
                ChargerConfig(name=f"charger-{name}", power_mw=power_mw)
                for name, power_mw in enumerate(charger_mws)
            ]
        )
        self.spill_charger_config = np.array(
            [ChargerConfig(name="charger-spill", power_mw=defaults.spill_charge_max_mw)]
        )

        self.optimizer = Optimizer()

    def optimize(
        self,
        charge_events: typing.Union[list[list[int]], np.ndarray],
        charge_event_mwh: typing.Union[list[int], np.ndarray],
        electricity_prices,
        freq_mins: int = defaults.freq_mins,
    ):
        freq = Freq(freq_mins)

        #  transpose to have time as first dimension
        charge_events = np.array(charge_events).T
        charge_event_mwh = np.array(charge_event_mwh)

        interval_data = epl.data.IntervalData(
            electricity_prices=electricity_prices,
            evs=epl.data.EVIntervalData(
                charge_events=charge_events,
                charge_event_mwh=charge_event_mwh,
            ),
        )
