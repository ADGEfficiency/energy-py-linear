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
import collections
import pathlib
import typing

import numpy as np
import pandas as pd
import pulp
import pydantic

import energypylinear as epl
from energypylinear import site
from energypylinear.assets.asset import AssetOneInterval
from energypylinear.defaults import defaults
from energypylinear.freq import Freq
from energypylinear.objectives import objectives
from energypylinear.optimizer import Optimizer


class ChargerConfig(pydantic.BaseModel):
    """Electric vehicle (EV) charger asset configuration."""

    name: str
    power_max_mw: float
    power_min_mw: float


class EVOneInterval(AssetOneInterval):
    """EV asset in a single interval"""

    #  could use `electric_charge_mwh`
    #  or event electric_load_mwh
    charge_mwh: pulp.LpVariable
    charge_binary: pulp.LpVariable

    class Config:
        arbitrary_types_allowed = True


class EVsArrayOneInterval(AssetOneInterval):
    """EV asset in a single interval, but with 2D array data."""

    #  could use `electric_charge_mwh`
    #  or event electric_load_mwh
    charge_mwh: np.ndarray
    charge_binary: np.ndarray
    charger_cfgs: np.ndarray

    class Config:
        arbitrary_types_allowed = True


def evs_one_interval(
    optimizer: Optimizer,
    chargers: np.ndarray,
    charge_event: np.ndarray,
    i: int,
    freq: Freq,
) -> tuple[list[EVOneInterval], EVsArrayOneInterval]:
    """Create a EV asset model for a single interval."""
    n_chargers = len(chargers)
    n_charge_events = charge_event.shape[1]

    evs = []
    charges_mwh = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    charges_binary = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    charger_cfgs = np.zeros((1, n_charge_events, n_chargers)).astype(object)

    for charge_event_idx in range(n_charge_events):
        for charger_idx, charger in enumerate(chargers):
            name = f"charge-event:{charge_event_idx},i:{i},charger:{charger.name}"
            charge_mwh = optimizer.continuous(
                f"charge_mwh,{name}", low=0, up=freq.mw_to_mwh(charger.power_max_mw)
            )
            charge_binary = optimizer.binary(f"charge_binary,{name}")
            evs.append(
                EVOneInterval(
                    charge_mwh=charge_mwh,
                    charge_binary=charge_binary,
                )
            )
            charges_mwh[0, charge_event_idx, charger_idx] = charge_mwh
            charges_binary[0, charge_event_idx, charger_idx] = charge_binary
            charger_cfgs[0, charge_event_idx, charger_idx] = charger

    evs_array = EVsArrayOneInterval(
        charge_mwh=charges_mwh, charge_binary=charges_binary, charger_cfgs=charger_cfgs
    )

    return evs, evs_array


def stack_ev(vars: dict, attr: str) -> np.ndarray:
    evs = np.concatenate(
        [getattr(v, attr) for v in vars["evs-array"]],
        axis=0,
    )
    spill = np.concatenate(
        [getattr(v, attr) for v in vars["spill-evs-array"]],
        axis=0,
    )
    return np.concatenate([evs, spill], axis=2)


def constrain_within_interval(
    optimizer: Optimizer,
    evs: EVsArrayOneInterval,
    charge_event: np.ndarray,
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
                continuous, binary, freq.mw_to_mwh(charger.power_max_mw)
            )
            optimizer.constrain_min(
                continuous, binary, freq.mw_to_mwh(charger.power_min_mw)
            )
            #  only let the binary be positive when the charge_event is positive
            #  this forces the charger to only charge during a charge event
            optimizer.constrain(binary <= charge_event[i, charge_event_idx])

    #  required to handle the spill charger case
    #  where we don't want to limit it
    if add_single_charger_or_event_constraints is True:

        #  constrain to only one charger per charging event
        #  sum across all chargers for one charge event <= 1
        for charge_event_idx in range(n_charge_events):
            optimizer.constrain(
                optimizer.sum(evs.charge_binary[0, charge_event_idx, :] <= 1)
            )
        #  constrain to only one charge event per charger
        #  sum across all charge events for one charger <= 1
        for charger_idx in range(n_chargers):
            optimizer.constrain(
                optimizer.sum(evs.charge_binary[0, :, charger_idx] <= 1)
            )


def constrain_after_intervals(
    optimizer: Optimizer,
    vars: dict,
    interval_data: "epl.interval_data.IntervalData",
    charger_cfgs: np.ndarray,
    spill_charger_config: np.ndarray,
) -> None:
    assert interval_data.evs

    stacked_charge_mwh = stack_ev(vars, "charge_mwh")

    #  check the stack worked correctly
    #  TODO move after interval data refactor
    stacked_charge_mwh = stack_ev(vars, "charge_mwh")
    assert stacked_charge_mwh.shape[0] == len(interval_data.idx)
    assert isinstance(interval_data.evs.charge_events, np.ndarray)
    assert stacked_charge_mwh.shape[1] == interval_data.evs.charge_events.shape[1]
    assert (
        stacked_charge_mwh.shape[2]
        == charger_cfgs.shape[0] + spill_charger_config.shape[0]
    )

    #  how would this constraint change with bi-directional?
    #  I think alot!
    #  perhaps better to just constrain the state of charge at the end of
    #  the charge event
    #  but this would require tracking the ev SOC
    #  which you would need to do as well
    #  TODO - revisit with bi-directional charging
    for charge_event_idx, required_charge_mwh in enumerate(
        interval_data.evs.charge_event_mwh
    ):
        optimizer.constrain(
            optimizer.sum(stacked_charge_mwh[:, charge_event_idx, :].flatten().tolist())
            == required_charge_mwh
        )


class EVs:
    def __init__(self, charger_mws: list[float], charger_turndown: float = 0.1):
        """
        EV asset class - handles optimization and plotting of results over many intervals.

        Args:
            charger_mws - size of EV chargers in mega-watts.
            charger_turndown - minimum charger output defined by the charger_turndown as a percent of the charger size in mega-watts.
        """

        self.charger_cfgs = np.array(
            [
                ChargerConfig(
                    name=f"charger-{name}",
                    power_max_mw=power_mw,
                    power_min_mw=power_mw * charger_turndown,
                )
                for name, power_mw in enumerate(charger_mws)
            ]
        )
        self.spill_charger_config = np.array(
            [
                ChargerConfig(
                    name="charger-spill",
                    power_max_mw=defaults.spill_charge_max_mw,
                    power_min_mw=0,
                )
            ]
        )

    def optimize(
        self,
        charge_events: typing.Union[list[list[int]], np.ndarray],
        charge_event_mwh: typing.Union[list[int], np.ndarray],
        electricity_prices,
        electricity_carbon_intensities=None,
        gas_prices=None,
        freq_mins: int = defaults.freq_mins,
        objective: str = "price",
    ) -> "epl.results.SimulationResult":
        """
        Optimize the EVs's dispatch using a mixed-integer linear program.

            Args:
            charge_events: 2D matrix representing when a charge event is active.

                Shape is (n_charge_events, n_timesteps).

                A charge events matrix for 4 charge events over 5 intervals.
                ```
                charge_events = [
                    [1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1],
                    [0, 1, 0, 0, 0],
                ]
                ```

            charge_event_mwh: 1D array of the total charge energy required for each charge event - total across all intervals in mega-watts.

                Length is the number of charge events.

                ```python
                charge_event_mwh = [50, 100, 30, 40]
                ```
            electricity_prices - the price of electricity in each interval.
            gas_prices - the prices of natural gas, used in CHP and boilers in each interval.
            electricity_carbon_intensities - carbon intensity of electricity in each interval.
            freq_mins - the size of an interval in minutes.
            objective - the optimization objective - either "price" or "carbon".
        """
        self.optimizer = Optimizer()
        freq = Freq(freq_mins)

        #  transpose charge_events to have time as first dimension
        #  this makes it more natural to select by the first dimension as the time dimension
        #  equivilant to the batch dimension when training deep learning
        charge_events = np.array(charge_events).T
        charge_event_mwh = np.array(charge_event_mwh)

        interval_data = epl.interval_data.IntervalData(
            electricity_prices=electricity_prices,
            gas_prices=gas_prices,
            electricity_carbon_intensities=electricity_carbon_intensities,
            evs=epl.interval_data.EVIntervalData(
                charge_events=charge_events,
                charge_event_mwh=charge_event_mwh,
            ),
        )
        assert interval_data.evs
        self.site_cfg = epl.site.SiteConfig()
        self.spill_cfg = epl.spill.SpillConfig()
        self.valve_cfg = epl.valve.ValveConfig(name="valve")

        #  TODO - difficult to type the list of list thing
        #  maybe sign something should be reworked
        vars: collections.defaultdict[str, typing.Any] = collections.defaultdict(list)
        for i in interval_data.idx:
            vars["sites"].append(
                site.site_one_interval(self.optimizer, self.site_cfg, i, freq)
            )
            vars["spills"].append(
                epl.spill.spill_one_interval(self.optimizer, self.spill_cfg, i, freq)
            )
            vars["valves"].append(
                epl.valve.valve_one_interval(self.optimizer, self.valve_cfg, i, freq)
            )

            evs, evs_array = evs_one_interval(
                self.optimizer,
                self.charger_cfgs,
                interval_data.evs.charge_events,
                i,
                freq,
            )
            spill_evs, spill_evs_array = evs_one_interval(
                self.optimizer,
                self.spill_charger_config,
                interval_data.evs.charge_events,
                i,
                freq,
            )
            vars["evs"].append(evs)
            vars["evs-array"].append(evs_array)
            vars["spill-evs"].append(spill_evs)
            vars["spill-evs-array"].append(spill_evs_array)
            vars["assets"].append([*evs, *spill_evs])

            site.constrain_within_interval(self.optimizer, vars, interval_data, i)
            constrain_within_interval(
                self.optimizer,
                evs_array,
                interval_data.evs.charge_events,
                freq,
                self.charger_cfgs,
                i,
            )
            constrain_within_interval(
                self.optimizer,
                spill_evs_array,
                interval_data.evs.charge_events,
                freq,
                self.spill_charger_config,
                i,
                add_single_charger_or_event_constraints=False,
            )

        assert isinstance(interval_data.evs.charge_events, np.ndarray)
        assert isinstance(self.charger_cfgs, np.ndarray)
        assert isinstance(self.spill_charger_config, np.ndarray)
        constrain_after_intervals(
            self.optimizer,
            vars,
            interval_data,
            self.charger_cfgs,
            self.spill_charger_config,
        )
        objective_fn = objectives[objective]
        self.optimizer.objective(objective_fn(self.optimizer, vars, interval_data))
        self.optimizer.solve()
        self.interval_data = interval_data
        return epl.results.extract_results(interval_data, vars)

    def plot(
        self,
        results: "epl.results.SimulationResult",
        path: typing.Union[pathlib.Path, str],
    ) -> None:
        return epl.plot.plot_evs(results, pathlib.Path(path))
