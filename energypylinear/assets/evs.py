"""Electric vehicle asset for optimizing the smart charging of electric vehicles."""
import collections
import pathlib
import typing

import numpy as np
import pulp
import pydantic

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval
from energypylinear.defaults import defaults
from energypylinear.flags import Flags
from energypylinear.freq import Freq
from energypylinear.optimizer import Optimizer


class ChargerConfig(pydantic.BaseModel):
    """Electric vehicle (EV) single charger asset configuration."""

    name: str
    power_max_mw: float
    power_min_mw: float


class EVsConfig(pydantic.BaseModel):
    """Electric vehicle (EV) asset configuration."""

    name: str

    @pydantic.validator("name")
    def check_name(cls, name: str) -> str:
        """Ensure we can identify this asset correctly."""
        assert "evs" in name
        return name


class EVOneInterval(AssetOneInterval):
    """EV asset data for a single interval"""

    #  could use `electric_charge_mwh`
    #  or even electric_load_mwh
    charge_mwh: pulp.LpVariable
    charge_binary: pulp.LpVariable

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed = True


class EVsArrayOneInterval(AssetOneInterval):
    """EV asset for a single interval as a 2D array.."""

    #  could use `electric_charge_mwh`
    #  or even electric_load_mwh
    charge_mwh: np.ndarray
    charge_binary: np.ndarray
    charger_cfgs: np.ndarray

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed = True


def evs_one_interval(
    optimizer: Optimizer,
    chargers: np.ndarray,
    charge_event: np.ndarray,
    i: int,
    freq: Freq,
) -> tuple[list[EVOneInterval], EVsArrayOneInterval]:
    """Create EV asset data for a single interval."""
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
    """Stack electric vehicle charge and charge_event data."""
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
    """Constrain chargers during one interval."""
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
    #  where we don't want to limit the matching of chargers and charge events
    if add_single_charger_or_event_constraints is True:

        #  constrain to only one charger per charging event
        #  sum across all chargers for one charge event <= 1
        for charge_event_idx in range(n_charge_events):
            optimizer.constrain(
                optimizer.sum(evs.charge_binary[0, charge_event_idx, :].tolist()) <= 1
            )
        #  constrain to only one charge event per charger
        #  sum across all charge events for one charger <= 1
        for charger_idx in range(n_chargers):
            optimizer.constrain(
                optimizer.sum(evs.charge_binary[0, :, charger_idx].tolist()) <= 1
            )


def constrain_after_intervals(
    optimizer: Optimizer,
    vars: dict,
    interval_data: "epl.interval_data.IntervalData",
    charger_cfgs: np.ndarray,
    spill_charger_config: np.ndarray,
) -> None:
    """Constrain chargers after all intervals over."""
    assert interval_data.evs

    stacked_charge_mwh = stack_ev(vars, "charge_mwh")

    #  check the stack worked correctly
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
    """EV asset class - handles optimization and plotting of results over many intervals.

    Args:
        charger_mws - size of EV chargers in mega-watts.
        charger_turndown - minimum charger output defined by the charger_turndown as a percent of the charger size in mega-watts.
    """

    def __init__(
        self,
        charger_mws: list[float],
        charger_turndown: float = 0.1,
        name: str = "evs",
    ):
        """Initialize an electric vehicle asset model."""

        self.cfg = EVsConfig(name=name)

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

    def one_interval(
        self, optimizer: Optimizer, i: int, freq: Freq, flags: Flags = Flags()
    ) -> tuple:
        """Create EV asset data for a single interval."""
        assert self.interval_data.evs is not None
        assert self.interval_data.evs.charge_events is not None

        evs, evs_array = evs_one_interval(
            optimizer,
            self.charger_cfgs,
            np.array(self.interval_data.evs.charge_events),
            i,
            freq,
        )
        spill_evs, spill_evs_array = evs_one_interval(
            optimizer,
            self.spill_charger_config,
            np.array(self.interval_data.evs.charge_events),
            i,
            freq,
        )
        return evs, evs_array, spill_evs, spill_evs_array

    def constrain_within_interval(
        self,
        optimizer: Optimizer,
        vars: collections.defaultdict,
        interval_data: "epl.IntervalData",
        i: int,
        freq: Freq,
        flags: Flags = Flags(),
    ) -> None:
        """Constrain EVs dispatch within a single interval"""
        evs_array = vars["evs-array"][i]
        assert self.interval_data.evs is not None
        constrain_within_interval(
            optimizer,
            evs_array,
            self.interval_data.evs.charge_events,
            freq,
            self.charger_cfgs,
            i,
        )
        spill_evs_array = vars["spill-evs-array"][i]
        constrain_within_interval(
            optimizer,
            spill_evs_array,
            self.interval_data.evs.charge_events,
            freq,
            self.spill_charger_config,
            i,
            add_single_charger_or_event_constraints=False,
        )

    def constrain_after_intervals(
        self,
        optimizer: Optimizer,
        vars: collections.defaultdict,
        interval_data: "epl.IntervalData",
    ) -> None:
        """Constrain EVs after all interval asset models are created."""
        constrain_after_intervals(
            optimizer, vars, interval_data, self.charger_cfgs, self.spill_charger_config
        )

    def optimize(
        self,
        charge_events: typing.Union[list[list[int]], np.ndarray],
        charge_event_mwh: typing.Union[list[int], np.ndarray],
        electricity_prices: typing.Union[np.ndarray, list[float]],
        gas_prices: typing.Union[None, np.ndarray, float] = None,
        electricity_carbon_intensities: typing.Union[
            None, np.ndarray, list[float], float
        ] = None,
        freq_mins: int = defaults.freq_mins,
        objective: str = "price",
        verbose: int = 0,
    ) -> "epl.results.SimulationResult":
        """Optimize the EVs's dispatch using a mixed-integer linear program.

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
                ```
                charge_event_mwh = [50, 100, 30, 40]
                ```

            electricity_prices - the price of electricity in each interval.
            gas_prices - the prices of natural gas, used in CHP and boilers in each interval.
            electricity_carbon_intensities - carbon intensity of electricity in each interval.
            freq_mins - the size of an interval in minutes.
            objective - the optimization objective - either "price" or "carbon".
            verbose: level of printing.

        Returns:
            epl.results.SimulationResult
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
        self.site = epl.Site()
        self.spill = epl.spill.Spill()

        #  needed in self.one_interval
        self.interval_data = interval_data

        vars: collections.defaultdict[str, typing.Any] = collections.defaultdict(list)
        for i in interval_data.idx:
            vars["sites"].append(
                self.site.one_interval(self.optimizer, self.site.cfg, i, freq)
            )
            assert isinstance(interval_data.evs.charge_events, np.ndarray)
            evs, evs_array, spill_evs, spill_evs_array = self.one_interval(
                self.optimizer,
                i,
                freq,
            )

            #  should get rid of all these and just use `assets`
            #  reason to use the separate keys is that we don't want
            #  both evs_array and evs in assets for epl.Site
            vars["evs-array"].append(evs_array)
            vars["spill-evs-array"].append(spill_evs_array)
            vars["assets"].append(
                [
                    *evs,
                    *spill_evs,
                    self.spill.one_interval(self.optimizer, i, freq),
                ]
            )

            self.site.constrain_within_interval(self.optimizer, vars, interval_data, i)

            self.constrain_within_interval(
                self.optimizer, vars, interval_data, i, freq=freq
            )

        assert isinstance(interval_data.evs.charge_events, np.ndarray)
        assert isinstance(self.charger_cfgs, np.ndarray)
        assert isinstance(self.spill_charger_config, np.ndarray)
        self.constrain_after_intervals(
            self.optimizer,
            vars,
            interval_data,
            # self.charger_cfgs,
            # self.spill_charger_config,
        )

        objective_fn = epl.objectives[objective]
        self.optimizer.objective(objective_fn(self.optimizer, vars, interval_data))
        status = self.optimizer.solve(verbose=verbose)
        self.interval_data = interval_data
        return epl.results.extract_results(
            interval_data, vars, feasible=status.feasible
        )

    def plot(
        self,
        results: "epl.results.SimulationResult",
        path: typing.Union[pathlib.Path, str],
    ) -> None:
        """Plot simulation results."""
        return epl.plot.plot_evs(results, pathlib.Path(path))
