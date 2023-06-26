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


class ChargeEventConfig(pydantic.BaseModel):
    """Electric vehicle (EV) single charge event configuration.

    Capacity here refers to the battery capacity of the vehicle being charged.
    """

    name: str
    capacity_mwh: float
    efficiency_pct: float


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

    electric_charge_mwh: pulp.LpVariable
    electric_charge_binary: pulp.LpVariable
    electric_discharge_mwh: pulp.LpVariable | int
    electric_discharge_binary: pulp.LpVariable | int
    initial_soc_mwh: pulp.LpVariable
    final_soc_mwh: pulp.LpVariable
    electric_loss_mwh: pulp.LpVariable

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed = True


class EVsArrayOneInterval(AssetOneInterval):
    """EV asset for a single interval as a 2D array.."""

    electric_charge_mwh: np.ndarray
    electric_charge_binary: np.ndarray
    electric_discharge_mwh: np.ndarray
    electric_discharge_binary: np.ndarray
    charger_cfgs: np.ndarray
    charge_event_cfgs: np.ndarray
    initial_soc_mwh: np.ndarray
    final_soc_mwh: np.ndarray
    electric_losses_mwh: np.ndarray

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed = True


def evs_one_interval(
    optimizer: Optimizer,
    charger_cfgs: np.ndarray,
    charge_event_cfgs: np.ndarray,
    i: int,
    freq: Freq,
    flags: Flags = Flags(),
) -> tuple[list[EVOneInterval], EVsArrayOneInterval]:
    """Create EV asset data for a single interval.

    This data is a collection of linear programming variables for the EV charge, discharge and state of charges.

    The data is represented as a list of EVOneInterval objects and a single EVsArrayOneInterval.

    The list representation is used in the `vars` lists, which ends up in the objective function.

    The array representation allows easy selection of all chargers or all charge events.
    """
    n_chargers = len(charger_cfgs)
    n_charge_events = len(charge_event_cfgs)

    evs = []
    charges_mwh = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    charges_binary = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    discharges_mwh = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    discharges_binary = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    initial_socs_mwh = np.zeros((1, n_charge_events)).astype(object)
    final_socs_mwh = np.zeros((1, n_charge_events)).astype(object)
    losses_mwh = np.zeros((1, n_charge_events, n_chargers)).astype(object)

    """
    there is oppourtunity here to prune the number of variables
    - when the charge event binary is 0 for a charger, charge_event pair,
    - when the charge event is over (ie the last charge event)

    leave this as a fun project for later
    """

    for charge_event_idx, charge_event_cfg in enumerate(charge_event_cfgs):

        initial_charge = optimizer.continuous(
            f"initial_soc_mwh,i:{i},charge_event:{charge_event_cfg.name}",
            low=0,
            up=charge_event_cfg.capacity_mwh,
        )
        final_charge = optimizer.continuous(
            f"final_soc_mwh,i:{i},charge_event:{charge_event_cfg.name}",
            low=0,
            up=charge_event_cfg.capacity_mwh,
        )
        initial_socs_mwh[0, charge_event_idx] = initial_charge
        final_socs_mwh[0, charge_event_idx] = final_charge

        for charger_idx, charger_cfg in enumerate(charger_cfgs):
            name = f"charge-event:{charge_event_idx},i:{i},charger:{charger_cfg.name}"
            charge_mwh = optimizer.continuous(
                f"charge_mwh,{name}", low=0, up=freq.mw_to_mwh(charger_cfg.power_max_mw)
            )
            charge_binary = optimizer.binary(f"charge_binary,{name}")
            discharge_mwh = (
                optimizer.continuous(
                    f"discharge_mwh,{name}",
                    low=0,
                    up=freq.mw_to_mwh(charger_cfg.power_max_mw),
                )
                if flags.allow_evs_discharge
                else 0
            )
            discharge_binary = (
                optimizer.binary(f"discharge_binary,{name}")
                if flags.allow_evs_discharge
                else 0
            )
            loss_mwh = optimizer.continuous(
                f"losses_mwh,{name}", low=0, up=freq.mw_to_mwh(charger_cfg.power_max_mw)
            )

            evs.append(
                EVOneInterval(
                    electric_charge_mwh=charge_mwh,
                    electric_charge_binary=charge_binary,
                    electric_discharge_mwh=discharge_mwh,
                    electric_discharge_binary=discharge_binary,
                    initial_soc_mwh=initial_charge,
                    final_soc_mwh=final_charge,
                    electric_loss_mwh=loss_mwh,
                )
            )
            charges_mwh[0, charge_event_idx, charger_idx] = charge_mwh
            charges_binary[0, charge_event_idx, charger_idx] = charge_binary
            discharges_mwh[0, charge_event_idx, charger_idx] = charge_mwh
            discharges_binary[0, charge_event_idx, charger_idx] = charge_binary
            losses_mwh[0, charge_event_idx, charger_idx] = charge_mwh

    evs_array = EVsArrayOneInterval(
        electric_charge_mwh=charges_mwh,
        electric_charge_binary=charges_binary,
        electric_discharge_mwh=discharges_mwh,
        electric_discharge_binary=discharges_binary,
        charger_cfgs=charger_cfgs,
        charge_event_cfgs=charge_event_cfgs,
        initial_soc_mwh=initial_socs_mwh,
        final_soc_mwh=final_socs_mwh,
        electric_losses_mwh=losses_mwh,
    )

    return evs, evs_array


def stack_ev(vars: dict, attr: str) -> np.ndarray:
    """Stack electric vehicle charge and charge_event data."""
    # evs = np.array([getattr(v, attr) for v in vars["evs-array"]]).reshape(-1, 1)
    # spill = np.array([getattr(v, attr) for v in vars["spill-evs-array"]]).reshape(-1, 1)
    evs = np.concatenate(
        [getattr(v, attr) for v in vars["evs-array"]],
        axis=0,
    )
    spill = np.concatenate(
        [getattr(v, attr) for v in vars["spill-evs-array"]],
        axis=0,
    )
    return np.concatenate([evs, spill], axis=2)


def constrain_charge_discharge(
    optimizer: Optimizer,
    evs: EVsArrayOneInterval,
    charge_event: np.ndarray,
    freq: Freq,
    charger_cfgs: np.ndarray,
    i: int,
    add_single_charger_or_event_constraints: bool = True,
) -> None:
    """Constrain chargers during one interval."""
    n_charge_events = charge_event.shape[1]
    n_chargers = charger_cfgs.shape[0]

    for charge_event_idx in range(n_charge_events):
        for charger_idx, charger in enumerate(charger_cfgs):
            #  add the connection between the continuous and binary variables
            for continuous, binary in [
                (
                    evs.electric_charge_mwh[0, charge_event_idx, charger_idx],
                    evs.electric_charge_binary[0, charge_event_idx, charger_idx],
                ),
                (
                    evs.electric_discharge_mwh[0, charge_event_idx, charger_idx],
                    evs.electric_discharge_binary[0, charge_event_idx, charger_idx],
                ),
            ]:
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
                optimizer.sum(
                    evs.electric_charge_binary[0, charge_event_idx, :].tolist()
                    + evs.electric_discharge_binary[0, charge_event_idx, :].tolist()
                )
                <= 1
            )
        #  constrain to only one charge event per charger
        #  sum across all charge events for one charger <= 1
        for charger_idx in range(n_chargers):
            optimizer.constrain(
                optimizer.sum(
                    evs.electric_charge_binary[0, :, charger_idx].tolist()
                    + evs.electric_discharge_binary[0, :, charger_idx].tolist()
                )
                <= 1
            )


def constrain_charge_event_electricity_balance(
    optimizer: Optimizer, vars: collections.defaultdict, i: int
) -> None:
    """ """
    evs_array = vars["evs-array"][i]
    for charge_event_idx, charge_event_cfg in enumerate(evs_array.charge_event_cfgs):
        optimizer.constrain(
            evs_array.initial_soc_mwh[0, charge_event_idx]
            + optimizer.sum(evs_array.electric_charge_mwh[0, charge_event_idx])
            - optimizer.sum(evs_array.electric_discharge_mwh[0, charge_event_idx])
            - optimizer.sum(evs_array.electric_losses_mwh[0, charge_event_idx])
            == optimizer.sum(evs_array.final_soc_mwh[0, charge_event_idx])
        )

        optimizer.constrain(
            optimizer.sum(evs_array.electric_charge_mwh[0, charge_event_idx])
            * (1 - charge_event_cfg.efficiency_pct)
            == optimizer.sum(evs_array.electric_losses_mwh[0, charge_event_idx])
        )


def constrain_connection_charge_events_between_intervals(
    optimizer: Optimizer, vars: collections.defaultdict, i: int
) -> None:
    if i == 0:
        return None

    all_evs_array = vars["evs-array"]
    old = all_evs_array[-2]
    new = all_evs_array[-1]
    for initial_soc, final_soc in zip(
        new.initial_soc_mwh[0], old.final_soc_mwh[0], strict=True
    ):
        optimizer.constrain(initial_soc == final_soc)


def constrain_initial_final_charge(
    optimizer: Optimizer,
    vars: dict,
    interval_data: "epl.interval_data.IntervalData",
    charger_cfgs: np.ndarray,
    spill_charger_config: np.ndarray,
    charge_event_cfgs: np.ndarray,
) -> None:
    """Constrain state of charge at the beginning and end of the simulation."""
    assert interval_data.evs

    # stacked_charge_mwh = stack_ev(vars, "electric_charge_mwh")

    # #  check the stack worked correctly
    # assert stacked_charge_mwh.shape[0] == len(interval_data.idx)
    # assert isinstance(interval_data.evs.charge_events, np.ndarray)
    # assert stacked_charge_mwh.shape[1] == interval_data.evs.charge_events.shape[1]
    # assert (
    #     stacked_charge_mwh.shape[2]
    #     == charger_cfgs.shape[0] + spill_charger_config.shape[0]
    # )

    for charge_event_idx, charge_event_cfg in enumerate(charge_event_cfgs):
        #  constrain the intial soc
        optimizer.constrain(
            vars["evs-array"][0].initial_soc_mwh[0, charge_event_idx] == 0
        )

        #  constrain the final soc
        optimizer.constrain(
            vars["evs-array"][-1].final_soc_mwh[0, charge_event_idx]
            == charge_event_cfg.capacity_mwh
        )

    #  intentionally don't constrain the spill charger


class EVs:
    """EV asset class - handles optimization and plotting of results over many intervals.

    Args:
        charger_mws - size of EV chargers in mega-watts.
        charger_turndown - minimum charger output defined by the charger_turndown as a percent of the charger size in mega-watts.
    """

    def __init__(
        self,
        chargers_power_mw: list[float],
        charge_events_capacity_mwh: list[float],
        charge_event_efficiency: float = 0.9,
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
                for name, power_mw in enumerate(chargers_power_mw)
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
        self.charge_event_cfgs = np.array(
            [
                ChargeEventConfig(
                    name=f"charge-event-{name}",
                    capacity_mwh=capacity_mwh,
                    efficiency_pct=charge_event_efficiency,
                )
                for name, capacity_mwh in enumerate(charge_events_capacity_mwh)
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
            self.charge_event_cfgs,
            i,
            freq,
            flags=flags,
        )
        spill_evs, spill_evs_array = evs_one_interval(
            optimizer,
            self.spill_charger_config,
            self.charge_event_cfgs,
            i,
            freq,
            flags=flags,
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
        #  TODO spill charger
        constrain_charge_discharge(
            optimizer,
            evs_array,
            self.interval_data.evs.charge_events,
            freq,
            self.charger_cfgs,
            i,
        )
        spill_evs_array = vars["spill-evs-array"][i]
        constrain_charge_discharge(
            optimizer,
            spill_evs_array,
            self.interval_data.evs.charge_events,
            freq,
            self.spill_charger_config,
            i,
            add_single_charger_or_event_constraints=False,
        )

        constrain_charge_event_electricity_balance(optimizer, vars, i)
        # constrain_connection_charge_events_between_intervals(optimizer, vars, i)

    def constrain_after_intervals(
        self,
        optimizer: Optimizer,
        vars: collections.defaultdict,
        interval_data: "epl.IntervalData",
    ) -> None:
        """Constrain EVs after all interval asset models are created."""
        #  TODO spill charger
        constrain_initial_final_charge(
            optimizer,
            vars,
            interval_data,
            self.charger_cfgs,
            self.spill_charger_config,
            self.charge_event_cfgs,
        )

    def optimize(
        self,
        charge_events: typing.Union[list[list[int]], np.ndarray],
        electricity_prices: typing.Union[np.ndarray, list[float]],
        gas_prices: typing.Union[None, np.ndarray, float] = None,
        electricity_carbon_intensities: typing.Union[
            None, np.ndarray, list[float], float
        ] = None,
        freq_mins: int = defaults.freq_mins,
        objective: str = "price",
        verbose: int = 0,
        flags: Flags = Flags(),
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

        def validate_charge_events(charge_event_cfgs, charge_events):
            assert charge_event_cfgs.shape[0] == charge_events.shape[1]

        validate_charge_events(self.charge_event_cfgs, charge_events)

        interval_data = epl.interval_data.IntervalData(
            electricity_prices=electricity_prices,
            gas_prices=gas_prices,
            electricity_carbon_intensities=electricity_carbon_intensities,
            evs=epl.interval_data.EVIntervalData(
                charge_events=charge_events,
                charge_events_capacity_mwh=[
                    cfg.capacity_mwh for cfg in self.charge_event_cfgs
                ],
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
                self.optimizer, i, freq, flags=flags
            )

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
                self.optimizer, vars, interval_data, i, freq=freq, flags=flags
            )

        assert isinstance(interval_data.evs.charge_events, np.ndarray)
        assert isinstance(self.charger_cfgs, np.ndarray)
        assert isinstance(self.spill_charger_config, np.ndarray)
        self.constrain_after_intervals(
            self.optimizer,
            vars,
            interval_data,
        )

        objective_fn = epl.objectives[objective]
        self.optimizer.objective(objective_fn(self.optimizer, vars, interval_data))
        status = self.optimizer.solve(verbose=verbose)
        self.interval_data = interval_data
        return epl.results.extract_results(
            interval_data, vars, feasible=status.feasible, flags=flags
        )

    def plot(
        self,
        results: "epl.results.SimulationResult",
        path: typing.Union[pathlib.Path, str],
    ) -> None:
        """Plot simulation results."""
        return epl.plot.plot_evs(results, pathlib.Path(path))
