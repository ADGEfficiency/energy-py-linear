"""Electric vehicle asset for optimizing the smart charging of electric vehicles."""
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


def validate_charge_events(
    charge_event_cfgs: np.ndarray, charge_events: np.ndarray
) -> None:
    """Helper used to handle charge events.

    This is a bit of a smell - will probably be reworked once I see a way.
    """
    assert charge_event_cfgs.shape[0] == charge_events.shape[1]


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
    charger_cfgs: np.ndarray
    spill_charger_cfgs: np.ndarray
    charge_event_cfgs: np.ndarray

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed = True

    @pydantic.validator("name")
    def check_name(cls, name: str) -> str:
        """Ensure we can identify this asset correctly."""
        assert "evs" in name
        return name


class EVOneInterval(AssetOneInterval):
    """EV asset data for a single interval.

    This single data structure contains data for one asset
    in one interval.
    """

    cfg: EVsConfig

    electric_charge_mwh: pulp.LpVariable | float
    electric_charge_binary: pulp.LpVariable | int
    electric_discharge_mwh: pulp.LpVariable | float
    electric_discharge_binary: pulp.LpVariable | int
    initial_soc_mwh: pulp.LpVariable | float
    final_soc_mwh: pulp.LpVariable | float
    electric_loss_mwh: pulp.LpVariable

    is_spill: bool = False

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed = True


class EVsArrayOneInterval(AssetOneInterval):
    """EV asset for a single interval as a 1D, 2D & 3D arrays.

    This single data structure contains all the data for all assets
    in one interval.
    """

    i: int
    cfg: EVsConfig
    is_spill: bool = False

    initial_soc_mwh: np.ndarray
    final_soc_mwh: np.ndarray
    electric_charge_mwh: np.ndarray
    electric_charge_binary: np.ndarray
    electric_discharge_mwh: np.ndarray
    electric_discharge_binary: np.ndarray
    electric_loss_mwh: np.ndarray

    class Config:
        """pydantic.BaseModel configuration."""

        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<EVsArrayOneInterval i: {self.i} chargers: {len(self.cfg.charger_cfgs)} charge-events: {len(self.cfg.charge_event_cfgs)}>"


def evs_one_interval(
    optimizer: Optimizer,
    cfg: EVsConfig,
    charger_cfgs: np.ndarray,
    charge_event_cfgs: np.ndarray,
    charge_events: np.ndarray,
    i: int,
    freq: Freq,
    asset_name: str,
    flags: Flags = Flags(),
    create_charge_event_soc: bool = True,
    create_discharge_variables: bool = True,
    is_spill: bool = False,
) -> tuple[list[EVOneInterval], EVsArrayOneInterval]:
    """Create EV asset data for a single interval.

    Create a list of `EVOneInterval` and one `EVsArrayOneInterval` data structures.

    This data is a collection of linear programming variables for the EV charge, discharge and state of charges.

    The list is used in the `vars` lists, which ends up in the objective function.

    The array representation is used in constructing the linear program.
    """
    n_chargers = len(charger_cfgs)
    n_charge_events = len(charge_event_cfgs)

    #  1D array
    evs = []

    #  2D arrays
    initial_socs_mwh = np.zeros((1, n_charge_events)).astype(object)
    final_socs_mwh = np.zeros((1, n_charge_events)).astype(object)

    #  3D arrays
    charges_mwh = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    charges_binary = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    discharges_mwh = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    discharges_binary = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    losses_mwh = np.zeros((1, n_charge_events, n_chargers)).astype(object)

    """
    there is opportunity here to prune the number of variables
    - when the charge event binary is 0 for a charger, charge_event pair,
    - when the charge event is over (ie the last charge event)

    leave this as a fun project for later

    when the charge event binary is 0 for a charger, charge_event pair,
    - this seems like the real profitable one
    - most of the time the charge event binary is 0
    """

    for charge_event_idx, charge_event_cfg in enumerate(charge_event_cfgs):
        if create_charge_event_soc:
            name = f"i:{i},asset:{asset_name},charge_event:{charge_event_cfg.name}"
            initial_charge = optimizer.continuous(
                f"initial_soc_mwh,{name}",
                low=0,
                up=charge_event_cfg.capacity_mwh,
            )
            final_charge = optimizer.continuous(
                f"final_soc_mwh,{name}",
                low=0,
                up=charge_event_cfg.capacity_mwh,
            )
        else:
            initial_charge = 0
            final_charge = 0

        initial_socs_mwh[0, charge_event_idx] = initial_charge
        final_socs_mwh[0, charge_event_idx] = final_charge

        for charger_idx, charger_cfg in enumerate(charger_cfgs):
            name = f"i:{i},asset:{asset_name},charge-event:{charge_event_idx},charger:{charger_cfg.name}"

            if flags.limit_charge_variables_to_valid_events:
                charge_event = charge_events[i, charge_event_idx]
                charge_mwh = (
                    optimizer.continuous(
                        f"electric_charge_mwh,{name}",
                        low=0,
                        up=freq.mw_to_mwh(charger_cfg.power_max_mw),
                    )
                    if charge_event == 1
                    else 0
                )
                charge_binary = (
                    optimizer.binary(f"electric_charge_binary,{name}")
                    if charge_event == 1
                    else 0
                )

            else:
                charge_mwh = optimizer.continuous(
                    f"electric_charge_mwh,{name}",
                    low=0,
                    up=freq.mw_to_mwh(charger_cfg.power_max_mw),
                )
                charge_binary = optimizer.binary(f"electric_charge_binary,{name}")

            discharge_mwh = 0
            discharge_binary = 0
            if create_discharge_variables:
                discharge_mwh = (
                    optimizer.continuous(
                        f"electric_discharge_mwh,{name}",
                        low=0,
                        up=freq.mw_to_mwh(charger_cfg.power_max_mw),
                    )
                    if flags.allow_evs_discharge
                    else 0
                )
                discharge_binary = (
                    optimizer.binary(f"electric_discharge_binary,{name}")
                    if flags.allow_evs_discharge
                    else 0
                )

            loss_mwh = optimizer.continuous(
                f"electric_loss_mwh,{name}",
                low=0,
            )

            evs.append(
                EVOneInterval(
                    cfg=cfg,
                    electric_charge_mwh=charge_mwh,
                    electric_charge_binary=charge_binary,
                    electric_discharge_mwh=discharge_mwh,
                    electric_discharge_binary=discharge_binary,
                    initial_soc_mwh=initial_charge,
                    final_soc_mwh=final_charge,
                    electric_loss_mwh=loss_mwh,
                    is_spill=is_spill,
                )
            )
            charges_mwh[0, charge_event_idx, charger_idx] = charge_mwh
            charges_binary[0, charge_event_idx, charger_idx] = charge_binary
            discharges_mwh[0, charge_event_idx, charger_idx] = discharge_mwh
            discharges_binary[0, charge_event_idx, charger_idx] = discharge_binary
            losses_mwh[0, charge_event_idx, charger_idx] = loss_mwh

    evs_array = EVsArrayOneInterval(
        i=i,
        electric_charge_mwh=charges_mwh,
        electric_charge_binary=charges_binary,
        electric_discharge_mwh=discharges_mwh,
        electric_discharge_binary=discharges_binary,
        cfg=cfg,
        # charger_cfgs=charger_cfgs,
        # charge_event_cfgs=charge_event_cfgs,
        initial_soc_mwh=initial_socs_mwh,
        final_soc_mwh=final_socs_mwh,
        electric_loss_mwh=losses_mwh,
        is_spill=is_spill,
    )
    return evs, evs_array


def constrain_charge_discharge_min_max(
    optimizer: Optimizer,
    evs: EVsArrayOneInterval,
    charge_event: np.ndarray,
    freq: Freq,
    charger_cfgs: np.ndarray,
    charge_event_cfgs: np.ndarray,
    i: int,
    allow_evs_discharge: bool,
) -> None:
    """Add constraints between the continuous and binary variables
    for the EVs charge and discharge electricity.

    This constraint is applied once per interval."""
    for charge_event_idx, _ in enumerate(charge_event_cfgs):
        for charger_idx, charger_cfg in enumerate(charger_cfgs):

            variables = [
                (
                    evs.electric_charge_mwh[0, charge_event_idx, charger_idx],
                    evs.electric_charge_binary[0, charge_event_idx, charger_idx],
                )
            ]
            if allow_evs_discharge:
                variables += [
                    (
                        evs.electric_discharge_mwh[0, charge_event_idx, charger_idx],
                        evs.electric_discharge_binary[0, charge_event_idx, charger_idx],
                    ),
                ]

            for continuous, binary in variables:
                if isinstance(continuous, pulp.LpVariable):
                    optimizer.constrain_max(
                        continuous, binary, freq.mw_to_mwh(charger_cfg.power_max_mw)
                    )
                    optimizer.constrain_min(
                        continuous, binary, freq.mw_to_mwh(charger_cfg.power_min_mw)
                    )
                    #  only let the binary be positive when the charge_event is positive
                    #  this forces the charger to only charge during a charge event
                    #  this could be a separate function
                    optimizer.constrain(binary <= charge_event[i, charge_event_idx])


def constrain_single_charger_charge_event(
    optimizer: Optimizer,
    evs: EVsArrayOneInterval,
    charger_cfgs: np.ndarray,
    charge_event_cfgs: np.ndarray,
) -> None:
    """Add constraints to limit when charging can occur.

    This constraint is applied once per interval."""

    #  constrain to only one charger per charging event
    #  this also has the effect of only allowing
    #  charge or discharge
    for charge_event_idx, _ in enumerate(charge_event_cfgs):
        optimizer.constrain(
            optimizer.sum(
                evs.electric_charge_binary[0, charge_event_idx, :].tolist()
                + evs.electric_discharge_binary[0, charge_event_idx, :].tolist()
            )
            <= 1
        )
    #  constrain to only one charge event per charger
    for charger_idx, _ in enumerate(charger_cfgs):
        optimizer.constrain(
            optimizer.sum(
                evs.electric_charge_binary[0, :, charger_idx].tolist()
                + evs.electric_discharge_binary[0, :, charger_idx].tolist()
            )
            <= 1
        )


def constrain_charge_event_electricity_balance(
    optimizer: Optimizer,
    evs_array: EVsArrayOneInterval,
    spill_evs_array: EVsArrayOneInterval,
) -> None:
    """Constrain the electricity balance across the battery.  This is within one interval.

    This constrant is applied once per interval."""
    for charge_event_idx, charge_event_cfg in enumerate(
        evs_array.cfg.charge_event_cfgs
    ):
        optimizer.constrain(
            evs_array.initial_soc_mwh[0, charge_event_idx]
            + optimizer.sum(evs_array.electric_charge_mwh[0, charge_event_idx].tolist())
            + optimizer.sum(
                spill_evs_array.electric_charge_mwh[0, charge_event_idx].tolist()
            )
            - optimizer.sum(
                evs_array.electric_discharge_mwh[0, charge_event_idx].tolist()
            )
            - optimizer.sum(
                spill_evs_array.electric_discharge_mwh[0, charge_event_idx].tolist()
            )
            - optimizer.sum(evs_array.electric_loss_mwh[0, charge_event_idx].tolist())
            == evs_array.final_soc_mwh[0, charge_event_idx]
        )

        #  losses
        """
        used to not do the iteration over chargers here and just take the sum
        means less constraints
        """
        for charger_idx, _ in enumerate(evs_array.cfg.charger_cfgs):
            optimizer.constrain(
                optimizer.sum(
                    evs_array.electric_charge_mwh[0, charge_event_idx, charger_idx]
                )
                * (1 - charge_event_cfg.efficiency_pct)
                == optimizer.sum(
                    evs_array.electric_loss_mwh[0, charge_event_idx, charger_idx]
                )
            )


def constrain_connection_charge_events_between_intervals(
    optimizer: Optimizer, evs: list[EVsArrayOneInterval], i: int
) -> None:
    """Constrain state of charges between intervals.

    This uses data from two adjacent intervals.

    This constrant is conditionally applied once per interval."""
    if i == 0:
        return None

    old = evs[-2]
    new = evs[-1]
    for initial_soc, final_soc in zip(
        new.initial_soc_mwh[0], old.final_soc_mwh[0], strict=True
    ):
        if isinstance(initial_soc, pulp.LpVariable) and isinstance(
            final_soc, pulp.LpVariable
        ):
            optimizer.constrain(initial_soc == final_soc)


def constrain_initial_final_charge(
    optimizer: Optimizer,
    evs: list[EVsArrayOneInterval],
    charge_event_cfgs: np.ndarray,
) -> None:
    """Constrain state of charge at the beginning and end of the simulation."""

    for charge_event_idx, charge_event_cfg in enumerate(charge_event_cfgs):
        #  constrain the intial soc
        optimizer.constrain(evs[0].initial_soc_mwh[0, charge_event_idx] == 0)

        #  constrain the final soc
        optimizer.constrain(
            evs[-1].final_soc_mwh[0, charge_event_idx] == charge_event_cfg.capacity_mwh
        )

    #  intentionally don't constrain the spill charger


class EVs:
    """Electric vehicle asset, used to represent multiple chargers.

    Can handle vehicle-to-grid charging.

    Handles optimization and plotting of results over many intervals.

    Args:
        chargers_power_mw: size of EV chargers in mega-watts.
        charge_events_capacity_mwh:
            1D array of final SOC for each charge event.
            Length is the number of charge events.
        charge_event_efficiency:
            Roundtrip efficiency of the charge event charge & discharge.
        charger_turndown:
            minimum charger output as a percent of the
            charger size in mega-watts.
        name
        charge_events:
            An optional 2D array of charge events.  The epl.Site API
            makes use of this to pass in charge events from epl.Site.optimize.
    """

    def __init__(
        self,
        chargers_power_mw: list[float] | None = None,
        charge_events_capacity_mwh: typing.Sequence[float] | None = None,
        charge_event_efficiency: float = 0.9,
        charger_turndown: float = 0.1,
        name: str = "evs",
        charge_events: list | None = None,
    ):
        """Initialize an electric vehicle asset model."""

        if chargers_power_mw is None:
            ds = epl.data_generation.generate_random_ev_input_data(
                48, n_chargers=3, charge_length=3, n_charge_events=12, seed=42
            )
            chargers_power_mw = ds["charger_mws"]
            charge_events_capacity_mwh = ds["charge_events_capacity_mwh"]
            charge_events = ds["charge_events"]

        assert chargers_power_mw is not None
        assert charge_events_capacity_mwh is not None

        charger_cfgs = np.array(
            [
                ChargerConfig(
                    name=f"charger-{name}",
                    power_max_mw=power_mw,
                    power_min_mw=power_mw * charger_turndown,
                )
                for name, power_mw in enumerate(chargers_power_mw)
            ]
        )
        spill_charger_config = np.array(
            [
                ChargerConfig(
                    name=f"charger-spill-{name}",
                    power_max_mw=defaults.spill_charge_max_mw,
                    power_min_mw=0,
                )
            ]
        )
        charge_event_cfgs = np.array(
            [
                ChargeEventConfig(
                    name=f"charge-event-{name}",
                    capacity_mwh=capacity_mwh,
                    efficiency_pct=charge_event_efficiency,
                )
                for name, capacity_mwh in enumerate(charge_events_capacity_mwh)
            ]
        )

        self.cfg = EVsConfig(
            name=name,
            charger_cfgs=charger_cfgs,
            spill_charger_cfgs=spill_charger_config,
            charge_event_cfgs=charge_event_cfgs,
        )

        self.site = epl.Site()
        self.spill = epl.spill.Spill()

        #  this feels a bit wierd, because `charge_events` is interval data
        #  interval data is usually input with the `asset.optimize()` call
        #  we need this shortcut to get the site API working
        if charge_events is not None:
            # transpose charge_events to have time as first dimension
            # equivilant to the batch dimension in neural nets
            self.charge_events: np.ndarray | None = np.array(charge_events).T
            validate_charge_events(
                self.cfg.charge_event_cfgs,
                self.charge_events
            )
        else:
            self.charge_events = None

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<energypylinear.EVs n_chargers: {len(self.cfg.charger_cfgs)}, n_charge_events: {len(self.cfg.charge_event_cfgs)}>"

    def one_interval(
        self, optimizer: Optimizer, i: int, freq: Freq, flags: Flags = Flags()
    ) -> tuple:
        """Create EV asset data for a single interval."""

        assert isinstance(self.cfg.charge_event_cfgs, np.ndarray)
        assert isinstance(self.cfg.charger_cfgs, np.ndarray)
        assert isinstance(self.cfg.spill_charger_cfgs, np.ndarray)
        assert isinstance(self.charge_events, np.ndarray)

        evs, evs_array = evs_one_interval(
            optimizer,
            self.cfg,
            self.cfg.charger_cfgs,
            self.cfg.charge_event_cfgs,
            self.charge_events,
            i,
            freq,
            asset_name=self.cfg.name,
            flags=flags,
        )
        spill_evs, spill_evs_array = evs_one_interval(
            optimizer,
            self.cfg,
            self.cfg.spill_charger_cfgs,
            self.cfg.charge_event_cfgs,
            self.charge_events,
            i,
            freq,
            asset_name=self.cfg.name,
            flags=flags,
            create_charge_event_soc=False,
            create_discharge_variables=False,
            is_spill=True,
        )
        return evs, evs_array, spill_evs, spill_evs_array

    def constrain_within_interval(
        self,
        optimizer: Optimizer,
        ivars: "epl.interval_data.IntervalVars",
        interval_data: "epl.IntervalData",
        i: int,
        freq: Freq,
        flags: Flags = Flags(),
    ) -> None:
        """Constrain EVs dispatch within a single interval"""

        assert self.charge_events is not None

        constrain_charge_discharge_min_max(
            optimizer,
            ivars.filter_evs_array(is_spill=False, i=i, asset_name=self.cfg.name),
            self.charge_events,
            freq,
            self.cfg.charger_cfgs,
            self.cfg.charge_event_cfgs,
            i,
            allow_evs_discharge=flags.allow_evs_discharge,
        )

        constrain_charge_discharge_min_max(
            optimizer,
            ivars.filter_evs_array(is_spill=True, i=i, asset_name=self.cfg.name),
            self.charge_events,
            freq,
            self.cfg.spill_charger_cfgs,
            self.cfg.charge_event_cfgs,
            i,
            #  never allow a spill charger to discharge
            allow_evs_discharge=False,
        )

        constrain_single_charger_charge_event(
            optimizer,
            ivars.filter_evs_array(is_spill=False, i=i, asset_name=self.cfg.name),
            self.cfg.charger_cfgs,
            self.cfg.charge_event_cfgs,
        )

        constrain_charge_event_electricity_balance(
            optimizer,
            ivars.filter_evs_array(is_spill=False, i=i, asset_name=self.cfg.name),
            ivars.filter_evs_array(is_spill=True, i=i, asset_name=self.cfg.name),
        )
        constrain_connection_charge_events_between_intervals(
            optimizer, ivars.filter_all_evs_array(False, self.cfg.name), i
        )

    def constrain_after_intervals(
        self,
        optimizer: Optimizer,
        ivars: "epl.interval_data.IntervalVars",
        interval_data: "epl.IntervalData",
    ) -> None:
        """Constrain EVs after all interval asset models are created."""
        assert isinstance(self.cfg.charge_event_cfgs, np.ndarray)
        constrain_initial_final_charge(
            optimizer,
            ivars.filter_all_evs_array(False, self.cfg.name),
            self.cfg.charge_event_cfgs,
        )

    def optimize(
        self,
        electricity_prices: list[float] | np.ndarray,
        electricity_carbon_intensities: float | list[float] | np.ndarray | None = None,
        gas_prices: float | list[float] | np.ndarray | None = None,
        charge_events: list[list[int]] | np.ndarray | None = None,
        freq_mins: int = defaults.freq_mins,
        objective: str = "price",
        verbose: bool = True,
        flags: Flags = Flags(),
    ) -> "epl.results.SimulationResult":
        """Optimize the EVs's dispatch using a mixed-integer linear program.

        Args:
            electricity_prices - the price of electricity in each interval.
            gas_prices - the prices of natural gas, used in CHP and boilers in each interval.
            electricity_carbon_intensities - carbon intensity of electricity in each interval.
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

            freq_mins - the size of an interval in minutes.
            objective - the optimization objective - either "price" or "carbon".
            verbose: level of printing.

        Returns:
            epl.results.SimulationResult
        """
        self.optimizer = Optimizer()
        freq = Freq(freq_mins)

        """
        complexity here is to be able to accept `charge_events` as an input in either the class init or in optimize()

        - want class init during site api,
        - want optimize during asset api
        """
        if self.charge_events is None:
            assert charge_events is not None
            self.charge_events = np.array(charge_events).T
            validate_charge_events(self.cfg.charge_event_cfgs, self.charge_events)

        self.interval_data = epl.interval_data.IntervalData(
            electricity_prices=electricity_prices,
            gas_prices=gas_prices,
            electricity_carbon_intensities=electricity_carbon_intensities,
            evs=epl.interval_data.EVIntervalData(
                charge_events=self.charge_events,
                charge_events_capacity_mwh=[
                    cfg.capacity_mwh for cfg in self.cfg.charge_event_cfgs
                ],
            ),
        )

        ivars = epl.interval_data.IntervalVars()
        for i in self.interval_data.idx:
            ivars.append(self.site.one_interval(self.optimizer, self.site.cfg, i, freq))
            assert isinstance(self.charge_events, np.ndarray)
            evs, evs_array, spill_evs, spill_evs_array = self.one_interval(
                self.optimizer,
                i,
                freq,
                flags=flags,
            )

            ivars.append(evs_array)
            ivars.append(spill_evs_array)
            ivars.append(
                [
                    *evs,
                    *spill_evs,
                    self.spill.one_interval(self.optimizer, i, freq),
                ]
            )

            self.site.constrain_within_interval(
                self.optimizer, ivars, self.interval_data, i
            )

            self.constrain_within_interval(
                self.optimizer, ivars, self.interval_data, i, freq=freq, flags=flags
            )

        assert isinstance(self.charge_events, np.ndarray)
        assert isinstance(self.cfg.charger_cfgs, np.ndarray)
        assert isinstance(self.cfg.spill_charger_cfgs, np.ndarray)
        self.constrain_after_intervals(
            self.optimizer,
            ivars,
            self.interval_data,
        )

        objective_fn = epl.objectives[objective]
        self.optimizer.objective(
            objective_fn(self.optimizer, ivars, self.interval_data)
        )

        status = self.optimizer.solve(verbose=verbose)
        return epl.results.extract_results(
            self.interval_data,
            ivars,
            feasible=status.feasible,
            flags=flags,
            verbose=verbose,
        )

    def plot(
        self, results: "epl.results.SimulationResult", path: pathlib.Path | str
    ) -> None:
        """Plot simulation results."""
        return epl.plot.plot_evs(results, pathlib.Path(path))
