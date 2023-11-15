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


def validate_ev_interval_data(
    idx: np.ndarray,
    charge_events: np.ndarray,
) -> None:
    """Helper used to validate EV interval data.

    TODO move into an EVsIntervalData model.
    """
    assert idx.shape[0] == charge_events.shape[0]
    assert all(np.array(charge_events).sum(axis=0) > 0), "sum across axis=0"


def validate_charge_events(
    charge_event_cfgs: np.ndarray, charge_events: np.ndarray
) -> None:
    """Helper used to handle charge events.

    This is a bit of a smell - will probably be reworked once I see a way
    to avoid the duplicate of the arrays and list version of the events.
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
    charge_events: np.ndarray
    freq_mins: int
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<EVsConfig name: {self.name}, n_chargers: {len(self.charger_cfgs)}, n_charge_events: {len(self.charge_event_cfgs)}>"

    def __str__(self) -> str:
        """A string representation of self."""
        return repr(self)

    @pydantic.field_validator("name")
    @classmethod
    def check_name(cls, name: str) -> str:
        """Ensure we can identify this asset correctly."""
        assert "evs" in name
        return name

    @pydantic.model_validator(mode="after")
    def validate_charge_events(self) -> "EVsConfig":
        """Check charge events match the configs"""
        validate_charge_events(self.charge_event_cfgs, self.charge_events)
        return self


class EVOneInterval(AssetOneInterval):
    """EV asset data for a single interval.

    This single data structure contains data for one asset
    in one interval."""

    i: int
    charge_event_idx: int
    charge_event_cfg: ChargeEventConfig
    charger_idx: int
    charger_cfg: ChargerConfig

    electric_charge_mwh: pulp.LpVariable | float
    electric_charge_binary: pulp.LpVariable | int
    electric_discharge_mwh: pulp.LpVariable | float
    electric_discharge_binary: pulp.LpVariable | int
    initial_soc_mwh: pulp.LpVariable | float
    final_soc_mwh: pulp.LpVariable | float
    electric_loss_mwh: pulp.LpVariable | float

    is_spill: typing.Literal[False] = False
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<EVOneInterval i: {self.i} charge_event_idx: {self.charge_event_idx} charger_idx: {self.charger_idx}, is_spill: {self.is_spill}>"

    def __str__(self) -> str:
        """A string representation of self."""
        return repr(self)


class EVSpillOneInterval(AssetOneInterval):
    """Spill EV asset data for a single interval.

    This single data structure contains data for one asset
    in one interval."""

    i: int
    charge_event_idx: int
    charge_event_cfg: ChargeEventConfig
    charger_idx: int
    charger_cfg: ChargerConfig

    electric_charge_mwh: pulp.LpVariable | float
    electric_charge_binary: pulp.LpVariable | int
    electric_discharge_mwh: pulp.LpVariable | float
    electric_discharge_binary: pulp.LpVariable | int
    initial_soc_mwh: pulp.LpVariable | float
    final_soc_mwh: pulp.LpVariable | float
    electric_loss_mwh: pulp.LpVariable | float

    is_spill: typing.Literal[True] = True
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<EVSpillOneInterval i: {self.i} charge_event_idx: {self.charge_event_idx} charger_idx: {self.charger_idx}, is_spill: {self.is_spill}>"


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
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    charge_event_idxs: np.ndarray
    charger_idxs: np.ndarray

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<EVsArrayOneInterval i: {self.i} chargers: {len(self.cfg.charger_cfgs)} charge-events: {len(self.cfg.charge_event_cfgs)}, is_spill: {self.is_spill}>"


def create_evs_array(
    ivars: "epl.IntervalVars", i: int, asset_name: str, is_spill: bool
) -> EVsArrayOneInterval:
    """Creates the array representation from EVOneInterval objects."""
    one_intervals: list[EVOneInterval] | list[EVSpillOneInterval]
    if not is_spill:
        one_intervals = typing.cast(
            list[EVOneInterval],
            ivars.filter_objective_variables(
                instance_type=epl.assets.evs.EVOneInterval, i=i, asset_name=asset_name
            ),
        )
    else:
        one_intervals = typing.cast(
            list[EVSpillOneInterval],
            ivars.filter_objective_variables(
                instance_type=epl.assets.evs.EVSpillOneInterval,
                i=i,
                asset_name=asset_name,
            ),
        )
    asset_cfg = one_intervals[0].cfg
    if is_spill:
        assert len(one_intervals) == len(asset_cfg.charge_event_cfgs) * 1

    else:
        assert len(one_intervals) == len(asset_cfg.charge_event_cfgs) * len(
            asset_cfg.charger_cfgs
        )

    n_chargers = len(asset_cfg.charger_cfgs)
    n_charge_events = len(asset_cfg.charge_event_cfgs)

    #  2D arrays
    initial_socs_mwh = np.zeros((1, n_charge_events)).astype(object)
    final_socs_mwh = np.zeros((1, n_charge_events)).astype(object)

    #  3D arrays
    charges_mwh = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    charges_binary = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    discharges_mwh = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    discharges_binary = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    losses_mwh = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    charge_event_idxs = np.zeros((1, n_charge_events, n_chargers)).astype(object)
    charger_idxs = np.zeros((1, n_charge_events, n_chargers)).astype(object)

    for oi in one_intervals:
        initial_socs_mwh[0, oi.charge_event_idx] = oi.initial_soc_mwh
        final_socs_mwh[0, oi.charge_event_idx] = oi.final_soc_mwh
        charges_mwh[0, oi.charge_event_idx, oi.charger_idx] = oi.electric_charge_mwh
        charges_binary[
            0, oi.charge_event_idx, oi.charger_idx
        ] = oi.electric_charge_binary
        discharges_mwh[
            0, oi.charge_event_idx, oi.charger_idx
        ] = oi.electric_discharge_mwh
        discharges_binary[
            0, oi.charge_event_idx, oi.charger_idx
        ] = oi.electric_discharge_binary
        losses_mwh[0, oi.charge_event_idx, oi.charger_idx] = oi.electric_loss_mwh
        charge_event_idxs[0, oi.charge_event_idx, oi.charger_idx] = oi.charge_event_idx
        charger_idxs[0, oi.charge_event_idx, oi.charger_idx] = oi.charger_idx

    evs_array = epl.assets.evs.EVsArrayOneInterval(
        i=i,
        cfg=asset_cfg,
        charge_event_idxs=charge_event_idxs,
        charger_idxs=charger_idxs,
        electric_charge_mwh=charges_mwh,
        electric_charge_binary=charges_binary,
        electric_discharge_mwh=discharges_mwh,
        electric_discharge_binary=discharges_binary,
        initial_soc_mwh=initial_socs_mwh,
        final_soc_mwh=final_socs_mwh,
        electric_loss_mwh=losses_mwh,
        is_spill=is_spill,
    )

    return evs_array


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
) -> list[EVOneInterval] | list[EVSpillOneInterval]:
    """Create EV asset data for a single interval.

    Create a list of `EVOneInterval` and one `EVsArrayOneInterval` data structures.

    This data is a collection of linear programming variables for the EV charge, discharge and state of charges.

    The list is used in the `vars` lists, which ends up in the objective function.

    The array representation is used in constructing the linear program.
    """

    """
    there is opportunity here to prune the number of variables
    - when the charge event binary is 0 for a charger, charge_event pair,
    - when the charge event is over (ie the last charge event)

    leave this as a fun project for later

    when the charge event binary is 0 for a charger, charge_event pair,
    - this seems like the real profitable one
    - most of the time the charge event binary is 0
    """
    evs: list = []

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

        for charger_idx, charger_cfg in enumerate(charger_cfgs):
            name = f"i:{i},asset:{asset_name},charge-event:{charge_event_idx},charger:{charger_cfg.name},is_spill:{is_spill}"

            if flags.limit_charge_variables_to_valid_events:
                charge_event = charge_events[i, charge_event_idx]
                electric_charge_mwh = (
                    optimizer.continuous(
                        f"electric_charge_mwh,{name}",
                        low=0,
                        up=freq.mw_to_mwh(charger_cfg.power_max_mw),
                    )
                    if charge_event == 1
                    else 0
                )
                electric_charge_binary = (
                    optimizer.binary(f"electric_charge_binary,{name}")
                    if charge_event == 1
                    else 0
                )

            else:
                electric_charge_mwh = optimizer.continuous(
                    f"electric_charge_mwh,{name}",
                    low=0,
                    up=freq.mw_to_mwh(charger_cfg.power_max_mw),
                )
                electric_charge_binary = optimizer.binary(
                    f"electric_charge_binary,{name}"
                )

            electric_discharge_mwh = 0
            electric_discharge_binary = 0
            if create_discharge_variables:
                electric_discharge_mwh = (
                    optimizer.continuous(
                        f"electric_discharge_mwh,{name}",
                        low=0,
                        up=freq.mw_to_mwh(charger_cfg.power_max_mw),
                    )
                    if flags.allow_evs_discharge
                    else 0
                )
                electric_discharge_binary = (
                    optimizer.binary(f"electric_discharge_binary,{name}")
                    if flags.allow_evs_discharge
                    else 0
                )

            loss_mwh = optimizer.continuous(
                f"electric_loss_mwh,{name}",
                low=0,
            )
            if is_spill:
                evs.append(
                    EVSpillOneInterval(
                        i=i,
                        cfg=cfg,
                        charge_event_idx=charge_event_idx,
                        charge_event_cfg=charge_event_cfg,
                        charger_idx=charger_idx,
                        charger_cfg=charger_cfg,
                        electric_charge_mwh=electric_charge_mwh,
                        electric_charge_binary=electric_charge_binary,
                        electric_discharge_mwh=electric_discharge_mwh,
                        electric_discharge_binary=electric_discharge_binary,
                        initial_soc_mwh=initial_charge,
                        final_soc_mwh=final_charge,
                        electric_loss_mwh=loss_mwh,
                        is_spill=is_spill,
                    )
                )
            else:
                evs.append(
                    EVOneInterval(
                        i=i,
                        cfg=cfg,
                        charge_event_idx=charge_event_idx,
                        charge_event_cfg=charge_event_cfg,
                        charger_idx=charger_idx,
                        charger_cfg=charger_cfg,
                        electric_charge_mwh=electric_charge_mwh,
                        electric_charge_binary=electric_charge_binary,
                        electric_discharge_mwh=electric_discharge_mwh,
                        electric_discharge_binary=electric_discharge_binary,
                        initial_soc_mwh=initial_charge,
                        final_soc_mwh=final_charge,
                        electric_loss_mwh=loss_mwh,
                        is_spill=is_spill,
                    )
                )
    return evs


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

    This constraint is applied once per interval."""
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
        for arr in [evs_array, spill_evs_array]:
            for charger_idx, _ in enumerate(arr.cfg.charger_cfgs):
                optimizer.constrain(
                    optimizer.sum(
                        arr.electric_charge_mwh[0, charge_event_idx, charger_idx]
                    )
                    * (1 - charge_event_cfg.efficiency_pct)
                    == optimizer.sum(
                        arr.electric_loss_mwh[0, charge_event_idx, charger_idx]
                    )
                )


def constrain_connection_charge_events_between_intervals(
    optimizer: Optimizer,
    old: EVsArrayOneInterval,
    new: EVsArrayOneInterval,
) -> None:
    """Constrain state of charges between intervals.

    This uses data from two adjacent intervals.

    This constraint is conditionally applied once per interval."""

    for initial_soc, final_soc in zip(
        new.initial_soc_mwh[0], old.final_soc_mwh[0], strict=True
    ):
        if isinstance(initial_soc, pulp.LpVariable) and isinstance(
            final_soc, pulp.LpVariable
        ):
            optimizer.constrain(initial_soc == final_soc)


def constrain_initial_final_charge(
    optimizer: Optimizer,
    initial: EVsArrayOneInterval,
    final: EVsArrayOneInterval,
    charge_event_cfgs: np.ndarray,
) -> None:
    """Constrain state of charge at the beginning and end of the simulation."""

    for charge_event_idx, charge_event_cfg in enumerate(charge_event_cfgs):
        #  constrain the initial soc
        optimizer.constrain(initial.initial_soc_mwh[0, charge_event_idx] == 0)

        #  constrain the final soc
        optimizer.constrain(
            final.final_soc_mwh[0, charge_event_idx] == charge_event_cfg.capacity_mwh
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
        name: asset name
        electricity_prices - the price of electricity in each interval.
        electricity_carbon_intensities - carbon intensity of electricity in each interval.
        charge_events: 2D matrix representing when a charge event is active.
            Shape is (n_charge_events, n_timesteps).
            A charge events matrix for 4 charge events over 5 intervals:
            charge_events = [
                [1, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1],
                [0, 1, 0, 0, 0],
            ]
    """

    def __init__(
        self,
        charge_events: np.ndarray | list[list[int]],
        chargers_power_mw: np.ndarray | list[float],
        charge_events_capacity_mwh: np.ndarray | list[float],
        charge_event_efficiency: float = 0.9,
        charger_turndown: float = 0.1,
        name: str = "evs",
        electricity_prices: np.ndarray | list[float] | np.ndarray | None = None,
        export_electricity_prices: np.ndarray | list[float] | np.ndarray | None = None,
        electricity_carbon_intensities: np.ndarray
        | list[float]
        | np.ndarray
        | None = None,
        freq_mins: int = defaults.freq_mins,
    ):
        """Initialize an electric vehicle asset model."""

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
            # transpose charge_events to have time as first dimension
            charge_events=np.array(charge_events).T,
            freq_mins=freq_mins,
        )

        if electricity_prices is not None or electricity_carbon_intensities is not None:
            assets = [self, epl.Spill()]
            self.site = epl.Site(
                assets=assets,
                electricity_prices=electricity_prices,
                export_electricity_prices=export_electricity_prices,
                electricity_carbon_intensities=electricity_carbon_intensities,
                freq_mins=self.cfg.freq_mins,
            )
            assert isinstance(self.site.cfg.interval_data.idx, np.ndarray)
            validate_ev_interval_data(
                self.site.cfg.interval_data.idx, self.cfg.charge_events
            )

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<energypylinear.EVs n_chargers: {len(self.cfg.charger_cfgs)}, n_charge_events: {len(self.cfg.charge_event_cfgs)}>"

    def one_interval(
        self, optimizer: Optimizer, i: int, freq: Freq, flags: Flags = Flags()
    ) -> list[EVOneInterval]:
        """Create EV asset data for a single interval."""

        assert isinstance(self.cfg.charge_event_cfgs, np.ndarray)
        assert isinstance(self.cfg.charger_cfgs, np.ndarray)
        assert isinstance(self.cfg.spill_charger_cfgs, np.ndarray)
        assert isinstance(self.cfg.charge_events, np.ndarray)

        evs = evs_one_interval(
            optimizer,
            self.cfg,
            self.cfg.charger_cfgs,
            self.cfg.charge_event_cfgs,
            self.cfg.charge_events,
            i,
            freq,
            asset_name=self.cfg.name,
            flags=flags,
        )
        spill_evs = evs_one_interval(
            optimizer,
            self.cfg,
            self.cfg.spill_charger_cfgs,
            self.cfg.charge_event_cfgs,
            self.cfg.charge_events,
            i,
            freq,
            asset_name=self.cfg.name,
            flags=flags,
            create_charge_event_soc=False,
            create_discharge_variables=False,
            is_spill=True,
        )
        return [*evs, *spill_evs]

    def constrain_within_interval(
        self,
        optimizer: Optimizer,
        ivars: "epl.interval_data.IntervalVars",
        i: int,
        freq: Freq,
        flags: Flags = Flags(),
    ) -> None:
        """Constrain EVs dispatch within a single interval"""

        assert self.cfg.charge_events is not None

        constrain_charge_discharge_min_max(
            optimizer,
            create_evs_array(ivars, i=i, asset_name=self.cfg.name, is_spill=False),
            self.cfg.charge_events,
            freq,
            self.cfg.charger_cfgs,
            self.cfg.charge_event_cfgs,
            i,
            allow_evs_discharge=flags.allow_evs_discharge,
        )

        constrain_charge_discharge_min_max(
            optimizer,
            # ivars.filter_evs_array(is_spill=True, i=i, asset_name=self.cfg.name),
            create_evs_array(ivars, i=i, asset_name=self.cfg.name, is_spill=True),
            self.cfg.charge_events,
            freq,
            self.cfg.spill_charger_cfgs,
            self.cfg.charge_event_cfgs,
            i,
            #  never allow a spill charger to discharge
            allow_evs_discharge=False,
        )

        constrain_single_charger_charge_event(
            optimizer,
            # ivars.filter_evs_array(is_spill=False, i=i, asset_name=self.cfg.name),
            create_evs_array(ivars, i=i, asset_name=self.cfg.name, is_spill=False),
            self.cfg.charger_cfgs,
            self.cfg.charge_event_cfgs,
        )

        constrain_charge_event_electricity_balance(
            optimizer,
            # ivars.filter_evs_array(is_spill=False, i=i, asset_name=self.cfg.name),
            # ivars.filter_evs_array(is_spill=True, i=i, asset_name=self.cfg.name),
            create_evs_array(ivars, i=i, asset_name=self.cfg.name, is_spill=False),
            create_evs_array(ivars, i=i, asset_name=self.cfg.name, is_spill=True),
        )

        if i > 0:
            constrain_connection_charge_events_between_intervals(
                optimizer,
                create_evs_array(
                    ivars, i=i - 1, asset_name=self.cfg.name, is_spill=False
                ),
                create_evs_array(ivars, i=i, asset_name=self.cfg.name, is_spill=False),
            )

    def constrain_after_intervals(
        self,
        optimizer: Optimizer,
        ivars: "epl.interval_data.IntervalVars",
    ) -> None:
        """Constrain EVs after all interval asset models are created."""
        assert isinstance(self.cfg.charge_event_cfgs, np.ndarray)
        constrain_initial_final_charge(
            optimizer,
            create_evs_array(ivars, i=0, asset_name=self.cfg.name, is_spill=False),
            create_evs_array(ivars, i=-1, asset_name=self.cfg.name, is_spill=False),
            # ivars.filter_all_evs_array(False, self.cfg.name),
            self.cfg.charge_event_cfgs,
        )

    def optimize(
        self,
        objective: "str | dict | epl.objectives.CustomObjectiveFunction" = "price",
        verbose: int | bool = 2,
        flags: Flags = Flags(),
        optimizer_config: "epl.OptimizerConfig" = epl.optimizer.OptimizerConfig(),
    ) -> "epl.SimulationResult":
        """Optimize the EVs's dispatch using a mixed-integer linear program.

        Args:
            objective: the optimization objective - either "price" or "carbon".
            flags: boolean flags to change simulation and results behaviour.
            verbose: level of printing.
            optimizer_config: configuration options for the optimizer.

        Returns:
            epl.results.SimulationResult
        """
        return self.site.optimize(
            objective=objective,
            flags=flags,
            verbose=verbose,
            optimizer_config=optimizer_config,
        )

    def plot(self, results: "epl.SimulationResult", path: pathlib.Path | str) -> None:
        """Plot simulation results."""
        return epl.plot.plot_evs(results, pathlib.Path(path), asset_name=self.cfg.name)
