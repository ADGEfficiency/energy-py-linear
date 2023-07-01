"""Test electric vehicle asset."""
import hypothesis
import numpy as np
import pytest

import energypylinear as epl
from energypylinear.flags import Flags


def test_evs_optimization_price() -> None:
    """Test EV optimization for price."""

    charge_events_capacity_mwh = [50, 100, 30, 40]
    evs = epl.evs.EVs(
        chargers_power_mw=[100, 100],
        charge_events_capacity_mwh=charge_events_capacity_mwh,
        charger_turndown=0.0,
        charge_event_efficiency=1.0,
    )
    results = evs.optimize(
        electricity_prices=[-100, 50, 30, 50, 40],
        charge_events=[
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0],
        ],
        flags=Flags(
            allow_evs_discharge=False,
            fail_on_spill_asset_use=True,
            allow_infeasible=False,
        ),
        freq_mins=60,
    )
    simulation = results.simulation
    print(simulation[[c for c in simulation.columns if "soc" in c]])

    #  test total import power equal to total charge event mwh
    #  requires efficiency to be 100%
    np.testing.assert_equal(
        simulation["site-import_power_mwh"].sum(), sum(charge_events_capacity_mwh)
    )

    #  no exporting at all
    np.testing.assert_equal(simulation["site-export_power_mwh"].sum(), 0)

    #  test dispatch exactly as we expect
    np.testing.assert_array_equal(
        simulation["site-import_power_mwh"], [50, 40, 100, 0, 30]
    )


def test_evs_optimization_carbon() -> None:
    """Test EV optimization for carbon."""
    charge_events_capacity_mwh = [50, 100, 30, 40]
    evs = epl.evs.EVs(
        chargers_power_mw=[100, 100],
        charge_events_capacity_mwh=charge_events_capacity_mwh,
        charger_turndown=0.0,
        charge_event_efficiency=1.0,
    )
    results = evs.optimize(
        electricity_prices=[-100, 50, 30, 50, 40, 10],
        electricity_carbon_intensities=[0.1, 0.3, -0.5, 0.95, 0.9, 0.0],
        charge_events=[
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
        ],
        flags=Flags(
            allow_evs_discharge=False,
            fail_on_spill_asset_use=True,
            allow_infeasible=False,
        ),
        freq_mins=60,
        objective="carbon",
    )
    simulation = results.simulation
    #  test total import power equal to total charge event mwh
    #  requires efficiency to be 100%
    np.testing.assert_equal(
        simulation["site-import_power_mwh"].sum(), sum(charge_events_capacity_mwh)
    )

    #  no exporting at all
    np.testing.assert_equal(simulation["site-export_power_mwh"].sum(), 0)

    #  test dispatch exactly as we expect
    np.testing.assert_array_equal(
        simulation["site-import_power_mwh"], [50.0, 0.0, 100.0, 0.0, 30.0, 40]
    )


@pytest.mark.parametrize(
    "efficiency",
    [
        1.0,
        0.9,
        0.5,
        # 0.1,  TODO causes problems
        # 0.0,
        # -0.1
    ],
)
def test_evs_efficiency_losses(efficiency: float) -> None:
    charge_events_capacity_mwh = [50, 100, 30, 40]
    evs = epl.evs.EVs(
        chargers_power_mw=[500, 500],
        charge_events_capacity_mwh=charge_events_capacity_mwh,
        charger_turndown=0.0,
        charge_event_efficiency=efficiency,
    )
    results = evs.optimize(
        electricity_prices=[-100, 50, 30, 50, 40],
        charge_events=[
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0],
        ],
        flags=Flags(
            allow_evs_discharge=False,
            fail_on_spill_asset_use=True,
            allow_infeasible=False,
        ),
        freq_mins=60,
    )
    simulation = results.simulation
    print(simulation[[c for c in simulation.columns if "soc" in c]])

    #  test total import versus charge event mwh
    #  a less efficient charge event should mean we import more power
    #  our expected losses are a function of the amount we charge a charge event

    losses_mwh = simulation["total-electric_loss_mwh"].sum()
    # np.testing.assert_equal(
    #     simulation["site-import_power_mwh"].sum(),
    #     sum(charge_events_capacity_mwh) + losses_mwh,
    # )
    print(
        f'import power: {simulation["site-import_power_mwh"].sum()}, charge event {sum(charge_events_capacity_mwh)}, {losses_mwh=}'
    )

    np.testing.assert_allclose(
        simulation["total-electric_charge_mwh"] * (1 - efficiency),
        simulation["total-electric_loss_mwh"],
    )
    #  TODO in future add charger efficiency
    #  this would allow matching of different efficiency chargers and charge events


def test_v2g():
    charge_events_capacity_mwh = [50, 100, 30, 40]
    evs = epl.evs.EVs(
        chargers_power_mw=[100, 100],
        charge_events_capacity_mwh=charge_events_capacity_mwh,
        charger_turndown=0.0,
        charge_event_efficiency=1.0,
    )
    results = evs.optimize(
        electricity_prices=[-100, 50, 30, 50, 40],
        charge_events=[
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0],
        ],
        flags=Flags(
            allow_evs_discharge=True,
            fail_on_spill_asset_use=True,
            allow_infeasible=False,
        ),
        freq_mins=60,
    )
    simulation = results.simulation

    from energypylinear.debug import debug_simulation

    debug_simulation(results.simulation)
    """
    how to actuall test this
    - check that some of the electric_discharge_mwh is positive

    check final soc are correct
    """


@hypothesis.settings(
    print_blob=True,
    max_examples=200,
    verbosity=hypothesis.Verbosity.verbose,
    deadline=4000,
)
@hypothesis.given(
    idx_length=hypothesis.strategies.integers(min_value=10, max_value=24),
    n_chargers=hypothesis.strategies.integers(min_value=4, max_value=24),
    charger_turndown=hypothesis.strategies.floats(min_value=0.1, max_value=0.4),
    charge_length=hypothesis.strategies.integers(min_value=2, max_value=22),
    prices_mu=hypothesis.strategies.floats(min_value=-1000, max_value=1000),
    prices_std=hypothesis.strategies.floats(min_value=0.1, max_value=25),
    v2g=hypothesis.strategies.booleans(),
)
def test_evs_hypothesis(
    idx_length: int,
    n_chargers: int,
    charger_turndown: float,
    charge_length: int,
    prices_mu: float,
    prices_std: float,
    v2g: bool,
) -> None:
    """Test EV optimization using hypothesis."""
    ds = epl.data_generation.generate_random_ev_input_data(
        idx_length,
        n_chargers,
        charge_length,
        prices_mu=prices_mu,
        prices_std=prices_std,
    )
    evs = epl.evs.EVs(
        chargers_power_mw=ds["charger_mws"].tolist(),
        charge_events_capacity_mwh=ds["charge_events_capacity_mwh"].tolist(),
        charger_turndown=charger_turndown,
    )
    ds.pop("charger_mws")
    ds.pop("charge_events_capacity_mwh")
    evs.optimize(
        **ds,
        verbose=True,
        flags=Flags(
            allow_evs_discharge=v2g,
            fail_on_spill_asset_use=False,
            allow_infeasible=False,
        ),
    )
