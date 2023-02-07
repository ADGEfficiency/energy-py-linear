"""Utilites for generating interval data"""
import numpy as np


def generate_random_ev_input_data(
    idx_length: int,
    n_chargers: int,
    charge_length: int,
    normal_mu: float = 0,
    normal_std: float = 1,
    uniform_min: float = 0,
    uniform_max: float = 10,
    n_charge_events: int = 10,
    prices_mu: float = 100,
    prices_std: float = 20,
) -> dict:
    """Create interval data for the `epl.evs.EVs` smart electric charging asset."""
    np.random.seed(2)
    electricity_prices = np.random.normal(prices_mu, prices_std, idx_length)
    charger_mws = np.random.randint(10, 100, n_chargers)

    charge_events = np.zeros((n_charge_events, idx_length))
    charge_length = min(idx_length - 2, charge_length)

    for step in range(charge_events.shape[0]):
        length = int(np.random.randint(1, charge_length, 1))
        start = int(np.random.randint(0, charge_events.shape[1] - length - 1, 1))
        charge_events[step, start: start + length] = 1  # fmt: skip

    charge_event_mwh = np.random.normal(
        normal_mu, normal_std, n_charge_events
    ) + np.random.uniform(uniform_min, uniform_max, n_charge_events)
    charge_event_mwh = np.clip(charge_event_mwh, a_min=0, a_max=None)

    return {
        "electricity_prices": electricity_prices.tolist(),
        "charger_mws": charger_mws,
        "charge_event_mwh": charge_event_mwh,
        "charge_events": charge_events,
    }
