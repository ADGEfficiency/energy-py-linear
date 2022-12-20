import hypothesis
import numpy as np
import pandas as pd

import energypylinear as epl


def generate_random_ev_input_data(
    idx_length,
    n_chargers,
    charge_length,
    normal_mu=0,
    normal_std=1,
    uniform_min=0,
    uniform_max=10,
    n_charge_events=10,
    prices_mu=100,
    prices_std=20,
):
    np.random.seed(2)
    electricity_prices = np.random.normal(prices_mu, prices_std, idx_length)
    charger_mws = np.random.randint(10, 100, n_chargers)

    charge_events = np.zeros((n_charge_events, idx_length))
    charge_length = min(idx_length - 2, charge_length)

    for step in range(charge_events.shape[0]):
        length = int(np.random.randint(1, charge_length, 1))
        start = int(np.random.randint(0, charge_events.shape[1] - length - 1, 1))
        charge_events[step, start : start + length] = 1
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
