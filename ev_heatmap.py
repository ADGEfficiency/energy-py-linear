import pandas as pd

import energypylinear as epl


if __name__ == "__main__":
    idx_length = 24
    n_chargers = 5
    charge_length = 8
    charger_turndown = 0.1
    ds = epl.data_generation.generate_random_ev_input_data(
        idx_length, n_chargers, charge_length, n_charge_events=12
    )
    pd.DataFrame(ds["charger_mws"]).to_csv("temp.csv")
    evs = epl.evs.EVs(
        charger_mws=ds["charger_mws"].tolist(), charger_turndown=charger_turndown
    )
    ds.pop("charger_mws")
    results = evs.optimize(**ds)
