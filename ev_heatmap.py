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
    results.to_csv("temp2.csv")
    """
    would like two heat maps
    - one showing the charge event energy charge versus idx
    - one showing the charger use versus idx

    could also show model error
    """
    charger_usage = results[
        [
            c
            for c in results.columns
            if c.startswith("charger-") and c.endswith("-charge_mwh")
        ]
    ].values
    charge_event_usage = results[
        [
            c
            for c in results.columns
            if c.startswith("charge-event-") and c.endswith("total-charge_mwh")
        ]
    ].values

    import pathlib

    import matplotlib.pyplot as plt
    import seaborn

    fig, axes = plt.subplots(
        ncols=3, figsize=(14, 6), width_ratios=(5, 10, 1), sharey=True
    )

    heatmap_config = {
        "annot_kws": {
            "size": 6,
        },
        "annot": True,
    }
    data = charger_usage
    seaborn.heatmap(data, ax=axes[0], **heatmap_config, mask=data == 0)

    data = charge_event_usage
    #  want to unmask out the periods where charge_event was positive
    seaborn.heatmap(
        data,
        ax=axes[1],
        **heatmap_config,
        mask=evs.interval_data.evs.charge_events == 0
    )

    spill_charge_usage = results["charger-spill-charge_mwh"].values.reshape(-1, 1)
    data = spill_charge_usage
    seaborn.heatmap(data, ax=axes[2], **heatmap_config, mask=data >= 0)

    # data = prices
    # seaborn.heatmap(data, ax=axes[2], **heatmap_config, mask=data == 0)

    plt.tight_layout()
    pathlib.Path("./figs").mkdir(
        exist_ok=True,
        parents=True,
    )
    fig.savefig("./figs/evs.png")
