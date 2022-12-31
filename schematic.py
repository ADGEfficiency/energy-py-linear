"""
Want to draw a diagram / plot of the system over time
"""

import matplotlib
import matplotlib.pyplot as plt
import pydantic

import energypylinear as epl


class SchematicConfig(pydantic.BaseModel):
    fig_height_cm: float = 10
    fig_width_cm: float = 13
    dpi: float = 300
    line_color: str = "dimgrey"

    #  headers
    header_x: float = 1.0
    electric_header_height: float = 8
    ht_header_height: float = 5
    lt_header_height: float = 2

    #  assets
    generator_height: float = 6
    ht_height: float = 3
    lt_height: float = 0


def get_fig(cfg: SchematicConfig, remove_ticks: bool = False):
    fig, ax = plt.subplots(
        figsize=[cfg.fig_width_cm / 2.54, cfg.fig_height_cm / 2.54], dpi=cfg.dpi
    )

    #  set ax limits so 1 unit = 1 cm
    ax.set_xlim(0, cfg.fig_width_cm)
    ax.set_ylim(0, cfg.fig_height_cm)
    ax.autoscale_view("tight")

    #  turn off tick labels
    if remove_ticks:
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        ax.tick_params(
            labelbottom=False, labeltop=False, labelleft=False, labelright=False
        )
    plt.tight_layout()
    return fig, ax


def plot_header_or_busbar(
    x: float,
    y: float,
    name: str,
):
    #  title
    ax.annotate(name, (0.3, x + 0.1), horizontalalignment="left")
    #  line
    ax.plot((y, 12), (x, x), color="black")


def draw_generator(
    x,
    y,
    electric=True,
    high_temperature=True,
    low_temperature=True,
):
    patches = [
        matplotlib.patches.Rectangle(xy=(x, y), width=1.0, height=1.0),
    ]
    if high_temperature:
        patches.append(
            matplotlib.patches.FancyArrow(
                x + 1 / 3, 6, 0, -1, width=0.05, length_includes_head=True
            )
        )
    if low_temperature:
        patches.append(
            matplotlib.patches.FancyArrow(
                x + 2 / 3, 6, 0, -4, width=0.05, length_includes_head=True
            )
        )
    if electric:
        patches.append(
            matplotlib.patches.FancyArrow(
                x + 1 / 2, 7, 0, 1, width=0.05, length_includes_head=True
            )
        )
    return patches


def draw_battery(x, y):
    patches = [
        matplotlib.patches.Rectangle(xy=(x, y), width=1.0, height=1.0),
    ]
    patches.append(
        matplotlib.patches.FancyArrow(
            x + 1 / 3, 7, 0, 1, width=0.05, length_includes_head=True
        )
    )
    patches.append(
        matplotlib.patches.FancyArrow(
            x + 2 / 3, 8, 0, -1, width=0.05, length_includes_head=True
        )
    )
    return patches


def draw_load(x, y, header_height):
    patches = [
        matplotlib.patches.Rectangle(xy=(x, y), width=1.0, height=1.0),
    ]
    patches.append(
        matplotlib.patches.FancyArrow(
            x + 1 / 2, header_height, 0, -1, width=0.05, length_includes_head=True
        )
    )
    return patches


if __name__ == "__main__":
    cfg = SchematicConfig()

    asset = epl.chp.Generator(
        electric_power_max_mw=100,
        electric_power_min_mw=50,
        electric_efficiency_pct=0.3,
        high_temperature_efficiency_pct=0.5,
    )
    results = asset.optimize(
        electricity_prices=[1000, -100, 1000],
        gas_prices=20,
        high_temperature_load_mwh=[20, 20, 1000],
        freq_mins=60,
    )

    fig, ax = get_fig(cfg, remove_ticks=False)

    plot_header_or_busbar(8.0, cfg.header_x, "Electric")
    plot_header_or_busbar(cfg.ht_header_height, cfg.header_x, "HT")
    plot_header_or_busbar(cfg.lt_header_height, cfg.header_x, "LT")

    collection = []
    collection.extend(draw_generator(6, cfg.generator_height))
    collection.extend(draw_generator(8, cfg.generator_height))
    collection.extend(draw_battery(10, cfg.generator_height))
    collection.extend(draw_load(2, cfg.generator_height, cfg.electric_header_height))
    collection.extend(draw_load(2, cfg.ht_height, cfg.ht_header_height))
    collection.extend(draw_load(2, cfg.lt_height, cfg.lt_header_height))

    collection.extend(draw_boiler(2, cfg.generator_height, cfg.electric_header_height))

    pc = matplotlib.collections.PatchCollection(collection)
    ax.add_collection(pc)
    fig.savefig("./temp.png")
