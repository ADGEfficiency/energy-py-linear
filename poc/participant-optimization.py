"""

# was in log.md
### Participants, Tariffs and Objective Functions

Participants -> tariffs -> objective function

Need to make the objective function from a list of tariffs

objective = create_objective(
    participant.tariffs, vars
)

- a tariff applies to one asset
- the tariff attaches a rate to a quantity
- quantity comes from the OneInterval stuff
- the site asset serves as the sum / balance of all assets



   was in site.py
   if objective_fn is None:
            objective_fn = epl.objectives[objective]
            tariffs = []

        else:
            objective_fn = objective_fn
            assert objective in participants
            tariffs = participants[objective].tariffs



"""
"""
## How to make `asset.optimize` work

participant level optimization only enabled for the site API - for any smaller asset, it just uses the data that goes in.

- currently those assets get electricity_prices, gas prices, electricity_carbon_intensities as inputs - reasoning is to keep this interface for those who don't want to worry about concepts like participants, assets or tariffs

If you want to do participant level optimization for a single asset, you can create a site with one asset

asset knows about the site & participant api, and can implement it during it's `optimize` function

asset also knows about the objective function api, and can implement it

asset.optimize(electricity_prices)
    tariffs = [
        Tariff(asset="site", quantity="import_power_mwh", rate=electricity_prices),
        Tariff(asset="site", quantity="export_power_mwh", rate=electricity_prices),
    ]
    #  wouldn't really need this, but could
    participant = Participant(flows=tariffs, name="global")

    objective = make_objective(tariffs)

- do manually
- maybe need a automatic way as well - something like `quantity=power_generation_mwh, rate=rate` which get automatically propagated into the objective function


## Examples

### How to implement a PPA

- virtual PPA = just swap cashflows
- on site ppa, off site etc

### How to implement a classic retailer / customer / network thing

"""
import collections

import numpy as np
import pydantic

import energypylinear as epl
from energypylinear.defaults import defaults


class Account(pydantic.BaseModel):
    items: int
    cost: float
    revenue: float
    profit: float
    emissions: float


def get_accounts_for_participants(
    interval_data, simulation, participants
) -> dict[str, Account]:
    epl.results.validate_results(interval_data, simulation)

    out = {}
    for p_name, participant in participants.items():
        account_items = {}
        for tariff in participant.tariffs:
            name = f"{tariff.asset}-{tariff.quantity}"
            name = "_".join(name.split("_")[:-1]) + f"_{tariff.output_unit}"
            account_items[name] = (
                simulation[tariff.quantity] * tariff.rate * tariff.cost
            ).sum()

        out[p_name] = account_items
        cost = -sum(
            [
                a
                for name, a in account_items.items()
                if (a < 0) and (name.endswith("_$"))
            ]
        )
        revenue = sum(
            [
                a
                for name, a in account_items.items()
                if (a > 0) and (name.endswith("_$"))
            ]
        )
        emissions = sum(
            [a for name, a in account_items.items() if name.endswith("_tC")]
        )
        print(p_name, cost, revenue, emissions)

        out[p_name] = Account(
            items=len(account_items),
            cost=cost,
            revenue=revenue,
            profit=revenue - cost,
            emissions=emissions,
        )

    return out


class Tariff(pydantic.BaseModel):
    asset: str
    quantity: str
    rate: np.ndarray

    output_unit: str = "$"

    #  -1.0 = cost, 0.0 = off, 1.0 = revenue
    cost: int = -1.0

    class Config:
        arbitrary_types_allowed = True


class Participant(pydantic.BaseModel):
    name: str
    tariffs: list[Tariff]

    class Config:
        arbitrary_types_allowed = True


def get_participants(n=268):
    return {
        "site": Participant(
            name="site",
            tariffs=[
                Tariff(
                    asset="site",
                    quantity="import_power_mwh",
                    rate=np.random.uniform(-1000, 1000, n),
                ),
                Tariff(
                    asset="site",
                    quantity="export_power_mwh",
                    rate=np.random.uniform(-1000, 1000, n),
                ),
            ],
        ),
        "planet": Participant(
            name="planet",
            tariffs=[
                Tariff(
                    asset="site",
                    quantity="import_power_mwh",
                    rate=np.random.uniform(-1, 1, n),
                    output_unit="tC",
                )
            ],
        ),
    }


def create_objective_from_tariffs(optimizer, vars, interval_data, tariffs):
    """
    the list of tariffs fully specifies the objective function

    """
    obj = []
    print(optimizer)

    spills = vars.get("spills")
    #  tariff independent stuff
    if spills:
        obj.extend(
            [
                spills[i].electric_generation_mwh * defaults.spill_objective_penalty
                + spills[i].high_temperature_generation_mwh
                * defaults.spill_objective_penalty
                + spills[i].electric_load_mwh * defaults.spill_objective_penalty
                + spills[i].high_temperature_load_mwh
                for i in interval_data.idx
            ]
        )

    #  TODO rewrite to one loop

    spill_evs = vars.get("spill_evs")
    if spill_evs:
        for i in interval_data.idx:
            obj.extend(
                [
                    spill_ev.charge_mwh * defaults.spill_objective_penalty
                    for spill_ev in spill_evs[i]
                ]
            )
    generators = vars.get("generators")
    if generators:
        for i in interval_data.idx:
            obj.extend(
                [
                    generator.gas_consumption_mwh * interval_data.gas_prices[i]
                    for generator in generators[i]
                ]
            )

    boilers = vars.get("boilers")
    if boilers:
        for i in interval_data.idx:
            obj.extend(
                [
                    boiler.gas_consumption_mwh * interval_data.gas_prices[i]
                    for boiler in boilers[i]
                ]
            )

    sites = vars["sites"]
    for tariff in tariffs:
        #  ignore the multiple loops for now - note the n**2

        if tariff.asset == "site" and tariff.quantity == "import_power_mwh":
            obj.extend(
                sites[i].import_power_mwh * interval_data.electricity_prices[i]
                for i in interval_data.idx
            )

        if tariff.asset == "site" and tariff.quantity == "export_power_mwh":
            obj.extend(
                -1 * sites[i].export_power_mwh * interval_data.electricity_prices[i]
                for i in interval_data.idx
            )
    return optimizer.sum(obj)


def test_create_objective():
    participants = get_participants()
    participant = participants["site"]

    interval_data = epl.interval_data.IntervalData(
        electricity_prices=np.array([20, 30, 100, -10, 0]),
        electricity_load_mwh=np.array([100, 50, 50, 100, 100]),
    )

    vars: collections.defaultdict[str, typing.Any] = collections.defaultdict(list)
    site = epl.assets.site.Site(assets=[])

    optimizer = epl.Optimizer()
    freq = epl.Freq(30)

    for i in interval_data.idx:
        vars["sites"].append(site.one_interval(optimizer, site.cfg, i, freq))
        vars["assets"].append([])
        site.constrain_within_interval(optimizer, vars, interval_data, i)

    assert len(interval_data.idx) == len(vars["assets"]) == len(vars["sites"])

    #  create simple objective
    objective = create_objective_from_tariffs(
        optimizer, vars, interval_data, participant.tariffs
    )
    optimizer.objective(objective)
    status = optimizer.solve(allow_infeasible=False)
    results = epl.results.extract_results(interval_data, vars, feasible=status.feasible)
    assert (
        results.simulation["import_power_mwh"].sum()
        == interval_data.electricity_load_mwh.sum()
    )
    assert results.simulation["export_power_mwh"].sum() == 0

    """
    create an objective equivilant to the price_objective
    then test it's the same as doing objective = price_objective()
    """
    interval_data = epl.interval_data.IntervalData(
        electricity_prices=np.random.uniform(-1000, 1000, 268)
    )

    battery = epl.Battery()

    site = epl.Site(assets=[battery])
    results_left = site.optimize(
        electricity_prices=interval_data.electricity_prices,
        objective="price",
    )

    site = epl.Site(assets=[battery])
    results_right = site.optimize(
        electricity_prices=interval_data.electricity_prices,
        objective="site",
        objective_fn=create_objective_from_tariffs,
        participants=participants,
    )
    np.testing.assert_allclose(
        results_left.simulation["import_power_mwh"],
        results_right.simulation["import_power_mwh"],
    )
    np.testing.assert_allclose(
        results_left.simulation["export_power_mwh"],
        results_right.simulation["export_power_mwh"],
    )
    assert results_left.simulation["export_power_mwh"].sum() > 0


