import collections

import energypylinear as epl
from energypylinear.defaults import defaults


def wip():
    assets = [
        epl.Battery(
            power_mw=20,
            capacity_mwh=40,
            efficiency=0.9,
        ),
        epl.Battery(power_mw=40, capacity_mwh=10, efficiency=0.8, name="battery-2"),
        epl.Generator(
            electric_power_max_mw=100,
            electric_power_min_mw=50,
            electric_efficiency_pct=0.3,
            high_temperature_efficiency_pct=0.5,
        ),
    ]

    site = epl.Site()

    participant = epl.Participant(
        name="retailer",
        tariffs=[
            Tariff(asset="site", quantity="import_power_mwh", rate=np.array([1, 2])),
            Tariff(asset="site", quantity="export_power_mwh", rate=np.array([1, 2])),
        ],
    )

    planet = epl.Participant(
        name="planet",
        tariffs=[
            Tariff(
                quantity="import_power_mwh", rate=np.array([1, 2]), name="grid_carbon"
            )
        ],
    )

    participants = [participant, planet]

    #  TODO
    """
    how to do this?
    - do I want multiple loads - hmmmmm

    I think load can just be another asset

    """
    loads = None

    #  accounts for all participants

    #  TODO think about interval data stuff
    accounts = site.optimize(
        assets=assets, participants=participants, objective="retailer"
    )


"""
## What the Site API Doesn't Do

- scenarios
- baselines
- complex tariffs - leave til later
- inflation / future stuff

## How to make the objective function work

Participants -> tariffs -> objective function

Need to make the objective function from a list of tariffs

objective = create_objective(
    participant.tariffs, vars
)

- a tariff applies to one asset
- the tariff attaches a rate to a quantity
- quantity comes from the OneInterval stuff
- the site asset serves as the sum / balance of all assets

## How to make `asset.optimize` work

participant level optimization only enabled for the site API - for any smaller asset, it just uses the data that goes in

- currently those assets get electricity_prices, gas prices, electricity_carbon_intensities as inputs

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

This allows the public interface of `electricity_prices` to integrate with the `Tariff` or `Flow` API

it's up to the Asset.__init__ to implement ways for costs to be assigned to attirbutes of the one_interval class

- do manually
- maybe need a automatic way as well - something like `quantity=power_generation_mwh, rate=rate` which get automatically propagated into the objective function


## Examples

### How to implement a PPA

- virtual PPA = just swap cashflows
- on site ppa, off site etc

### How to implement a classic retailer / customer / network thing

"""

import numpy as np
import pydantic


class Tariff(pydantic.BaseModel):
    asset: str
    quantity: str
    rate: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class Participant(pydantic.BaseModel):
    name: str
    tariffs: list[Tariff]

    class Config:
        arbitrary_types_allowed = True


def get_participants():
    return {
        "site": Participant(
            name="site",
            tariffs=[
                Tariff(
                    asset="site", quantity="import_power_mwh", rate=np.array([1, 2])
                ),
                Tariff(
                    asset="site", quantity="export_power_mwh", rate=np.array([1, 2])
                ),
            ],
        )
    }


def create_objective(optimizer, vars, interval_data, tariffs):
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
    objective = create_objective(optimizer, vars, interval_data, participant.tariffs)
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
    participant = participants["site"]

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
        objective_fn=create_objective,
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
