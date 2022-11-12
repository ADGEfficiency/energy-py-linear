"""
Asset
"""
import collections
import typing

import pulp
import pydantic


class IntervalData:
    default_price: float = 100
    default_carbon_intensity: float = 0.5

    def __init__(
        self,
        prices: typing.Optional[typing.Union[float, list[float]]] = None,
        forecasts: typing.Optional[list[float]] = None,
        carbon_intensities: typing.Optional[typing.Union[float, list[float]]] = None,
    ):
        assert any((prices, carbon_intensities))

        if prices is None:
            assert isinstance(carbon_intensities, list)
            prices = [self.default_price for _ in range(len(carbon_intensities))]

        if carbon_intensities is None:
            assert isinstance(prices, list)
            carbon_intensities = [
                self.default_carbon_intensity for _ in range(len(prices))
            ]

        if forecasts is None:
            assert isinstance(prices, list)
            forecasts = prices

        assert isinstance(prices, list)
        assert isinstance(carbon_intensities, list)
        assert isinstance(forecasts, list)
        assert len(prices) == len(forecasts) == len(carbon_intensities)

        self.prices = prices
        self.forecasts = forecasts
        self.carbon_intensities = carbon_intensities
        self.idx = range(len(self.prices))


class Pulp:
    def __init__(self):
        self.prob = pulp.LpProblem()
        self.solver = pulp.PULP_CBC_CMD(msg=0)

    def continuous(self, name: str, low: float = 0, up: typing.Optional[float] = None):
        return pulp.LpVariable(name=name, lowBound=low, upBound=up, cat="Continuous")

    def binary(self, name: str):
        return pulp.LpVariable(name=name, cat="Binary")

    def sum(self, vector: list[pulp.LpAffineExpression]):
        return pulp.lpSum(vector)

    def constrain(
        self, constraint: pulp.LpConstraint, name: typing.Optional[str] = None
    ):
        return self.prob.addConstraint(constraint, name)

    def objective(self, objective):
        return self.prob.setObjective(objective)

    def solve(self):
        self.solver.solve(self.prob)
        return self.status()

    def status(self):
        return pulp.LpStatus[self.prob.status]


class SiteCfg(pydantic.BaseModel):
    import_limit_mw: float = 10000
    export_limit_mw: float = 10000


class SiteOneInterval(pydantic.BaseModel):
    import_power_mwh: pulp.LpVariable
    export_power_mwh: pulp.LpVariable
    import_power_bin: pulp.LpVariable
    export_power_bin: pulp.LpVariable

    import_limit_mwh: float
    export_limit_mwh: float

    class Config:
        arbitrary_types_allowed: bool = True


class BatteryCfg(pydantic.BaseModel):
    power_mw: float
    capacity_mwh: float
    efficiency_pct: float
    initial_charge_mwh: float = 0
    final_charge_mwh: float = 0


class Asset(pydantic.BaseModel):
    generation_mwh: typing.Union[pulp.LpVariable, float] = 0
    load_mwh: typing.Union[pulp.LpVariable, float] = 0
    charge_mwh: typing.Union[pulp.LpVariable, float] = 0
    discharge_mwh: typing.Union[pulp.LpVariable, float] = 0

    class Config:
        arbitrary_types_allowed: bool = True


class BatteryOneInterval(Asset):
    charge_mwh: pulp.LpVariable
    discharge_mwh: pulp.LpVariable
    losses_mwh: pulp.LpVariable
    initial_charge_mwh: pulp.LpVariable
    final_charge_mwh: pulp.LpVariable
    efficiency_pct: float


class Freq:
    def __init__(self, mins: int):
        self.mins = mins

    def mw_to_mwh(self, mw: float):
        return mw * 60 / self.mins


def site_one_interval(
    framework: Pulp, site: SiteCfg, i: int, freq: Freq
) -> SiteOneInterval:
    return SiteOneInterval(
        import_power_mwh=framework.continuous(
            f"import_power_mw-{i}", up=freq.mw_to_mwh(site.import_limit_mw)
        ),
        export_power_mwh=framework.continuous(
            f"export_power_mw-{i}", up=freq.mw_to_mwh(site.import_limit_mw)
        ),
        import_power_bin=framework.binary(f"import_power_bin-{i}"),
        export_power_bin=framework.binary(f"export_power_bin-{i}"),
        import_limit_mwh=freq.mw_to_mwh(site.import_limit_mw),
        export_limit_mwh=freq.mw_to_mwh(site.export_limit_mw),
    )


def battery_one_interval(
    framework: Pulp, cfg: BatteryCfg, i: int, freq: Freq
) -> BatteryOneInterval:
    return BatteryOneInterval(
        charge_mwh=framework.continuous(
            f"charge_mwh-{i}", up=freq.mw_to_mwh(cfg.power_mw)
        ),
        discharge_mwh=framework.continuous(
            f"discharge_mwh-{i}", up=freq.mw_to_mwh(cfg.power_mw)
        ),
        losses_mwh=framework.continuous(f"losses_mwh-{i}"),
        initial_charge_mwh=framework.continuous(f"initial_charge_mwh-{i}"),
        final_charge_mwh=framework.continuous(f"final_charge_mwh-{i}"),
        efficiency_pct=cfg.efficiency_pct,
    )


def constrain_battery_electricity_balance(
    framework: Pulp, assets: collections.defaultdict
) -> None:
    for battery in assets["batteries"][-1]:
        framework.constrain(
            battery.initial_charge_mwh
            + battery.charge_mwh
            - battery.discharge_mwh
            - battery.losses_mwh
            == battery.final_charge_mwh
        )
        framework.constrain(
            battery.losses_mwh == battery.charge_mwh * (1 - battery.efficiency_pct)
        )


def constrain_connection_batteries_between_intervals(
    framework: Pulp, assets: collections.defaultdict
) -> None:
    batteries = assets["batteries"]

    #  if in first interval, do nothing
    #  could also do something based on `i` here...
    if len(batteries) < 2:
        return None

    else:
        old = batteries[-2]
        new = batteries[-1]
        for alt, neu in zip(old, new, strict=True):
            framework.constrain(alt.final_charge_mwh == neu.initial_charge_mwh)


def constrain_initial_final_charge(
    framework: Pulp, assets: collections.defaultdict, battery_cfgs: list[BatteryCfg]
) -> None:

    batteries = assets["batteries"]
    first = batteries[0]
    for battery, cfg in zip(first, battery_cfgs, strict=True):
        framework.constrain(battery.initial_charge_mwh == cfg.initial_charge_mwh)

    last = batteries[-1]
    for battery, cfg in zip(last, battery_cfgs, strict=True):
        framework.constrain(battery.final_charge_mwh == cfg.final_charge_mwh)


def constrain_site_electricity_balance(framework, site, assets):
    """
    in = out + accumulation
    import + generation = (export + load) + (charge - discharge)
    import + generation - (export + load) - (charge - discharge) = 0

    should losses occur here?
    """
    assets = assets["assets"][-1]
    framework.constrain(
        site.import_power_mwh
        + framework.sum([a.generation_mwh for a in assets])
        - (site.export_power_mwh + framework.sum([a.load_mwh for a in assets]))
        - (
            framework.sum([a.charge_mwh for a in assets])
            - framework.sum([a.discharge_mwh for a in assets])
        )
        == 0
    )


def constrain_site_import_export(framework, site):
    framework.constrain(
        site.import_power_mwh - site.import_limit_mwh * site.import_power_bin <= 0
    )
    framework.constrain(
        site.export_power_mwh - site.export_limit_mwh * site.export_power_bin <= 0
    )
    framework.constrain(site.import_power_bin + site.export_power_bin == 1)


class Battery:
    def __init__(self, power_mw: float, capacity_mwh: float, efficiency: float = 0.9):
        self.cfg = BatteryCfg(
            power_mw=float(power_mw),
            capacity_mwh=float(capacity_mwh),
            efficiency_pct=float(efficiency),
        )
        self.framework = Pulp()

    def optimize(
        self,
        freq_mins: int,
        prices: typing.Optional[list[float]],
        forecasts: typing.Optional[list[float]] = None,
        carbon_intensities: typing.Optional[list[float]] = None,
        initial_charge_mwh: float = 0.0,
        objective: typing.Literal["price", "forecast", "carbon"] = "price",
    ):
        self.site_cfg = SiteCfg()

        self.cfg.initial_charge_mwh = initial_charge_mwh
        self.cfg.final_charge_mwh = initial_charge_mwh

        assert objective in {"price", "forecast", "carbon"}
        objective = objective

        freq = Freq(freq_mins)

        interval_data = IntervalData(
            prices=prices, forecasts=forecasts, carbon_intensities=carbon_intensities
        )

        assets = collections.defaultdict(list)

        for i in interval_data.idx:
            #  setup linear program variables
            site = site_one_interval(self.framework, self.site_cfg, i, freq)
            assets["sites"].append(site)

            battery = battery_one_interval(self.framework, self.cfg, i, freq)
            batteries = [battery]
            assets["assets"].append(batteries)
            assets["batteries"].append(batteries)

            #  setup constraints

            #  site constraints
            constrain_site_electricity_balance(self.framework, site, assets)
            constrain_site_import_export(self.framework, site)

            #  battery constraints
            constrain_battery_electricity_balance(self.framework, assets)
            constrain_connection_batteries_between_intervals(self.framework, assets)

        assert (
            len(interval_data.idx)
            == len(assets["assets"])
            == len(assets["batteries"])
            == len(assets["sites"])
        )
        constrain_initial_final_charge(self.framework, assets, [self.cfg])

        #  objective functions
        sites = assets["sites"]
        price_objective = self.framework.sum(
            sites[i].import_power_mwh * interval_data.prices[i]
            - sites[i].export_power_mwh * interval_data.prices[i]
            for i in interval_data.idx
        )
        forecast_objective = self.framework.sum(
            sites[i].import_power_mwh * interval_data.forecasts[i]
            - sites[i].export_power_mwh * interval_data.forecasts[i]
            for i in interval_data.idx
        )
        carbon_objective = self.framework.sum(
            sites[i].import_power_mwh * interval_data.carbon_intensities[i]
            - sites[i].export_power_mwh * interval_data.carbon_intensities[i]
            for i in interval_data.idx
        )

        objectives = {
            "price": price_objective,
            "forecast": forecast_objective,
            "carbon": carbon_objective,
        }

        self.framework.objective(objectives[objective])
        status = self.framework.solve()
        print(status)


asset = Battery(power_mw=2, capacity_mwh=4, efficiency=0.95)
results = asset.optimize(prices=[10, 50, 10, 50, 10], freq_mins=30)
