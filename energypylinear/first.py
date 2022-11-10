"""
Asset
"""
import typing

import pulp
import pydantic


class Pulp:
    def __init__(self):
        self.prob = pulp.LpProblem()

    def continuous(self, name: str, low: float = 0, up: typing.Optional[float] = None):
        return pulp.LpVariable(name=name, lowBound=low, upBound=up, cat="Continuous")

    def binary(self, name: str):
        return pulp.LpVariable(name=name, cat="Binary")

    def sum(self, vector: list[pulp.LpAffineExpression]):
        return pulp.lpSum(vector)

    def constrain(self, constraint, name=None):
        return self.prob.addConstraint(constraint, name)


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
    framework: Pulp, battery: BatteryCfg, i: int, freq: Freq
) -> BatteryOneInterval:
    return BatteryOneInterval(
        charge_mwh=framework.continuous(
            f"charge_mwh-{i}", up=freq.mw_to_mwh(battery.power_mw)
        ),
        discharge_mwh=framework.continuous(
            f"discharge_mwh-{i}", up=freq.mw_to_mwh(battery.power_mw)
        ),
        losses_mwh=framework.continuous(f"losses_mwh-{i}"),
        initial_charge_mwh=framework.continuous(f"initial_charge_mwh-{i}"),
        final_charge_mwh=framework.continuous(f"final_charge_mwh-{i}"),
    )


class Battery:
    def __init__(self, power_mw: float, capacity_mwh: float, efficiency: float = 0.9):
        self.cfg = BatteryCfg(
            power_mw=float(power_mw),
            capacity_mwh=float(capacity_mwh),
            efficiency_pct=float(efficiency),
        )
        self.framework = Pulp()

    def optimize(
        self, prices: list[float], freq_mins: int, initial_charge_mwh: float = 0.0
    ):
        """
        TODO
        - carbon, forecasts, carbon_price, initial charge, objective
        """
        self.site_cfg = SiteCfg()

        freq = Freq(freq_mins)

        #  probably want to create some sort of interval data object here
        #  that has prices and carbon intensities

        for i in range(len(prices)):
            #  setup linear program variables
            site = site_one_interval(self.framework, self.site_cfg, i, freq)
            battery = battery_one_interval(self.framework, self.cfg, i, freq)

            assets = [battery]

            def constrain_site_electricity_balance(framework, site, assets):
                """
                in = out + accumulation
                import + generation = (export + load) + (charge - discharge)
                import + generation - (export + load) - (charge - discharge) = 0

                should losses occur here?
                """
                framework.constrain(
                    site.import_power_mwh
                    + framework.sum([a.generation_mwh for a in assets])
                    - (
                        site.export_power_mwh
                        + framework.sum([a.load_mwh for a in assets])
                    )
                    - (
                        framework.sum([a.charge_mwh for a in assets])
                        - framework.sum([a.discharge_mwh for a in assets])
                    )
                    == 0
                )

            def constrain_site_import_export(framework, site, freq):
                framework.constrain(
                    site.import_power_mwh
                    - site.import_limit_mwh * site.import_power_bin
                    <= 0
                )
                framework.constrain(
                    site.export_power_mwh
                    - site.export_limit_mwh * site.export_power_bin
                    <= 0
                )
                framework.constrain(site.import_power_bin + site.export_power_bin == 1)

            #  setup site constraints
            constrain_site_electricity_balance(self.framework, site, assets)
            constrain_site_import_export(self.framework, site, freq)


asset = Battery(power_mw=2, capacity_mwh=4, efficiency=0.95)
results = asset.optimize(prices=[10, 50, 10, 50, 10], freq_mins=30)
