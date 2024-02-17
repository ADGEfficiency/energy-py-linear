"""Models for interval data for electricity & gas prices, thermal loads and carbon intensities."""
import typing

import numpy as np

import energypylinear as epl
from energypylinear.assets.asset import AssetOneInterval
from energypylinear.assets.site import SiteOneInterval

floats = typing.Union[float, np.ndarray, typing.Sequence[float], list[float]]

AssetOneIntervalType = typing.TypeVar("AssetOneIntervalType", bound=AssetOneInterval)


class IntervalVars:
    """Interval data of linear program variables."""

    def __init__(self) -> None:
        """Initializes the interval variables object."""
        self.objective_variables: list[list[AssetOneInterval]] = []

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<epl.IntervalVars i: {len(self.objective_variables)}>"

    def __len__(self) -> int:
        """Return the number of objective variable lists."""
        return len(self.objective_variables)

    def __getitem__(self, index: int) -> list[AssetOneInterval]:
        """Enable subscripting to get a list of AssetOneInterval at a given index."""
        return self.objective_variables[index]

    def append(self, one_interval: AssetOneInterval | SiteOneInterval | list) -> None:
        """Appends a one_interval object to the appropriate attribute.

        Args:
            one_interval (Union[AssetOneInterval, SiteOneInterval, list[AssetOneInterval]]): The interval data to append.
        """
        assert isinstance(one_interval, list)
        self.objective_variables.append(one_interval)

    def filter_objective_variables(
        self,
        i: int,
        instance_type: type[AssetOneInterval] | str | None = None,
        asset_name: str | None = None,
    ) -> list[AssetOneInterval]:
        """Filters objective variables based on type, interval index, and asset name."""
        if isinstance(instance_type, str):
            type_mapper: dict[str, type] = {
                "battery": epl.assets.battery.BatteryOneInterval,
                "boiler": epl.assets.boiler.BoilerOneInterval,
                "chp": epl.assets.chp.CHPOneInterval,
                "evs": epl.assets.evs.EVOneInterval,
                "heat-pump": epl.assets.heat_pump.HeatPumpOneInterval,
                "renewable-generator": epl.assets.renewable_generator.RenewableGeneratorOneInterval,
                "site": SiteOneInterval,
                "spill": epl.assets.spill.SpillOneInterval,
                "spill_evs": epl.assets.evs.EVSpillOneInterval,
            }
            instance_type = type_mapper[instance_type]

        if instance_type is not None:
            assert issubclass(instance_type, AssetOneInterval)

        assets = self.objective_variables[i]
        return [
            asset
            for asset in assets
            if (isinstance(asset, instance_type) if instance_type is not None else True)
            and (asset.cfg.name == asset_name if asset_name is not None else True)
        ]

    def filter_objective_variables_all_intervals(
        self,
        instance_type: type[AssetOneInterval] | str | None = None,
        asset_name: str | None = None,
    ) -> list[list[AssetOneInterval]]:
        """Filters objective variables based on type, interval index, and asset name."""
        pkg = []
        for assets_one_interval in self.objective_variables:
            if isinstance(instance_type, str):
                type_mapper: dict[str, type | None] = {
                    "battery": epl.assets.battery.BatteryOneInterval,
                    "boiler": epl.assets.boiler.BoilerOneInterval,
                    "chp": epl.assets.chp.CHPOneInterval,
                    "evs": epl.assets.evs.EVOneInterval,
                    "heat-pump": epl.assets.heat_pump.HeatPumpOneInterval,
                    "renewable-generator": epl.assets.renewable_generator.RenewableGeneratorOneInterval,
                    "site": SiteOneInterval,
                    "spill": epl.assets.spill.SpillOneInterval,
                    "spill_evs": epl.assets.evs.EVSpillOneInterval,
                    "*": None,
                }
                instance_type = type_mapper[instance_type]

            pkg.append(
                [
                    asset
                    for asset in assets_one_interval
                    if (
                        isinstance(asset, instance_type)
                        if instance_type is not None
                        else True
                    )
                    and (
                        asset.cfg.name == asset_name if asset_name is not None else True
                    )
                ]
            )
        return pkg
