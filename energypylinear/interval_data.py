"""Models for interval data for electricity & gas prices, thermal loads and carbon intensities."""
import collections
import typing

import numpy as np

from energypylinear.assets.asset import AssetOneInterval
from energypylinear.assets.evs import EVsArrayOneInterval
from energypylinear.assets.site import SiteOneInterval

floats = typing.Union[float, np.ndarray, typing.Sequence[float], list[float]]


class IntervalVars:
    """Interval data of linear program variables."""

    def __init__(self) -> None:
        """Initializes the interval variables object."""
        #  not every lp variable - only the ones we want to iterate over
        #  in the objective functions (price, carbon etc)
        self.objective_variables: list[list[AssetOneInterval]] = []
        self.asset: collections.defaultdict = collections.defaultdict(
            lambda: {"evs_array": [], "spill_evs_array": [], "site": []}
        )

    def __repr__(self) -> str:
        """A string representation of self."""
        return f"<epl.IntervalVars i: {len(self.objective_variables)}>"

    def append(
        self, one_interval: AssetOneInterval | SiteOneInterval | list[AssetOneInterval]
    ) -> None:
        """Appends a one_interval object to the appropriate attribute.

        Args:
            one_interval (Union[AssetOneInterval, SiteOneInterval, list[AssetOneInterval]]): The interval data to append.

        Raises:
            AssertionError: If one_interval is not a recognized type.
        """
        #  some OneInterval objects are special
        #  is this case it is the Array EV data structures
        #  TODO in future don't save these separately and
        #  dynamically create as needed from the objective variables
        if isinstance(one_interval, EVsArrayOneInterval):
            if one_interval.is_spill:
                self.asset[one_interval.cfg.name]["spill_evs_array"].append(
                    one_interval
                )
            else:
                self.asset[one_interval.cfg.name]["evs_array"].append(one_interval)
        elif isinstance(one_interval, SiteOneInterval):
            self.asset[one_interval.cfg.name]["site"].append(one_interval)

        else:
            assert isinstance(one_interval, list)
            self.objective_variables.append(one_interval)

    def filter_evs_array(
        self, is_spill: bool, i: int, asset_name: str
    ) -> EVsArrayOneInterval:
        """Filters and returns EVsArrayOneInterval data based on criteria.

        Args:
            is_spill (bool): Whether to filter spill EVs or regular EVs.
            i (int): Interval index.
            asset_name (str): Name of the asset.

        Returns:
            EVsArrayOneInterval: The filtered EVsArrayOneInterval object.
        """
        if is_spill:
            return self.asset[asset_name]["spill_evs_array"][i]
        else:
            return self.asset[asset_name]["evs_array"][i]

    def filter_all_evs_array(
        self, is_spill: bool, asset_name: str
    ) -> list[EVsArrayOneInterval]:
        """Filters EVsArrayOneInterval instances based on spill status and asset name.

        Args:
            is_spill (bool): Whether to filter by spill or not.
            asset_name (str): Name of the asset to filter by.

        Returns:
            list[EVsArrayOneInterval]: Filtered list of EVsArrayOneInterval instances.
        """
        if is_spill:
            return self.asset[asset_name]["spill_evs_array"]
        else:
            return self.asset[asset_name]["evs_array"]

    def filter_site(self, i: int, site_name: str) -> SiteOneInterval:
        """Filters and a SiteOneInterval based on interval index and site name.

        Args:
            i (int): Interval index.
            site_name (str): Name of the site to filter by.

        Returns:
            SiteOneInterval: Filtered SiteOneInterval instance.
        """
        return self.asset[site_name]["site"][i]

    def filter_objective_variables(
        self,
        instance_type: type[AssetOneInterval],
        i: int | None = None,
        asset_name: str | None = None,
    ) -> list[list[AssetOneInterval]]:
        """Filters objective variables based on type, interval index, and asset name.

        Args:
            instance_type (type[AssetOneInterval]):
                Type of the AssetOneInterval instances to filter.
            i (int | None, optional):
                Interval index. If None, filters across all intervals.
                Defaults to None.
            asset_name (str | None, optional):
                Name of the asset to filter by. If None, no filtering by name.
                Defaults to None.

        Returns:
            list[list[AssetOneInterval]]:
                Filtered list of AssetOneInterval instances.
        """
        #  here we return data for all intervals
        if i is None:
            pkg = []
            for i, assets_one_interval in enumerate(self.objective_variables):
                pkg.append(
                    [
                        asset
                        for asset in assets_one_interval
                        if isinstance(asset, instance_type)
                        and (
                            asset.cfg.name == asset_name
                            if asset_name is not None
                            else True
                        )
                    ]
                )
            return pkg

        #  here we return data for one interval
        else:

            #  why a list of lists??????????
            assets = self.objective_variables[i]
            return [
                [
                    asset
                    for asset in assets
                    if isinstance(asset, instance_type)
                    and (
                        asset.cfg.name == asset_name if asset_name is not None else True
                    )
                ]
            ]
