"""Models for interval data for electricity & gas prices, thermal loads and carbon intensities."""
import collections
import typing

import numpy as np

from energypylinear.assets.asset import AssetOneInterval
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

    def __len__(self) -> int:
        """Return the number of objective variable lists."""
        return len(self.objective_variables)

    def __getitem__(self, index: int) -> list[AssetOneInterval]:
        """Enable subscripting to get a list of AssetOneInterval at a given index."""
        return self.objective_variables[index]

    def append(
        self, one_interval: AssetOneInterval | SiteOneInterval | list[AssetOneInterval]
    ) -> None:
        """Appends a one_interval object to the appropriate attribute.

        Args:
            one_interval (Union[AssetOneInterval, SiteOneInterval, list[AssetOneInterval]]): The interval data to append.
        """
        if isinstance(one_interval, SiteOneInterval):
            self.asset[one_interval.cfg.name]["site"].append(one_interval)

        else:
            assert isinstance(one_interval, list)
            self.objective_variables.append(one_interval)

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
    ) -> list[list[AssetOneInterval]] | list[AssetOneInterval]:
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

            # used to return list of lists
            assets = self.objective_variables[i]
            return [
                asset
                for asset in assets
                if isinstance(asset, instance_type)
                and (asset.cfg.name == asset_name if asset_name is not None else True)
            ]
