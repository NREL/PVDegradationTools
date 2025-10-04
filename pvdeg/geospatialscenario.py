"""geospatialscenario.py."""

import pvdeg
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import os
import warnings
import pandas as pd
import xarray as xr
import numpy as np
from typing import List, Union, Optional, Callable
from IPython.display import display, HTML
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from dask.distributed import Client


class GeospatialScenario(pvdeg.Scenario):
    def __init__(
        self,
        name: str = None,
        path: str = None,
        gids: Optional[Union[list, np.ndarray]] = None,
        modules: list = [],
        pipeline: dict = {},
        file=None,
        results=None,
        hpc=False,
        geospatial=False,
        weather_data: xr.Dataset = None,
        meta_data: pd.DataFrame = None,
        func: Callable = None,
        template: xr.Dataset = None,
        dask_client: Client = None,
    ):
        super().__init__(
            name=name,
            path=path,
            gids=gids,
            modules=modules,
            pipeline=pipeline,
            file=file,
            results=results,
            weather_data=weather_data,
            meta_data=meta_data,
        )
        self.geospatial = geospatial
        self.hpc = hpc
        self.func = func
        self.template = template
        self.dask_client = dask_client
        self.kdtree = None  # sklearn kdtree

    def __eq__(self, other):
        raise NotImplementedError(
            """
            Cannot directly compare pvdeg.GeospatialScenario objects
            due to larger than memory/out of memory datasets stored in
            GeospatialScenario.weather_data attribute.
            """
        )

    def start_dask(self, hpc=None) -> None:
        """Start a dask cluster for parallel processing.

        Parameters
        ----------
        hpc : dict
            Dictionary containing dask hpc settings (see examples below).
            Supply `None` for a default configuration.

        Examples
        --------
        Local cluster:

        .. code-block:: python

            hpc = {'manager': 'local',
                'n_workers': 1,
                'threads_per_worker': 8,
                'memory_limit': '10GB'}

        SLURM cluster:

        .. code-block:: python

            kestrel = {
                'manager': 'slurm',
                'n_jobs': 1,  # Max number of nodes used for parallel processing
                'cores': 104,
                'memory': '246GB',
                'account': 'pvsoiling',
                'walltime': '4:00:00',
                'processes': 52,
                'local_directory': '/tmp/scratch',
                'job_extra_directives': ['-o ./logs/slurm-%j.out'],
                'death_timeout': 600,}
        """
        self.dask_client = pvdeg.geospatial.start_dask()

    def addLocation(
        self,
        country: Optional[str] = None,
        state: Optional[str] = None,
        county: Optional[str] = None,
        satellite: str = "Americas",
        year: Union[str, int] = "TMY",
        nsrdb_attributes: List[str] = [
            "air_temperature",
            "wind_speed",
            "dhi",
            "ghi",
            "dni",
            "relative_humidity",
        ],
        downsample_factor: int = 0,
        gids: Optional[Union[int, List[int], np.ndarray]] = None,
        bbox_kwarg: Optional[dict] = {},
        see_added: bool = False,
    ) -> None:
        """
        Add locations to the GeospatialScenario.

        Existing weather and meta data will be overwritten with weather and meta data
        gathered by this method.

        Parameters
        -----------
        country : str
            country to include from NSRDB. Currently supports full string names only.
            Either single string form or list of strings form.
            Examples:
            - ``country='United States'``
            - ``country=['United States']``
            - ``country=['Mexico', 'Canada']``

        state : str
            combination of states or provinces to include from NSRDB.
            Supports two-letter codes for American states. Can mix two-letter
            codes with full length strings. Can take single string, or list of
            strings(len >= 1)

            Examples:
            - ``state='Washington'``
            - ``state=WA`` (state abbr is case insensitive)
            - ``state=['CO', 'British Columbia']``

        county : str
            county to include from NSRDB. If duplicate county exists in two
            states present in the ``state`` argument, both will be included.
            If no state is provided
        downsample_factor : int
            downsample the weather and metadata attached to the region you have
            selected. default(0), means no downsampling
        year : int
            year of data to use from NSRDB, default = ``TMY`` otherwise provide integer
            like ``2022`` for psm3 yearly data.
        nsrdb_attributes : list(str)
            list of strings of weather attributes to grab from the NSRDB, must be valid
            NSRDB attributes (insert list of valid options here).

                Valid Options:
                - 'air_temperature'
                - 'dew_point'
                - 'dhi'
                - 'dni'
                - 'ghi'
                - 'surface_albedo'
                - 'surface_pressure'
                - 'wind_direction'
                - 'wind_speed'

        see_added : bool
            flag true if you want to see a runtime notification for added location/gids
        """
        # overwrite old location information
        self.gids, self.weather_data, self.meta_data = None, None, None

        weather_db = "NSRDB"
        weather_arg = {
            "satellite": satellite,
            "names": year,
            "NREL_HPC": True,
            "attributes": nsrdb_attributes,
        }

        geo_weather, geo_meta = pvdeg.weather.get(
            weather_db, geospatial=True, **weather_arg
        )

        if gids:
            geo_meta = geo_meta.loc[gids]

        if bbox_kwarg:
            bbox_gids = pvdeg.geospatial.apply_bounding_box(geo_meta, **bbox_kwarg)
            geo_meta = geo_meta.loc[bbox_gids]

        #                Downselect by Region
        # ======================================================

        # string to list whole word list or keep list
        def to_list(s):
            return s if isinstance(s, list) else [s]

        if country:
            countries = to_list(country)
            self._check_set(countries, set(geo_meta["country"]))
            geo_meta = geo_meta[geo_meta["country"].isin(countries)]

        if state:
            states = to_list(state)
            states = [
                pvdeg.utilities._get_state(entry) if len(entry) == 2 else entry
                for entry in states
            ]

            self._check_set(states, set(geo_meta["state"]))
            geo_meta = geo_meta[geo_meta["state"].isin(states)]

        if county:
            if isinstance(county, str):
                county = to_list(county)

            self._check_set(county, set(geo_meta["county"]))
            geo_meta = geo_meta[geo_meta["county"].isin(county)]
        # ======================================================

        geo_meta, geo_gids = pvdeg.utilities.gid_downsampling(
            geo_meta, downsample_factor
        )

        geo_weather = pvdeg.weather.map_weather(geo_weather)

        self.weather_data = geo_weather
        self.meta_data = geo_meta
        self.gids = geo_gids

        if see_added:
            message = f"Gids Added - {self.gids}"
            warnings.warn(message, UserWarning)

        return

    def downselect_CONUS(
        self,
    ) -> None:
        """Downselect US to contiguous US geospatial data"""

        geo_weather, geo_meta = self.weather_data, self.meta_data

        geo_meta = geo_meta[~geo_meta["state"].isin(["Alaska", "Hawaii"])]
        geo_weather = geo_weather.sel(gid=geo_meta.index)

        self.weather_data = geo_weather
        self.meta_data = geo_meta
        self.gids = geo_meta.index.values

    def location_bounding_box(
        self,
        coord_1: Optional[tuple[float]] = None,
        coord_2: Optional[tuple[float]] = None,
        coords: Optional[np.ndarray[float]] = None,
    ) -> None:
        """
        Apply latitude-longitude rectangular bounding box.

        Applies latitude-longitude rectangular bounding box to geospatial scenario
        metadata.

        Parameters:
        -----------
        coord_1 : list, tuple
            Top left corner of bounding box as lat-long coordinate pair as list or
            tuple.
        coord_2 : list, tuple
            Bottom right corner of bounding box as lat-long coordinate pair in list
            or tuple.
        coords : np.array
            2d tall numpy array of [lat, long] pairs. Bounding box around the most
            extreme entries of the array. Alternative to providing top left and
            bottom right box corners. Could be used to select amongst a subset of
            data points. ex) Given all points for the planet, downselect based on
            the most extreme coordinates for the United States coastline information.

        Returns:
        --------
        None
        """
        bbox_gids = pvdeg.geospatial.apply_bounding_box(
            self.meta_data, coord_1, coord_2, coords
        )

        self.meta_data = self.meta_data.loc[bbox_gids]

    def set_kdtree(self, kdtree=None) -> None:
        """Initialize a kidtree and save it to the GeospatialScenario."""
        if kdtree is None:
            self.kdtree = pvdeg.geospatial.meta_KDtree(meta_df=self.meta_data)
        else:
            self.kdtree = kdtree

    def classify_mountains_radii(
        self,
        rad_1: Union[float, int] = 12,
        rad_2: Union[float, int] = 1,
        threshold_factor: Union[float, int] = 1.25,
        elevation_floor: Union[float, int] = 0,
        bbox_kwarg: Optional[dict] = {},
        kdtree=None,
    ):
        """
        Find mountains from elevation metadata using sklearn kdtree for fast lookup.
        Compares a large area of points to a small area of points to find
        significant changes in elevation representing mountains. Tweak the radii
        to determine the sensitivity and noise. Bad radii cause the result to
        become unstable quickly. kdtree can be generated using
        ``pvdeg.geospatial.meta_KDTree``

        Parameters:
        -----------
        meta_df : pd.DataFrame
            Dataframe of metadata as generated by pvdeg.weather.get for geospatial
        rad_1 : float
            radius of the larger search area whose elevations are compared against
            the smaller search area. controls the kdtree query region.
        rad_2 : float
            radius of the smaller search area whose elevations are compared to the
            larger area. controls the kdtree query region.
        threshold_factor : float
            change the significance level of elevation difference between
            small and large regions. Higher means terrain must be more extreme to
            register as a mountain. Small changes result in large differences here.
            When the left side of the expression is greater, the datapoint is
            classified as a mountain.
            ``local mean elevation > broad mean elevation * threshold_factor``
        elevation_floor : int
            minimum inclusive elevation in meters. If a point has smaller location
            it will be clipped from result.
        kdtree : sklearn.neighbors.KDTree
            Generated automatically but can be provided externally.
            kdtree containing latitude-longitude pairs for quick lookups
            Generate using ``pvdeg.geospatial.meta_KDTree``

        Returns:
        --------
        None, strictly updates meta_data attribute of GeospatialScenario instance.
        """
        self.set_kdtree(kdtree=kdtree)

        gids = pvdeg.geospatial.identify_mountains_radii(
            meta_df=self.meta_data,
            kdtree=self.kdtree,
            rad_1=rad_1,
            rad_2=rad_2,
            threshold_factor=threshold_factor,
            elevation_floor=elevation_floor,
            bbox_kwarg=bbox_kwarg,
        )

        self.meta_data["mountain"] = (self.meta_data.index).isin(gids)
        return

    def classify_mountains_weights(
        self,
        threshold: int = 0,
        percentile: int = 75,
        k_neighbors: int = 3,
        method: str = "mean",
        normalization: str = "linear",
        kdtree=None,
    ):
        """
        Detect whether entry is near a mountain.

        Add a column to the scenario meta_data dataframe containing a boolean
        True or False value representing if the entry is a near a mountain.
        Calculated from weights assigned during stochastic downselection.

        Parameters:
        -----------
        threshold : float
            minimum weight that a mountain can be identifed.
            value between `[0,1]` (inclusive)
        percentile : float, int, (default = 75)
            mountain classification sensitivity. Calculates percentile of values
            remaining after thresholding, weights above this percentile are
            classified as mountains. value between `[0, 100]` (inclusive)
        k_neighbors : int, (default = 3)
            number of neighbors to check for elevation data in nearest neighbors
        method : str, (default = 'mean')
            method to calculate elevation weights for each point.
            Options : `'mean'`, `'sum'`, `'median'`
        normalization : str, (default = 'linear')
            function to apply when normalizing weights. Logarithmic uses log_e/ln
            options : `'linear'`, `'logarithmic'`, '`exponential'`
        kdtree : sklearn.neighbors.KDTree or str
            Generated automatically but can be provided externally.
            kdtree containing latitude-longitude pairs for quick lookups
            Generate using ``pvdeg.geospatial.meta_KDTree``. Can take a pickled
            kdtree as a path to the .pkl file.

        Returns:
        --------
        None, strictly updates meta_data attribute of scenario.

        See Also:
        ---------
        `pvdeg.geospatial.identify_mountains_weights`
        """
        self.set_kdtree(kdtree=kdtree)

        gids = pvdeg.geospatial.identify_mountains_weights(
            meta_df=self.meta_data,
            kdtree=self.kdtree,
            threshold=threshold,
            percentile=percentile,
            k_neighbors=k_neighbors,
            method=method,
            normalization=normalization,
        )

        self.meta_data["mountain"] = (self.meta_data.index).isin(gids)
        return

    def classify_feature(
        self,
        feature_name=None,
        resolution="10m",
        radius=None,
        kdtree=None,
        bbox_kwarg={},
    ):
        """
        Update metadata.

        feature_name : str
            cartopy.feature.NaturalEarthFeature feature key.
            Options: ``'lakes'``, ``'rivers_lake_centerlines'``, ``'coastline'``
        resolution : str
            cartopy.feature.NaturalEarthFeature resolution.
            Options: ``'10m'``, ``'50m'``, ``'110m'``
        radius : float
            Area around feature coordinates to include in the downsampled result.
            Bigger area means larger radius and more samples included.
            pass
        kdtree : sklearn.neighbors.KDTree or str
            Generated automatically but can be provided externally.
            kdtree containing latitude-longitude pairs for quick lookups
            Generate using ``pvdeg.geospatial.meta_KDTree``. Can take a pickled
            kdtree as a path to the .pkl file.

        Returns:
        --------
        None, strictly updates meta_data attribute of scenario.

        See Also:
        ---------
        `pvdeg.geospatial.feature_downselect`
        """
        self.set_kdtree(kdtree=kdtree)

        feature_gids = pvdeg.geospatial.feature_downselect(
            meta_df=self.meta_data,
            kdtree=self.kdtree,
            feature_name=feature_name,
            resolution=resolution,
            radius=radius,
            bbox_kwarg=bbox_kwarg,
        )

        self.meta_data[feature_name] = (self.meta_data.index).isin(feature_gids)
        return

    def downselect_elevation_stochastic(
        self,
        downselect_prop,
        k_neighbors=3,
        method="mean",
        normalization="linear",
        kdtree=None,
    ):
        """
        Prefenetially downselect data points based on elevation and update
        scenario metadata.

        Parameters
        -----------
        downselect_prop : float
            proportion of original datapoints to keep in output gids list
        k_neighbors : int, (default = 3)
            number of neighbors to check for elevation data in nearest neighbors
        method : str, (default = 'mean')
            method to calculate elevation weights for each point.
            Options : `'mean'`, `'sum'`, `'median'`
        normalization : str, (default = 'linear')
            function to apply when normalizing weights. Logarithmic uses $log_e$, $ln$
            options : `'linear'`, `'log'`, '`exp'`, `'invert-linear'`
        kdtree : sklearn.neighbors.KDTree or str
            Generated automatically but can be provided externally.
            kdtree containing latitude-longitude pairs for quick lookups
            Generate using ``pvdeg.geospatial.meta_KDTree``. Can take a pickled
            kdtree as a path to the .pkl file.

        Returns
        --------
        None

        Notes
        --------
        This method takes a random choice of points using a weighting for bias.
        This weighting is deterministic but the choice is random. To guarantee the same
        output each time, seed the numpy random number generator with a constant value.

        ``np.random.seed(value)``

        See Also
        ---------
        `pvdeg.geospatial.elevation_stochastic_downselect` for more info/docs
        """
        self.set_kdtree(kdtree=kdtree)

        gids = pvdeg.geospatial.elevation_stochastic_downselect(
            meta_df=self.meta_data,
            kdtree=self.kdtree,
            downselect_prop=downselect_prop,
            k_neighbors=k_neighbors,
            method=method,
            normalization=normalization,
        )

        self.meta_data = self.meta_data.loc[gids]
        self.gids = gids
        return

    def gid_downsample(self, downsample_factor: int) -> None:
        """
        Downsample the NSRDB GID grid by a factor of n.

        Returns:
        --------
        None

        See Also:
        ---------
        `pvdeg.utilities.gid_downsample`
        """
        self.meta_data, sub_gids = pvdeg.utilities.gid_downsampling(
            meta=self.meta_data, n=downsample_factor
        )

        self.gids = sub_gids

    @pvdeg.decorators.deprecated("not needed, use geospatialscenario.gids")
    def gids_tonumpy(self) -> np.array:
        """
        Convert the scenario's gids to a numpy array.

        Returns:
        --------
        gids : np.array
            all nsrdb gids from the scenario's metadata
        """
        return self.meta_data.index

    @pvdeg.decorators.deprecated("not needed, use list(geospatialscenario.gids)")
    def gids_tolist(self) -> np.array:
        """
        Convert the scenario's gids to a python list.

        Returns:
        --------
        gids : np.array
            all nsrdb gids from the scenario's metadata
        """
        return list(self.meta_data.index)

    @property
    def coords(self) -> np.array:
        """
        Create a tall 2d numpy array of gids of the shape.

        ```
        [
            [lat, long],
                ...
            [lat, long]
        ]
        ```
        Returns:
        --------
        coords : np.array
            tall numpy array of lat-long pairs
        """
        coords = np.column_stack(
            (self.meta_data["latitude"], self.meta_data["longitude"])
        )

        return coords

    @property
    def geospatial_data(self) -> tuple[xr.Dataset, pd.DataFrame]:
        """
        Extract geospatial weather dataset and metadata df from the scenario object.

        Example Use:
        >>> geo_weather, geo_meta = GeospatialScenario.geospatial_data()

        This gets us the result we would use in the traditional pvdeg geospatial
        approach.

        Parameters:
        -----------
        None

        Returns:
        --------
        (weather_data, meta_data): tuple[xr.Dataset, pd.DataFrame]
            A tuple of weather data as an `xarray.Dataset` and the corresponding meta
            data as a dataframe.
        """
        # downsample here, not done already happens at pipeline runtime
        geo_weather_sub = self.weather_data.sel(gid=self.meta_data.index).chunk(
            chunks={"time": -1, "gid": 50}
        )
        return geo_weather_sub, self.meta_data

    # @dispatch(xr.Dataset, pd.DataFrame)
    def set_geospatial_data(
        self, weather_ds: xr.Dataset, meta_df: pd.DataFrame
    ) -> None:
        """
        Parameters:
        -----------
        weather_ds : xarray.Dataset
            Dataset containing weather data for a block of gids.
        meta_df : pandas.DataFrame
            DataFrame containing meta data for a block of gids.

        Modifies:
        ----------
        self.weather_data
            sets to weather_ds
        self.meta_data
            sets to meta_df
        """
        self.weather_data, self.meta_data = weather_ds, meta_df

    def addJob(
        self,
        func: Callable,
        template: xr.Dataset = None,
        func_params: dict = {},
        see_added: bool = False,
    ) -> None:
        """
        Add a pvdeg geospatial function to the scenario pipeline. If no template is
        provided, `addJob` attempts to use `geospatial.auto_template` this will raise an

        Parameters:
        -----------
        func : function
            pvdeg function to use for geospatial analysis.
        template : xarray.Dataset
            Template for output data. Only required if a function is not supported by
            `geospatial.auto_template`.
        func_params : dict
            job specific keyword argument dictionary to provide to the function
        see_added : bool
            set flag to get a userWarning notifying the user of the job added
            to the pipeline in method call. ``default = False``
        """

        if template is None:

            # take the weather datapoints specified by metadata and create a template
            # based on them.
            self.weather_data = self.weather_data.sel(gid=self.meta_data.index)
            template = pvdeg.geospatial.auto_template(
                func=func, ds_gids=self.weather_data
            )

        self.template = template
        self.func = func
        self.func_params = func_params

        if see_added:
            message = f"{func.__name__} added to scenario with arguments {func_params}\
            using template: {template}"
            warnings.warn(message, UserWarning)

    def run(self, hpc_worker_conf: Optional[dict] = None) -> None:
        """
        Run the geospatial scenario stored in the geospatial scenario object.

        Only supports one function at a time. Unlike `Scenario` which supports unlimited
        conventional pipeline jobs.
        Results are stored in the `GeospatialScenario.results` attribute.

        Creates a dask client if it has not been initialized previously with
        `GeospatialScenario.start_dask`.

        Parameters:
        -----------
        hpc_worker_conf : dict
            Dictionary containing dask hpc settings (see examples below).
            When `None`, a default configuration is used.

            Examples
            --------
            Local cluster:

            .. code-block:: python

                hpc = {'manager': 'local',
                    'n_workers': 1,
                    'threads_per_worker': 8,
                    'memory_limit': '10GB'}

            SLURM cluster:

            .. code-block:: python

                kestrel = {
                    'manager': 'slurm',
                    'n_jobs': 1,  # Max number of nodes used for parallel processing
                    'cores': 104,
                    'memory': '246GB',
                    'account': 'pvsoiling',
                    'walltime': '4:00:00',
                    'processes': 52,
                    'local_directory': '/tmp/scratch',
                    'job_extra_directives': ['-o ./logs/slurm-%j.out'],
                    'death_timeout': 600,}
        """
        if self.dask_client and hpc_worker_conf:
            raise ValueError("Dask Client already exists, cannot configure new client.")
        elif not self.dask_client:
            self.dask_client = pvdeg.geospatial.start_dask(hpc=hpc_worker_conf)

        print("Dashboard:", self.dask_client.dashboard_link)

        analysis_result = pvdeg.geospatial.analysis(
            weather_ds=self.weather_data,
            meta_df=self.meta_data,
            func=self.func,
            template=self.template,  # provided or generated via autotemplate in
            # GeospatialScenario.addJob
        )

        self.results = analysis_result

        self.dask_client.shutdown()

    def restore_result_gids(self):
        """
        Restore gids to result Dataset as datavariable from original metadata.

        Assumes results will be in the same order as input metadata rows.
        Otherwise will fail silently and restore incorrect gids
        """
        flattened = self.results.stack(points=("latitude", "longitude"))

        gids = self.meta_data.index.values

        # Create a DataArray with the gids and assign it to the Dataset
        gids_da = xr.DataArray(gids, coords=[flattened["points"]], name="gids")

        # Unstack the DataArray to match the original dimensions of the Dataset
        gids_da = gids_da.unstack("points")

        self.results = self.results.assign(gids=gids_da)

    @pvdeg.decorators.deprecated("removing complexity")
    def _get_geospatial_data(year: int):
        """
        Get geospatial weather dataset and metadata dictionary, helper function.

        Parameters
        ----------
        Year : int
            select the year of data to take from the NSRDB

        Returns
        --------
        weather_ds : xarray.Dataset
            dataset with coordinates of gid and time and weather data as datavariables
        meta_df : pd.DataFrame
            dataframe with each row representing the metadata of each gid in the dataset
        """
        weather_db = "NSRDB"

        weather_arg = {
            "satellite": "Americas",
            "names": year,
            "NREL_HPC": True,
            # 'attributes': ['air_temperature', 'wind_speed', 'dhi', 'ghi', 'dni',
            # 'relative_humidity']}
            "attributes": [],  # does having do atributes break anything, should we just
            # pick one
        }

        weather_ds, meta_df = pvdeg.weather.get(
            weather_db, geospatial=True, **weather_arg
        )

        return weather_ds, meta_df

    @pvdeg.decorators.deprecated("removing co")
    def getValidRegions(
        self,
        country: Optional[str] = None,
        state: Optional[str] = None,
        county: Optional[str] = None,
        target_region: Optional[str] = None,
    ):
        """
        Get all valid region names in the NSRDB. Only works on HPC.

        Arguments
        ---------
        country : str, optional
        state : str, optional
        country : str, optional
        target_region : str
            Select return field. Options ``country``, ``state``, ``county``.

        Returns
        -------
        valid_regions : numpy.ndarray
            list of strings representing all unique region entries in the nsrdb.
        """
        if not self.geospatial:  # add hpc check
            return AttributeError(
                f"self.geospatial should be True. Current value = {self.geospatial}"
            )

        # discard_weather, meta_df = Scenario._get_geospatial_data(year=2022)
        discard_weather, meta_df = self._get_geospatial_data(year=2022)

        if country:
            meta_df = meta_df[meta_df["country"] == country]
        if state:
            meta_df = meta_df[meta_df["state"] == state]
        if county:
            meta_df = meta_df[meta_df["county"] == county]

        return meta_df[target_region].unique()

    def plot(self):
        """Unsable in GeospatialScenario class instance, only in Scenario instance."""
        # python has no way to hide a parent class method in the child, so this only
        # exists to prevent access
        raise AttributeError(
            "The 'plot' method is not accessible in GeospatialScenario, only in \
                Scenario"
        )

    def plot_coords(
        self,
        coord_1: Optional[tuple[float]] = None,
        coord_2: Optional[tuple[float]] = None,
        coords: Optional[np.ndarray[float]] = None,
        size: Union[int, float] = 1,
    ) -> tuple[matplotlib.figure, matplotlib.axes]:
        """
        Plot lat-long coordinate pairs on blank map.

        This function provides a way to view
        geospatial datapoints before your analysis.

        Parameters:
        -----------
        coord_1 : list, tuple
            Top left corner of bounding box as lat-long coordinate pair as list or
            tuple.
        coord_2 : list, tuple
            Bottom right corner of bounding box as lat-long coordinate pair in list
            or tuple.
        coords : np.array
            2d tall numpy array of [lat, long] pairs. Bounding box around the most
            extreme entries of the array. Alternative to providing top left and
            bottom right box corners. Could be used to select amongst a subset of
            data points. ex) Given all points for the planet, downselect based on
            the most extreme coordinates for the United States coastline information.
        size : float
            matplotlib scatter point size. Without any downsampling NSRDB
            points will siginficantly overlap and plot may appear as a solid color.

        Returns:
        --------
        fig, ax
            matplotlib figure and axis
        """
        fig = plt.figure(figsize=(15, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())

        if (coord_1 and coord_2) or (coords is not None):
            pvdeg.utilities._plot_bbox_corners(
                ax=ax, coord_1=coord_1, coord_2=coord_2, coords=coords
            )

        pvdeg.utilities._add_cartopy_features(ax=ax)

        ax.scatter(
            self.meta_data["longitude"],
            self.meta_data["latitude"],
            color="black",
            s=size,
            transform=ccrs.PlateCarree(),
        )

        plt.title(f"Coordinate Pairs from '{self.name}' Meta Data")
        plt.show()

        return fig, ax

    def plot_meta_classification(
        self,
        col_name: str = None,
        coord_1: Optional[tuple[float]] = None,
        coord_2: Optional[tuple[float]] = None,
        coords: Optional[np.ndarray[float]] = None,
        size: Union[int, float] = 1,
    ) -> tuple[matplotlib.figure, matplotlib.axes]:
        """
        Plot classified lat-long coordinate pairs on map.

        Quicly view geospatial datapoints with binary classification in a meta_data
        dataframe column before your analysis.

        Parameters:
        -----------
        col_name : str
            Column containing binary classification data. Ex: `mountain` after
            running ``downselect_mountains_weights``.
        coord_1 : list, tuple
            Top left corner of bounding box as lat-long coordinate pair as list or
            tuple.
        coord_2 : list, tuple
            Bottom right corner of bounding box as lat-long coordinate pair in list
            or tuple.
        coords : np.array
            2d tall numpy array of [lat, long] pairs. Bounding box around the most
            extreme entries of the array. Alternative to providing top left and
            bottom right box corners. Could be used to select amongst a subset of
            data points. ex) Given all points for the planet, downselect based on
            the most extreme coordinates for the United States coastline information.
        size : float
            matplotlib scatter point size. Without any downsampling NSRDB
            points will siginficantly overlap.

        Returns:
        --------
        fig, ax
            matplotlib figure and axis
        """
        if not col_name:
            raise ValueError("col_name cannot be none")

        if col_name not in self.meta_data.columns:
            raise ValueError(
                f"{col_name} not in self.meta_data columns as follows \
                    {self.meta_data.columns}"
            )

        col_dtype = self.meta_data[col_name].dtype
        if col_dtype != bool:
            raise ValueError(
                f"meta_data column {col_name} expected dtype bool not {col_dtype}"
            )

        near = self.meta_data[self.meta_data[col_name] is True]
        not_near = self.meta_data[self.meta_data[col_name] is False]

        fig = plt.figure(figsize=(15, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())

        if (coord_1 and coord_2) or (coords is not None):
            pvdeg.utilities._plot_bbox_corners(
                ax=ax, coord_1=coord_1, coord_2=coord_2, coords=coords
            )
        pvdeg.utilities._add_cartopy_features(ax=ax)

        ax.scatter(
            not_near["longitude"],
            not_near["latitude"],
            color="red",
            s=size,
            transform=ccrs.PlateCarree(),
            label=f"Not Near {col_name}",
        )
        ax.scatter(
            near["longitude"],
            near["latitude"],
            color="blue",
            s=size,
            transform=ccrs.PlateCarree(),
            label=f"Near {col_name}",
        )

        plt.title(f"Geographic Points with Proximity to {col_name} Highlighted")
        plt.legend()
        plt.show()

        return fig, ax

    def plot_world(
        self,
        data_variable: str,
        cmap: str = "viridis",
    ) -> tuple[matplotlib.figure, matplotlib.axes]:
        da = (self.results)[data_variable]

        fig, ax = plt.subplots(
            figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()}
        )

        da.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap)
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.gridlines(draw_labels=True)

        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAKES, edgecolor="black")
        plt.show()

        return fig, ax

    # test this
    def plot_USA(
        self,
        data_from_result: str,
        fpath: str = None,
        cmap: str = "viridis",
        vmin: Union[int, float] = 0,
        vmax: Optional[Union[int, float]] = None,
    ) -> tuple[matplotlib.figure, matplotlib.axes]:
        """
        Plot geospatial scenario result.

        Only works on geospatial scenarios.

        Parameters
        ----------
        data_from_result : str
            select the datavariable to plot from the result xarray
        fpath : str
            path to save plot output on, saves to current directory if ``None``
        cmap : str
            colormap to use in plot
        vmin : int
            lower bound on values in linear color map
        vmax : int
            upper bound on values in linear color map
        """
        if not self.geospatial:
            return False

        fig, ax = pvdeg.geospatial.plot_USA(
            self.results[data_from_result],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title="add_dynamic_title",
            cb_title=f"dynamic title : {data_from_result}",
        )

        fpath if fpath else [f"os.getcwd/{self.name}-{self.results[data_from_result]}"]
        fig.savefig()

        return fig, ax

    def _check_set(self, iterable, to_check: set):
        """Check if iterable is a subset of to_check."""
        if not isinstance(iterable, set):
            iterable = set(iterable)

        if not iterable.issubset(to_check):
            raise ValueError(f"All of iterable: {iterable} is not in {to_check}")

    def format_geospatial_work(self):
        if self.func:
            return f"""
                <p><strong>self.func:</strong> {self.func.__name__}</p>  # noqa
                <p><strong>self.template:</strong> {self.format_template()}</p> # noqa
            """

        return ""

    def format_dask_link(self):
        if self.dask_client:
            return f"""
                <a href="{self.dask_client.dashboard_link}" target="_blank">{self.dask_client.dashboard_link}</a></p>  # noqa
            """
        return ""

    def _ipython_display_(self):
        file_url = f"file:///{os.path.abspath(self.path).replace(os.sep, '/')}"  # noqa
        html_content = f"""
        <div style="border:1px solid #ddd; border-radius: 5px; padding: 3px; margin-top: 5px;">  # noqa
            <h2>self.name: {self.name}</h2>
            <p><strong>self.path:</strong> <a href="{file_url}" target="_blank">{self.path}</a></p>  # noqa
            <p><strong>self.hpc:</strong> {self.hpc}</p>
            <p><strong>self.gids:</strong> {self.gids}</p>
            <div>
                <h3>self.results</h3>
                {self.format_results() if self.results else ''}
            </div>
            <div>
                <h3>Geospatial Work</h3>
                {self.format_geospatial_work()}
            </div>
            <div>
                <h3>self.modules</h3>
                {super().format_modules()}
            </div>
            <div>
                <h3>self.weather_data</h3>
                {self.format_geo_weather()}
            </div>
            <div>
                <h3>self.meta_data</h3>
                {self.format_geo_meta()}
            </div>
            <div>
                <h3>self.kdtree</h3>
                {self.kdtree or ''}
            </div>
            <div>
                <h3>self.dask_client</h3>
                {self.format_dask_link()}
            </div>
        </div>
        <p><i>All attributes can be accessed by the names shown above.</i></p>
        <script>
            function toggleVisibility(id) {{
                var content = document.getElementById(id);
                var arrow = document.getElementById('arrow_' + id);
                if (content.style.display === 'none') {{
                    content.style.display = 'block';
                    arrow.innerHTML = '▼';
                }} else {{
                    content.style.display = 'none';
                    arrow.innerHTML = '►';
                }}
            }}
        </script>
        """
        display(HTML(html_content))

    def format_results(self):
        results_html = "<div>"
        if "geospatial_job" in self.results:
            result = self.results["geospatial_job"]
            result_id = "geospatial_result"
            formatted_output = self.format_output(result)
            result_content = f"""
            <div id="{result_id}" onclick="toggleVisibility('content_{result_id}')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">  # noqa
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_content_{result_id}" style="color: #b676c2;">►</span>  # noqa
                    Geospatial Result
                </h4>
            </div>
            <div id="content_{result_id}" style="display:none; margin-left: 20px; padding: 5px; background-color: #f0f0f0; color: #000;">  # noqa
                {formatted_output}
            </div>
            """
            results_html += result_content
        results_html += "</div>"
        return results_html

    def format_geo_meta(self):
        meta_data_html = ""

        if self.meta_data is not None:

            meta_data_html = f"""
            <div id="meta_data" onclick="toggleVisibility('content_meta_data')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">  # noqa
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_content_meta_data" style="color: #b676c2;">►</span>
                    Meta Data
                </h4>
            </div>
            <div id="content_meta_data" style="display:none; margin-left: 20px; padding: 5px;">  # noqa
                {self.meta_data._repr_html_()}
            </div>
            """

        return meta_data_html

    def format_template(self):
        template_html = ""

        if self.meta_data is not None:

            template_html = f"""
            <div id="template" onclick="toggleVisibility('content_template')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">  # noqa
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_content_template" style="color: #b676c2;">►</span>
                    Template
                </h4>
            </div>
            <div id="content_template" style="display:none; margin-left: 20px; padding: 5px;">  # noqa
                {self.template._repr_html_()}
            </div>
            """

        return template_html

    def format_geo_weather(self):
        weather_data_html = ""

        if self.weather_data is not None:

            weather_data_html = f"""
            <div id="weather_data" onclick="toggleVisibility('content_weather_data')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">  # noqa
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_content_weather_data" style="color: #b676c2;">►</span>  # noqa
                    Weather Data
                </h4>
            </div>
            <div>
            <div id="content_weather_data" style="display:none; margin-left: 20px; padding: 5px>  # noqa
                {self.weather_data._repr_html_()}
            </div>
            """

        return weather_data_html
