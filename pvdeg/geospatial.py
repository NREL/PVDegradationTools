"""Collection of classes and functions for geospatial analysis."""

from pvdeg import utilities

import xarray as xr
import dask.array as da
import pandas as pd
import numpy as np
from dask.distributed import Client, LocalCluster
from scipy.interpolate import griddata
from copy import deepcopy

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

from collections.abc import Callable
import cartopy.feature as cfeature

from typing import Tuple
from shapely import LineString, MultiLineString


def start_dask(hpc=None):
    """Start a dask cluster for parallel processing.

    Parameters
    ----------
    hpc : dict
        Dictionary containing dask hpc settings (see examples below).

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

    Returns
    -------
    client : dask.distributed.Client
        Dask client object.
    """
    if hpc is None:
        cluster = LocalCluster()
    else:
        manager = hpc.pop("manager")

        if manager == "local":
            cluster = LocalCluster(**hpc)
        elif manager == "slurm":
            from dask_jobqueue import SLURMCluster

            n_jobs = hpc.pop("n_jobs")
            cluster = SLURMCluster(**hpc)
            cluster.scale(jobs=n_jobs)

    client = Client(cluster)
    print("Dashboard:", client.dashboard_link)
    # client.wait_for_workers(n_workers=1)

    return client


# rename this?
# and combine into a single function with _df_from_arbitrary, this ds_from_arbitray isnt
# really doing anything anymore
# we only want ds_from_arbitrary and then convert to ds, but if the input is a dataset
# already then we dont want to anything
# def _ds_from_arbitrary(res, func):
#     """
#     Convert an arbitrary return type to xarray.Dataset.
#     """

#     ######## STRUCTURAL #########
#     # functions can just return xr.Dataset to take advantage of geospatial
#     # this should not be required to implement a new geospatial function

#     if isinstance(res, xr.Dataset):
#         return res


#     # if isinstance(res, pysam.inspirePysamReturn):
#     #     return pysam._handle_pysam_return(res)
#     # add more conditionals if we have special cases
#     # or add general case for mixed return dimensions: HARD

#     # handles collections with elements of same shapes
#     df = _df_from_arbitrary(res=res, func=func)
#     ds = xr.Dataset.from_dataframe(df)

#     if not df.index.name:
#         ds = ds.isel(index=0, drop=True)

#     return ds


def _df_from_arbitrary(res, func):
    """Convert an arbitrary return type to dataframe.

    Results must be of similar shape currently. Either all numerics or all timeseries.
    """
    numerics = (int, float, np.number)
    arrays = (np.ndarray, pd.Series)

    if isinstance(res, pd.DataFrame):
        return res
    elif isinstance(res, pd.Series):
        return pd.DataFrame(res, columns=[func.__name__])
    elif isinstance(res, (int, float)):
        return pd.DataFrame([res], columns=[func.__name__])
    elif isinstance(res, tuple) and all(isinstance(item, numerics) for item in res):
        return pd.DataFrame([res])
    elif isinstance(res, tuple) and all(isinstance(item, arrays) for item in res):
        # add check for same size, raise value error otherwise
        return pd.concat(
            res, axis=1
        )  # they must all be the same length here or this will error out
    else:
        raise NotImplementedError(
            f"function return type: {type(res)} not available for geospatial "
            "analysis yet. This could be result of mismatched coordinates of "
            "outputs. EX. tuple(dataframe, int)."
        )


def calc_gid(ds_gid, meta_gid, func, **kwargs):
    """Calculate a single grid for a given function.

    Parameters
    ----------
    ds_gid : xarray.Dataset
        Dataset containing weather data for a single gid.
    meta_gid : dict
        Dictionary containing meta data for a single gid.
    func : function
        Function to apply to weather data.
    kwargs : dict
        Keyword arguments to pass to func.

    Returns
    -------
    ds_res : xarray.Dataset
        Dataset with results for a single gid.
    """
    # meta gid was appearing with ('lat' : {gid, lat}, 'long' : {gid : long}), not
    # a permanent fix
    # hopefully this isn't too slow, what is causing this? how meta is being read
    # from dataset? @ martin?
    if type(meta_gid["latitude"]) is dict:
        meta_gid = utilities.fix_metadata(meta_gid)

    # set time index here? is there any reason the weather shouldn't always have only
    # pd.datetime index? @ martin?
    df_weather = ds_gid.to_dataframe()
    if isinstance(
        df_weather.index, pd.MultiIndex
    ):  # check for multiindex and convert to just time index, don't know what was
        # causing this
        df_weather = df_weather.reset_index().set_index("time")

    res = func(weather_df=df_weather, meta=meta_gid, **kwargs)
    # res is the type returned by func
    # can be float, tuple, list, dataframe, dataset, etc.
    # need to convert it to a dataset

    if isinstance(res, xr.Dataset):
        return res

    # handles collections with elements of same shapes
    df = _df_from_arbitrary(res=res, func=func)
    ds_res = xr.Dataset.from_dataframe(df)

    if not df.index.name:
        ds_res = ds_res.isel(index=0, drop=True)

    return ds_res


def calc_block(weather_ds_block, future_meta_df, func, func_kwargs):
    """Calculate a block of gids for a given function.

    Parameters
    ----------
    weather_ds_block : xarray.Dataset
        Dataset containing weather data for a block of gids.
    future_meta_df : pandas.DataFrame
        DataFrame containing meta data for a block of gids.
    func : function
        Function to apply to weather data.
    func_kwargs : dict
        Keyword arguments to pass to func.

    Returns
    -------
    ds_res : xarray.Dataset
        Dataset with results for a block of gids.
    """
    res = weather_ds_block.groupby("gid", squeeze=False).map(
        lambda ds_gid: calc_gid(
            ds_gid=ds_gid.squeeze(),
            meta_gid=future_meta_df.loc[ds_gid["gid"].values[0]].to_dict(),
            func=func,
            **func_kwargs,
        )
    )
    return res


def analysis(weather_ds, meta_df, func, template=None, **func_kwargs):
    """Apply a function to each gid of a weather dataset.

    `analysis` will attempt to
    create a template using `geospatial.auto_template`. If this process fails you will
    have to provide a geospatial template to the template argument.

    ValueError: <function-name> cannot be autotemplated. create a template manually
    with `geospatial.output_template`

    Parameters
    ----------
    weather_ds : xarray.Dataset
        Dataset containing weather data for a block of gids.
    meta_df : pandas.DataFrame
        DataFrame containing meta data for a block of gids.
    func : function
        Function to apply to weather data.
    template : xarray.Dataset
        Template for output data.
    func_kwargs : dict
        Keyword arguments to pass to func.

    Returns
    -------
    ds_res : xarray.Dataset
        Dataset with results for a block of gids.
    """
    if template is None:
        template = auto_template(func=func, ds_gids=weather_ds)

    # future_meta_df = client.scatter(meta_df)
    kwargs = {"func": func, "future_meta_df": meta_df, "func_kwargs": func_kwargs}

    stacked = weather_ds.map_blocks(
        calc_block, kwargs=kwargs, template=template
    ).compute()

    # lats = stacked.latitude.values.flatten()
    # lons = stacked.longitude.values.flatten()
    stacked = stacked.drop(["gid"])
    # stacked = stacked.drop_vars(['latitude', 'longitude'])
    # stacked.coords["gid"] = pd.MultiIndex.from_arrays(
    #     [meta_df["latitude"], meta_df["longitude"]], names=["latitude", "longitude"]
    # )
    mindex_obj = pd.MultiIndex.from_arrays(
        [meta_df["latitude"], meta_df["longitude"]], names=["latitude", "longitude"]
    )
    mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex_obj, "gid")
    stacked = stacked.assign_coords(mindex_coords)

    stacked = stacked.drop_duplicates("gid")
    res = stacked.unstack("gid")  # , sparse=True
    return res


def output_template(
    ds_gids: xr.Dataset,
    shapes: dict,
    attrs=dict(),
    global_attrs=dict(),
    add_dims=dict(),
):
    """Generate xarray template for output data.

    Output variables and associated
    dimensions need to be specified via the shapes dictionary. The dimension length are
    derived from the input data. Additonal output dimensions can be defined with the
    add_dims argument.

    Examples
    --------
    Providing the shapes dictionary can be confusing. Here is what the `shapes`
    dictionary should look like for `pvdeg.standards.standoff`. Refer to the docstring,
    the function will have one result per location so the only dimension for each return
    value is "gid", a geospatial ID number.

    .. code-block:: python
        shapes = {
            "x": ("gid",),
            "T98_inf": ("gid",),
            "T98_0": ("gid",),
        }

    **Note: The dimensions are stored in a tuple, this this why all of the parenthesis
    have commas after the single string, otherwise python will interpret the value as a
    string.**

    This is what the shapes dictinoary should look like for `pvdeg.humidity.module`.
    Refering to the docstring, we can see that the function will return a timeseries
    result for each location. This means we need dimensions of "gid" and "time".

    .. code-block:: python
        shapes = {
            "RH_surface_outside": ("gid", "time"),
            "RH_front_encap": ("gid", "time"),
            "RH_back_encap": ("gid", "time"),
            "RH_backsheet": ("gid", "time"),
        }

    Parameters
    ----------
    ds_gids : xarray.Dataset
        Dataset containing the gids and their associated dimensions.
        Dataset should already be chunked.
    shapes : dict
        Dictionary of variable names and their associated dimensions.
    attr : dict
        Dictionary of attributes for each variable (e.g. units).
    add_dims : dict
        Dictionary of dimensions to add to the output template.

    Returns
    -------
    output_template : xarray.Dataset
        Template for output data.
    """
    dims = set([d for dim in shapes.values() for d in dim])
    dims_size = dict(ds_gids.sizes) | add_dims

    # update the coordinates with the dims which include add_dims
    coords = {}
    for dim in dims:
        if dim in ds_gids:
            coords[dim] = ds_gids[dim]
        elif dim in add_dims:
            coords[dim] = np.arange(dims_size[dim])  # placeholder array (edge case)
        else:
            raise ValueError(f"dim: {dim} not in ds_gids or add_dims")

    output_template = xr.Dataset(
        data_vars={
            var: (
                dim,
                da.empty([dims_size[d] for d in dim]),
                attrs.get(var),
            )  # produces dask array with 1 chunk of the same size as the input
            for var, dim in shapes.items()
        },
        coords=coords,
        attrs=global_attrs,
    )

    # we can only chunk dimensions existing in the input
    # added dimensions will fail if chunked under this scheme
    # because they do not exist in ds gids
    if ds_gids.chunks:
        output_template = output_template.chunk(
            {dim: ds_gids.chunks[dim] for dim in dims if dim in ds_gids}
        )

    return output_template


def zero_template(
    lat_grid, lon_grid, shapes, attrs=dict(), global_attrs=dict(), add_dims=dict()
):
    """Create zero-filled xarray.Dataset based on provided grids and shapes."""
    gids = len(lat_grid)

    dims_size = {"gid": gids} | add_dims

    stacked = xr.Dataset(
        data_vars={
            var: (dim, da.zeros([dims_size[d] for d in dim]), attrs.get(var))
            for var, dim in shapes.items()
        },
        coords={"gid": np.linspace(0, gids - 1, gids, dtype=int)},
        attrs=global_attrs,
    )  # .chunk({dim: ds_gids.chunks[dim] for dim in dims})

    stacked = stacked.drop(["gid"])
    mindex_obj = pd.MultiIndex.from_arrays(
        [lat_grid, lon_grid], names=["latitude", "longitude"]
    )
    mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex_obj, "gid")
    stacked = stacked.assign_coords(mindex_coords)
    stacked = stacked.drop_duplicates("gid")
    res = stacked.unstack("gid")

    return res


def can_auto_template(func) -> None:
    """Check if we can use `geospatial.auto_template on a given function.

    Raise an error if the function was not declared with the `@geospatial_quick_shape`
    decorator. No error raised if we can run `geospatial.auto_template` on
    provided function, `func`.

    Parameters
    ----------
    func: callable
        function to create template from.

    Returns
    -------
    None
    """
    if not (hasattr(func, "numeric_or_timeseries") and hasattr(func, "shape_names")):
        raise ValueError(
            f"{func.__name__} cannot be autotemplated. create a template manually"
        )


def auto_template(func: Callable, ds_gids: xr.Dataset) -> xr.Dataset:
    """
    Automatically create a template for a target function: `func`.

    Only works on functions that have the `numeric_or_timeseries` and `shape_names`
    attributes. These attributes are assigned at function definition with the
    `@geospatial_quick_shape` decorator.

    Otherwise you will have to create your own template using
    `geospatial.output_template`. See the Geospatial Templates Notebook for more
    information.

    Examples
    --------
    The function returns a numeric value
    >>> pvdeg.design.edge_seal_width

    the function returns a timeseries result
    >>> pvdeg.module.humidity

    Counter example
    ---------------
    the function could either return a single numeric or a series based on changed in
    the input. Because it does not have a known result shape we cannot determine the
    attributes required for autotemplating ahead of time.

    Parameters
    ----------
    func: callable
        function to create template from. This will raise an error if the function was
        not declared with the `@geospatial_quick_shape` decorator.
    ds_gids : xarray.Dataset
        Dataset containing the gids and their associated dimensions. (geospatial weather
                                                                      dataset)
        Dataset should already be chunked.

    Returns
    -------
    output_template : xarray.Dataset
        Template for output data.
    """
    can_auto_template(func=func)

    if func.numeric_or_timeseries == "numeric":
        shapes = {datavar: ("gid",) for datavar in func.shape_names}
    elif func.numeric_or_timeseries == "timeseries":
        shapes = {datavar: ("gid", "time") for datavar in func.shape_names}
    else:
        raise ValueError(
            f"{func.__name__} 'numeric_or_timseries' attribute invalid. is\
            {func.numeric_or_timeseries} should be 'numeric' or 'timeseries'"
        )

    template = output_template(ds_gids=ds_gids, shapes=shapes)  # zeros_template?

    return template


def plot_USA(
    xr_res, cmap="viridis", vmin=None, vmax=None, title=None, cb_title=None, fp=None
):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(), frameon=False)
    ax.patch.set_visible(False)
    ax.set_extent([-120, -74, 22, 50], ccrs.Geodetic())

    shapename = "admin_1_states_provinces_lakes"
    states_shp = shpreader.natural_earth(
        resolution="110m", category="cultural", name=shapename
    )
    ax.add_geometries(
        shpreader.Reader(states_shp).geometries(),
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="gray",
    )

    cm = xr_res.plot(
        transform=ccrs.PlateCarree(),
        zorder=1,
        add_colorbar=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        subplot_kws={
            "projection": ccrs.LambertConformal(
                central_longitude=-95, central_latitude=45
            )
        },
    )

    cb = plt.colorbar(cm, shrink=0.5)
    cb.set_label(cb_title)
    ax.set_title(title)

    if fp is not None:
        plt.savefig(fp, dpi=1200)

    return fig, ax


def plot_Europe(
    xr_res, cmap="viridis", vmin=None, vmax=None, title=None, cb_title=None, fp=None
):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree(), frameon=True)
    ax.patch.set_visible(True)
    ax.set_extent([-12, 31.6, 35, 71.2], ccrs.PlateCarree())

    shapename = "admin_0_countries"
    states_shp = shpreader.natural_earth(
        resolution="110m", category="cultural", name=shapename
    )
    ax.add_geometries(
        shpreader.Reader(states_shp).geometries(),
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="gray",
    )

    cm = xr_res.plot(
        transform=ccrs.PlateCarree(),
        zorder=1,
        add_colorbar=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="gouraud",
        infer_intervals=False,
    )

    cb = plt.colorbar(cm, shrink=0.5)
    cb.set_label(cb_title)
    ax.set_title(title)

    ax.set_xticks([-10, 0, 10, 20, 30], crs=ccrs.PlateCarree())
    ax.set_yticks([30, 40, 50, 60, 70], crs=ccrs.PlateCarree())

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if fp is not None:
        plt.savefig(fp, dpi=1200)

    return fig, ax


def meta_KDtree(meta_df, leaf_size=40, fp=None):
    """Create an sklearn.neighbors.KDTree for fast geospatial lookup operations.

    Requires
    Scikit Learn library. Not included in pvdeg depency list.

    Parameters
    -----------
    meta_df: pd.DataFrame
        Dataframe of metadata as generated by pvdeg.weather.get for geospatial
    leaf_size:
        Number of points at which to switch to brute-force. See sci kit docs.
    fp: str, optional
        Location to save pickled kdtree so we don't have to rebuild the tree.
        If none, no file saved. must be ``.pkl`` file extension. Open saved
        pkl file with joblib (sklearn dependency).

    Returns
    -------
    kdtree: sklearn.neighbors.KDTree
        kdtree containing latitude-longitude pairs for quick lookups

    See Also:
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
    """
    from sklearn.neighbors import KDTree
    from joblib import dump

    coordinates = meta_df[["latitude", "longitude"]].values
    # elevations = meta_df["altitude"].values

    tree = KDTree(coordinates, leaf_size)

    if fp:
        dump(tree, fp)

    return tree


def _mountains(meta_df, kdtree, index, rad_1, rad_2, threshold_factor, elevation_floor):
    coordinates = meta_df[["latitude", "longitude"]].values
    elevations = meta_df["altitude"].values

    # Reshape the coordinate array to 2D
    query_point = coordinates[index].reshape(1, -1)

    # Query the KDTree for neighbors within the specified radii
    area_points = kdtree.query_radius(query_point, r=rad_1)[0]
    local_points = kdtree.query_radius(query_point, r=rad_2)[0]

    # If no area points are found, return False
    if len(area_points) == 0:
        return False

    # Calculate mean elevations for the area and local points
    area_elevations = elevations[area_points]
    local_elevations = elevations[local_points]
    area_mean = np.mean(area_elevations)
    local_mean = np.mean(local_elevations)

    # Determine if the point is a mountain based on the threshold factor
    if (
        local_mean > area_mean * threshold_factor
        and elevations[index] >= elevation_floor
    ):
        return True

    return False


# fix coastline detection, query points instead of radius?
def identify_mountains_radii(
    meta_df,
    kdtree,
    rad_1=12,
    rad_2=1,
    threshold_factor=1.25,
    elevation_floor=0,
    bbox_kwarg={},
) -> np.array:
    """Find mountains from elevation metadata using sklearn kdtree for fast lookup.

    Compares a large area of points to a small area of points to find significant
    changes in elevation representing mountains. Tweak the radii to determine the
    sensitivity and noise. Bad radii cause the result to become unstable quickly. kdtree
    can be generated using ``pvdeg.geospatial.meta_KDTree``

    Parameters:
    -----------
    meta_df : pd.DataFrame
        Dataframe of metadata as generated by pvdeg.weather.get for geospatial
    kdtree : sklearn.neighbors.KDTree
        kdtree containing latitude-longitude pairs for quick lookups
        Generate using ``pvdeg.geospatial.meta_KDTree``
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
    Returns:
    --------
    gids : np.array
        numpy array of gids in the mountains.
    """
    meta_df.loc[:, "mountain"] = [
        _mountains(meta_df, kdtree, i, rad_1, rad_2, threshold_factor, elevation_floor)
        for i in range(len(meta_df))
    ]

    if bbox_kwarg:
        gids_in_bbox = apply_bounding_box(meta_df=meta_df, **bbox_kwarg)
        outside_bbox = meta_df.index.difference(gids_in_bbox)
        meta_df.loc[outside_bbox, "mountain"] = False

    # new...
    gids = meta_df.index[meta_df["mountain"]].to_numpy()
    return gids

    # return meta_df


def identify_mountains_weights(
    meta_df,
    kdtree,
    threshold=0,
    percentile=75,
    k_neighbors=3,
    method="mean",
    normalization="linear",
) -> np.array:
    """Find mountains using weights calculated via changes in nearest neighbors.

    elevations.

    Parameters:
    -----------
    meta_df : pd.DataFrame
        Dataframe of metadata as generated by pvdeg.weather.get for geospatial
    kdtree : sklearn.neighbors.KDTree or str
        kdtree containing latitude-longitude pairs for quick lookups
        Generate using ``pvdeg.geospatial.meta_KDTree``. Can take a pickled
        kdtree as a path to the .pkl file.
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

    Returns:
    --------
    gids : np.array
        numpy array of gids classified as mountains.
    """
    coords = np.column_stack([meta_df["latitude"], meta_df["longitude"]])
    elevations = meta_df["altitude"].values

    weights = utilities._calc_elevation_weights(
        elevations=elevations,
        coords=coords,
        k_neighbors=k_neighbors,
        method=method,
        normalization=normalization,
        kdtree=kdtree,
    )

    if threshold != 0:
        weights = np.where(weights < threshold, np.nan, weights)

    percentile_threshold = np.nanpercentile(weights, percentile)
    indicies = np.where(weights > percentile_threshold)[0]

    meta_gids = meta_df.index.values
    gids = meta_gids[indicies]

    return gids


def feature_downselect(
    meta_df,
    kdtree=None,
    feature_name=None,
    resolution="10m",
    radius=None,
    bbox_kwarg={},
) -> np.array:
    """
    Downselect function.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Dataframe of metadata as generated by pvdeg.weather.get for geospatial
    kdtree : sklearn.neighbors.KDTree or str
        kdtree containing latitude-longitude pairs for quick lookups
        Generate using ``pvdeg.geospatial.meta_KDTree``. Can take a pickled
        kdtree as a path to the .pkl file.
    feature_name : str
        cartopy.feature.NaturalEarthFeature feature key.
        Options: ``'lakes'``, ``'rivers_lake_centerlines'``, ``'coastline'``
    resolution : str
        cartopy.feature.NaturalEarthFeature resolution.
        Options: ``'10m'``, ``'50m'``, ``'110m'``
    radius : float
        Area around feature coordinates to include in the downsampled result.
        Bigger area means larger radius and more samples included.

    Returns:
    --------
    gids : np.ndarray
    """
    if isinstance(kdtree, str):
        from joblib import load

        kdtree = load(kdtree)

    if radius is None:
        if feature_name == "coastline":
            radius = 1
        elif feature_name in ["river_lake_centerlines", "lakes"]:
            radius = 0.1

    feature = cfeature.NaturalEarthFeature("physical", feature_name, resolution)
    feature_geometries = []

    # Collect geometries
    for geom in feature.geometries():
        if isinstance(geom, LineString):
            feature_geometries.append(geom)
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                feature_geometries.append(line)

    # Extract points from geometries
    feature_points = []
    for geom in feature_geometries:
        coords = list(geom.coords)
        for coord in coords:
            feature_points.append(coord)

    feature_coords = np.array(feature_points, dtype=np.float32)

    meta_df.loc[:, feature_name] = False  # this raises an error but works as expected

    include_set = set()
    for coord in feature_coords:
        coord = np.array(coord).reshape(1, -1)
        flipped_coord = coord[
            :, [1, 0]
        ]  # biggest headache of my life to figure out that these were flipped
        indices = kdtree.query_radius(flipped_coord, radius)[0]
        include_set.update(indices.tolist())

    include_arr = np.fromiter(include_set, dtype=int, count=len(include_set))

    if bbox_kwarg:
        gids_in_bbox = apply_bounding_box(meta_df=meta_df, **bbox_kwarg)
        include_arr = np.union1d(include_arr, gids_in_bbox)

    include_gids = meta_df.index[include_arr]

    return include_gids


def apply_bounding_box(
    meta_df: pd.DataFrame, coord_1=None, coord_2=None, coords=None
) -> np.array:
    """Apply latitude-longitude rectangular bounding box to geospatial metadata.

    Parameters:
    -----------
    meta_df : pd.DataFrame
        Dataframe of metadata as generated by pvdeg.weather.get for geospatial
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
    gids : np.array
        Array of gids associated with NSRDB entries inside the bounding box.
    """
    lats, longs = utilities._find_bbox_corners(
        coord_1=coord_1, coord_2=coord_2, coords=coords
    )

    latitude_mask = (meta_df["latitude"] >= lats[0]) & (meta_df["latitude"] <= lats[1])
    longitude_mask = (meta_df["longitude"] >= longs[0]) & (
        meta_df["longitude"] <= longs[1]
    )

    box_mask = latitude_mask & longitude_mask

    return meta_df[box_mask].index


# add bounding box?
def elevation_stochastic_downselect(
    meta_df: pd.DataFrame,
    kdtree,
    downselect_prop: float,
    k_neighbors: int = 3,
    method: str = "mean",
    normalization: str = "linear",
):
    """
    Downsample function.

    Downsample, assigning each point a weight associated with its neighbors changes in
    height. Randomly choose points based on weights to preferentially select points next
    to or in mountains while drastically lowering the density of points in flat areas to
    find a non-uniformly dense sub-sample of original data points.

    Parameters:
    -----------
    meta_df : pd.DataFrame
        Dataframe of metadata as generated by pvdeg.weather.get for geospatial
    kdtree : sklearn.neighbors.KDTree or str
        kdtree containing latitude-longitude pairs for quick lookups
        Generate using ``pvdeg.geospatial.meta_KDTree``. Can take a pickled
        kdtree as a path to the .pkl file.
    downselect_prop : float
        proportion of original datapoints to keep in output gids list
    k_neighbors : int, (default = 3)
        number of neighbors to check for elevation data in nearest neighbors
    method : str, (default = 'mean')
        method to calculate elevation weights for each point.
        Options : `'mean'`, `'sum'`, `'median'`
    normalization : str, (default = 'linear')
        function to apply when normalizing weights. Logarithmic uses log_e/ln
        options : `'linear'`, `'log'`, '`exp'`, `'invert-linear'`

    Returns:
    --------
    gids : np.array
        numpy array of downselected gids.
    """
    coords = np.column_stack([meta_df["latitude"], meta_df["longitude"]])
    elevations = meta_df["altitude"].values

    m = int(len(elevations) * downselect_prop)

    normalized_weights = utilities._calc_elevation_weights(
        elevations=elevations,
        coords=coords,
        k_neighbors=k_neighbors,
        method=method,
        normalization=normalization,
        kdtree=kdtree,
    )

    selected_indicies = np.random.choice(
        a=len(coords), p=normalized_weights / np.sum(normalized_weights), size=m
    )

    return meta_df.index.values[np.unique(selected_indicies)]
    # return meta_df.iloc[np.unique(selected_indicies)].index.values
    # return np.unique(selected_indicies)


def interpolate_analysis(
    result: xr.Dataset, data_var: str, method="nearest", resolution=100j
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate sparse spatial result data against DataArray coordinates.

    Takes DataArray instead of Dataset, index one variable of a dataset to get a
    dataarray.

    Parameters:
    -----------
    resolution: complex
        Change the amount the input is interpolated.
        For more interpolation set higher (200j is more than 100j)

    Result:
    -------
    """
    da_copy = deepcopy(result[data_var])

    df = da_copy.to_dataframe().reset_index()
    df = df.dropna(
        subset=[data_var]
    )  # there may be a better way to do this, why are we making a dataframe, not good
    data = np.column_stack(
        (df["latitude"], df["longitude"], df[data_var])
    )  # probably a nicer way to do this

    grid_lat, grid_lon = np.mgrid[
        df["latitude"].min() : df["latitude"].max() : resolution,
        df["longitude"].min() : df["longitude"].max() : resolution,
    ]

    grid_z = griddata(data[:, 0:2], data[:, 2], xi=(grid_lat, grid_lon), method=method)

    return grid_z, grid_lat, grid_lon


# api could be updated to match that of plot_USA
def plot_sparse_analysis(
    result: xr.Dataset,
    data_var: str,
    method="nearest",
    resolution=100j,
    cmap="viridis",
    ax=None,
) -> None:
    grid_values, lat, lon = interpolate_analysis(
        result=result, data_var=data_var, method=method, resolution=resolution
    )

    if ax is None:
        fig = plt.figure()
        ax = fig.add_axes(
            [0, 0, 1, 1], projection=ccrs.LambertConformal(), frameon=False
        )
        ax.patch.set_visible(False)
        show = True
    else:
        fig = None
        show = False

    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    ax.set_extent(extent)
    img = ax.imshow(
        grid_values,
        extent=extent,
        origin="lower",
        cmap=cmap,
        transform=ccrs.PlateCarree(),
    )

    shapename = "admin_1_states_provinces_lakes"
    states_shp = shpreader.natural_earth(
        resolution="110m", category="cultural", name=shapename
    )

    ax.add_geometries(
        shpreader.Reader(states_shp).geometries(),
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="gray",
    )

    if fig is not None:
        cbar = plt.colorbar(img, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
        cbar.set_label("Value")
        plt.title("Interpolated Heatmap")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

    if show and fig is not None:
        plt.show()

    return fig, ax
