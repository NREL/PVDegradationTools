"""
Collection of classes and functions for geospatial analysis.
"""

from . import standards
from . import humidity
from . import letid

import xarray as xr
import dask.array as da
import pandas as pd
from dask.distributed import Client, LocalCluster

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader


def start_dask(hpc=None):
    """
    Starts a dask cluster for parallel processing.

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
            'memory': '256GB',
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
    client.wait_for_workers(n_workers=1)

    return client


def calc_gid(ds_gid, meta_gid, func, **kwargs):
    """
    Calculates a single gid for a given function.

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

    df_weather = ds_gid.to_dataframe()
    df_res = func(weather_df=df_weather, meta=meta_gid, **kwargs)
    ds_res = xr.Dataset.from_dataframe(df_res)

    if not df_res.index.name:
        ds_res = ds_res.isel(index=0, drop=True)

    return ds_res


def calc_block(weather_ds_block, future_meta_df, func, func_kwargs):
    """
    Calculates a block of gids for a given function.

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

    res = weather_ds_block.groupby("gid").map(
        lambda ds_gid: calc_gid(
            ds_gid=ds_gid,
            meta_gid=future_meta_df.loc[ds_gid["gid"].values].to_dict(),
            func=func,
            **func_kwargs,
        )
    )
    return res


def analysis(weather_ds, meta_df, func, template=None, **func_kwargs):
    """
    Applies a function to each gid of a weather dataset.

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
        param = template_parameters(func)
        template = output_template(weather_ds, **param)

    # future_meta_df = client.scatter(meta_df)
    kwargs = {"func": func, "future_meta_df": meta_df, "func_kwargs": func_kwargs}

    stacked = weather_ds.map_blocks(
        calc_block, kwargs=kwargs, template=template
    ).compute()

    # lats = stacked.latitude.values.flatten()
    # lons = stacked.longitude.values.flatten()
    stacked = stacked.drop(["gid"])
    # stacked = stacked.drop_vars(['latitude', 'longitude'])
    stacked.coords["gid"] = pd.MultiIndex.from_arrays(
        [meta_df["latitude"], meta_df["longitude"]], names=["latitude", "longitude"]
    )

    res = stacked.unstack("gid")  # , sparse=True
    return res


def output_template(ds_gids, shapes, attrs=dict(), add_dims=dict()):
    """
    Generates a xarray template for output data. Output variables and
    associated dimensions need to be specified via the shapes dictionary.
    The dimension length are derived from the input data. Additonal output
    dimensions can be defined with the add_dims argument.

    Parameters
    ----------
    ds_gids : xarray.Dataset
        Dataset containing the gids and their associated dimensions.
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
    dims_size = dict(ds_gids.dims) | add_dims

    output_template = xr.Dataset(
        data_vars={
            var: (dim, da.empty([dims_size[d] for d in dim]))
            for var, dim in shapes.items()
        },
        coords={dim: ds_gids[dim] for dim in dims},
        attrs=attrs,
    ).chunk({dim: ds_gids.chunks[dim] for dim in dims})

    return output_template


def template_parameters(func):
    """
    Output parameters for xarray template.

    Returns
    -------
    shapes : dict
        Dictionary of variable names and their associated dimensions.
    attrs : dict
        Dictionary of attributes for each variable (e.g. units).
    add_dims : dict
        Dictionary of dimensions to add to the output template.
    """

    if func == standards.standoff:
        shapes = {
            "x": ("gid",),
            "T98_inf": ("gid",),
            "T98_0": ("gid",),
        }

        attrs = {
            "x": {"units": "cm"},
            "T98_0": {"units": "Celsius"},
            "T98_inf": {"units": "Celsius"},
        }

        add_dims = {}

    elif func == humidity.module:
        shapes = {
            "RH_surface_outside": ("gid", "time"),
            "RH_front_encap": ("gid", "time"),
            "RH_back_encap": ("gid", "time"),
            "RH_backsheet": ("gid", "time"),
        }

        attrs = {}

        add_dims = {}

    elif func == letid.calc_letid_outdoors:
        shapes = {
            "Temperature": ("gid", "time"),
            "Injection": ("gid", "time"),
            "NA": ("gid", "time"),
            "NB": ("gid", "time"),
            "NC": ("gid", "time"),
            "tau": ("gid", "time"),
            "Jsc": ("gid", "time"),
            "Voc": ("gid", "time"),
            "Isc": ("gid", "time"),
            "FF": ("gid", "time"),
            "Pmp": ("gid", "time"),
            "Pmp_norm": ("gid", "time"),
        }

        attrs = {}

        add_dims = {}

    else:
        raise ValueError(f"No preset output template for function {func}.")

    parameters = {"shapes": shapes, "attrs": attrs, "add_dims": add_dims}

    return parameters


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
        facecolor="w",
        edgecolor="gray",
    )

    cm = xr_res.plot(
        transform=ccrs.PlateCarree(),
        zorder=10,
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
        plt.savefig(fp, dpi=600)

    return fig, ax
