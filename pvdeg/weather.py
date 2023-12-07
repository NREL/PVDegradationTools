"""
Collection of classes and functions to obtain spectral parameters.
"""

from pvlib import iotools
import os
import glob
import pandas as pd
from rex import NSRDBX, Outputs
from pvdeg import humidity
import datetime

import h5py
import dask.dataframe as dd
import xarray as xr


def get(database, id=None, geospatial=False, **kwargs):
    """
    Load weather data directly from  NSRDB or through any other PVLIB i/o
    Load weather data directly from  NSRDB or through any other PVLIB i/o
    tools function

    Parameters:
    -----------
    database : (str)
        'NSRDB' or 'PVGIS'
    id : (int or tuple)
        If NSRDB, id is the gid for the desired location
        If PVGIS, id is a tuple of (latitude, longitude) for the desired location
    geospatial : (bool)
        If True, initialize weather data via xarray dataset and meta data via
        dask dataframe. This is useful for large scael geospatial analyses on
        distributed compute systems. Geospaital analyses are only supported for
        NSRDB data and locally stored h5 files that follow pvlib conventions.
    **kwargs :
        Additional keyword arguments to pass to the get_weather function
        (see pvlib.iotools.get_psm3 for PVGIS, and get_NSRDB for NSRDB)

    Returns:
    --------
    weather_df : (pd.DataFrame)
        DataFrame of weather data
    meta : (dict)
        Dictionary of metadata for the weather data
    """

    META_MAP = {"elevation": "altitude", "Local Time Zone": "timezone"}

    if type(id) is tuple:
        location = id
        gid = None
        lat = location[0]
        lon = location[1]
    elif type(id) is int:
        gid = id
        location = None
    elif id is None:
        if not geospatial:
            raise TypeError(
                "Specify location via tuple (latitude, longitude), or gid integer."
            )

    if not geospatial:
        # TODO: decide wether to follow NSRDB or pvlib conventions...
        # e.g. temp_air vs. air_temperature
        # "map variables" will guarantee PVLIB conventions (automatic in coming update) which is "temp_air"
        if database == "NSRDB":
            weather_df, meta = get_NSRDB(gid=gid, location=location, **kwargs)
        elif database == "PVGIS":
            weather_df, _, meta, _ = iotools.get_pvgis_tmy(
                latitude=lat, longitude=lon, map_variables=True, **kwargs
            )
            meta = meta["location"]
        elif database == "PSM3":
            weather_df, meta = iotools.get_psm3(latitude=lat, longitude=lon, **kwargs)
        elif database == "local":
            fp = kwargs.pop("file")
            fn, fext = os.path.splitext(fp)
            weather_df, meta = read(gid=gid, file_in=fp, file_type=fext[1:], **kwargs)
        else:
            raise NameError("Weather database not found.")

        if "relative_humidity" not in weather_df.columns:
            print('Column "relative_humidity" not found in DataFrame. Calculating...')
            weather_df = humidity._ambient(weather_df)

        # map meta-names as needed
        for key in [*meta.keys()]:
            if key in META_MAP.keys():
                meta[META_MAP[key]] = meta.pop(key)

        return weather_df, meta

    elif geospatial:
        if database == "NSRDB":
            weather_ds, meta_df = get_NSRDB(geospatial=geospatial, **kwargs)
        elif database == "local":
            fp = kwargs.pop("file")
            weather_ds, meta_df = ini_h5_geospatial(fp)
        else:
            raise NameError(f"Geospatial analysis not implemented for {database}.")

        return weather_ds, meta_df


def read(file_in, file_type, **kwargs):
    """
    Read a locally stored weather file of any PVLIB compatible type

    #TODO: add error handling

    Parameters:
    -----------
    file_in : (path)
        full file path to the desired weather file
    file_type : (str)
        type of weather file from list below (verified)
        [psm3, tmy3, epw, h5]
    """

    META_MAP = {"elevation": "altitude", "Local Time Zone": "timezone"}

    supported = ["psm3", "tmy3", "epw", "h5"]
    file_type = file_type.upper()

    if file_type in ["PSM3", "PSM"]:
        weather_df, meta = iotools.read_psm3(filename=file_in, map_variables=True)
    elif file_type in ["TMY3", "TMY"]:
        weather_df, meta = iotools.read_tmy3(
            filename=file_in
        )  # map variable not worki - check pvlib for map_variables
    elif file_type == "EPW":
        weather_df, meta = iotools.read_epw(filename=file_in)
    elif file_type == "H5":
        weather_df, meta = read_h5(file=file_in, **kwargs)
    else:
        print(f"File-Type not recognized. supported types:\n{supported}")

    if not isinstance(meta, dict):
        meta = meta.to_dict()

    # map meta-names as needed
    for key in [*meta.keys()]:
        if key in META_MAP.keys():
            meta[META_MAP[key]] = meta.pop(key)

    return weather_df, meta


def read_h5(gid, file, attributes=None, **_):
    """
    Read a locally stored h5 weather file that follows NSRDB conventions.


    Parameters:
    -----------
    file_path : (str)
        file path and name of h5 file to be read
    gid : (int)
        gid for the desired location
    attributes : (list)
        List of weather attributes to extract from NSRDB

    Returns:
    --------
    weather_df : (pd.DataFrame)
        DataFrame of weather data
    meta : (dict)
        Dictionary of metadata for the weather data
    """

    if os.path.dirname(file):
        fp = file
    else:
        fp = os.path.join(os.path.dirname(__file__), os.path.basename(file))
    if os.path.dirname(file):
        fp = file
    else:
        fp = os.path.join(os.path.dirname(__file__), os.path.basename(file))

    with Outputs(fp, mode="r") as f:
        meta = f.meta.loc[gid]
        index = f.time_index
        dattr = f.attrs

    # TODO: put into utilities
    if attributes == None:
        attributes = list(dattr.keys())
        try:
            attributes.remove("meta")
            attributes.remove("tmy_year_short")
        except ValueError:
            pass

    weather_df = pd.DataFrame(index=index, columns=attributes)
    for dset in attributes:
        with Outputs(fp, mode="r") as f:
            weather_df[dset] = f[dset, :, gid]

    return weather_df, meta.to_dict()


def ini_h5_geospatial(fps):
    """
    initialize an h5 weather file that follows NSRDB conventions for
    geospatial analyses.

    Parameters:
    -----------
    file_path : (str)
        file path and name of h5 file to be read
    gid : (int)
        gid for the desired location
    attributes : (list)
        List of weather attributes to extract from NSRDB

    Returns:
    --------
    weather_df : (pd.DataFrame)
        DataFrame of weather data
    meta : (dict)
        Dictionary of metadata for the weather data
    """
    dss = []
    for i, fp in enumerate(fps):
        hf = h5py.File(fp, "r")
        attr = list(hf)
        attr_to_read = [elem for elem in attr if elem not in ["meta", "time_index"]]

        chunks = []
        shapes = []
        for var in attr_to_read:
            chunks.append(hf[var].chunks)
            shapes.append(hf[var].shape)
        chunks = min(set(chunks))
        shapes = min(set(shapes))

        if i == 0:
            time_index = pd.to_datetime(hf["time_index"][...].astype(str)).values
            meta_df = pd.read_hdf(fp, key="meta")
            coords = {"gid": meta_df.index.values, "time": time_index}
            coords_len = {"time": time_index.shape[0], "gid": meta_df.shape[0]}

        ds = xr.open_dataset(
            fp,
            engine="h5netcdf",
            phony_dims="sort",
            chunks={"phony_dim_0": chunks[0], "phony_dim_1": chunks[1]},
            drop_variables=["time_index", "meta"],
            mask_and_scale=False,
            decode_cf=True,
        )

        for var in ds.data_vars:
            if hasattr(getattr(ds, var), "psm_scale_factor"):
                scale_factor = 1 / ds[var].psm_scale_factor
                getattr(ds, var).attrs["scale_factor"] = scale_factor

        if tuple(coords_len.values()) == (
            ds.dims["phony_dim_0"],
            ds.dims["phony_dim_1"],
        ):
            rename = {"phony_dim_0": "time", "phony_dim_1": "gid"}
        elif tuple(coords_len.values()) == (
            ds.dims["phony_dim_1"],
            ds.dims["phony_dim_0"],
        ):
            rename = {"phony_dim_0": "gid", "phony_dim_1": "time"}
        else:
            raise ValueError("Dimensions do not match")
        ds = ds.rename(
            {"phony_dim_0": rename["phony_dim_0"], "phony_dim_1": rename["phony_dim_1"]}
        )
        ds = ds.assign_coords(coords)

        # TODO: In case re-chunking becomes necessary
        # ax0 = list(ds.dims.keys())[list(ds.dims.values()).index(shapes[0])]
        # ax1 = list(ds.dims.keys())[list(ds.dims.values()).index(shapes[1])]
        # ds = ds.chunk(chunks={ax0:chunks[0], ax1:chunks[1]})
        dss.append(ds)

    ds = xr.merge(dss)
    ds = xr.decode_cf(ds)

    # Rechunk time axis
    ds = ds.chunk(chunks={"time": -1, "gid": ds.chunks["gid"]})

    weather_ds = ds

    return weather_ds, meta_df


def get_NSRDB_fnames(satellite, names, NREL_HPC=False, **_):
    """
    Get a list of NSRDB files for a given satellite and year

    Parameters:
    -----------
    satellite : (str)
        'GOES', 'METEOSAT', 'Himawari', 'SUNY', 'CONUS', 'Americas'
    names : (int or str)
        PVLIB naming convention year or 'TMY':
        If int, year of desired data
        If str, 'TMY' or 'TMY3'
    NREL_HPC : (bool)
        If True, use NREL HPC path
        If False, use AWS path

    Returns:
    --------
    nsrdb_fnames : (list)
        List of NSRDB files for a given satellite and year
    hsds : (bool)
        If True, use h5pyd to access NSRDB files
        If False, use h5py to access NSRDB files
    """

    sat_map = {
        "GOES": "full_disc",
        "METEOSAT": "meteosat",
        "Himawari": "himawari",
        "SUNY": "india",
        "CONUS": "conus",
        "Americas": "current",
    }

    if NREL_HPC:
        hpc_fp = "/kfs2/pdatasets/NSRDB/"
        hsds = False
    else:
        hpc_fp = "/nrel/nsrdb/"
        hsds = True

    if type(names) in [int, float]:
        nsrdb_fp = os.path.join(
            hpc_fp, sat_map[satellite], "*_{}.h5".format(int(names))
        )
        nsrdb_fnames = glob.glob(nsrdb_fp)
    else:
        nsrdb_fp = os.path.join(
            hpc_fp, sat_map[satellite], "*_{}*.h5".format(names.lower())
        )
        nsrdb_fnames = glob.glob(nsrdb_fp)

    if len(nsrdb_fnames) == 0:
        raise FileNotFoundError(
            "Couldn't find NSRDB input files! \nSearched for: '{}'".format(nsrdb_fp)
        )

    return nsrdb_fnames, hsds


def get_NSRDB(
    satellite,
    names,
    NREL_HPC,
    gid=None,
    location=None,
    geospatial=False,
    attributes=None,
    **_,
):
    """
    Get NSRDB weather data from different satellites and years.
    Get NSRDB weather data from different satellites and years.
    Provide either gid or location tuple.

    Parameters:
    -----------
    satellite : (str)
        'GOES', 'METEOSAT', 'Himawari', 'SUNY', 'CONUS', 'Americas'
    names : (int or str)
        If int, year of desired data
        If str, 'TMY' or 'TMY3'
    NREL_HPC : (bool)
        If True, use NREL HPC path
        If False, use AWS path
    gid : (int)
        gid for the desired location
    location : (tuple)
        (latitude, longitude) for the desired location
    attributes : (list)
        List of weather attributes to extract from NSRDB

    Returns:
    --------
    weather_df : (pd.DataFrame)
        DataFrame of weather data
    meta : (dict)
        Dictionary of metadata for the weather data
    """

    DSET_MAP = {"air_temperature": "temp_air", "Relative Humidity": "relative_humidity"}

    META_MAP = {"elevation": "altitude"}

    if not geospatial:
        nsrdb_fnames, hsds = get_NSRDB_fnames(satellite, names, NREL_HPC)

        dattr = {}
        for i, file in enumerate(nsrdb_fnames):
            with NSRDBX(file, hsds=hsds) as f:
                if i == 0:
                    if gid == None:  # TODO: add exception handling
                        gid = f.lat_lon_gid(location)
                    meta = f["meta", gid].iloc[0]
                    index = f.time_index

                lattr = f.datasets
                for attr in lattr:
                    dattr[attr] = file

        if attributes == None:
            attributes = list(dattr.keys())
            try:
                attributes.remove("meta")
                attributes.remove("tmy_year_short")
            except ValueError:
                pass

        weather_df = pd.DataFrame(index=index)

        for dset in attributes:
            # switch dset names to pvlib standard
            if dset in [*DSET_MAP.keys()]:
                column_name = DSET_MAP[dset]
            else:
                column_name = dset

            with NSRDBX(dattr[dset], hsds=hsds) as f:
                weather_df[column_name] = f[dset, :, gid]

        # switch meta key names to pvlib standard
        re_idx = []
        for key in [*meta.index]:
            if key in META_MAP.keys():
                re_idx.append(META_MAP[key])
            else:
                re_idx.append(key)
        meta.index = re_idx

        return weather_df, meta.to_dict()

    elif geospatial:
        nsrdb_fnames, hsds = get_NSRDB_fnames(satellite, names, NREL_HPC)
        weather_ds, meta_df = ini_h5_geospatial(nsrdb_fnames)

        if attributes is not None:
            weather_ds = weather_ds[attributes]

        for dset in weather_ds.data_vars:
            if dset in DSET_MAP.keys():
                weather_ds = weather_ds.rename({dset: DSET_MAP[dset]})

        for mset in meta_df.columns:
            if mset in META_MAP.keys():
                meta_df.rename(columns={mset: META_MAP[mset]}, inplace=True)

        return weather_ds, meta_df


def repeat_annual_time_series(time_series, start_year, n_years):
    """
    Repeat a pandas time series dataframe containing annual data.
    For example, repeat TMY data by n_years, adding in leap days as necessary.
    For now, this function requires 1 or more full years of uniform
    interval (non-leap year) data, i.e. length must be a multiple of 8760.
    On leap days, all data is set to 0.

    TODO: make it possible to have weirder time series, e.g. non uniform intervals.
    Include option for synthetic leap day data

    Parameters:
    -----------
    time_series : (pd.DataFrame)
        pandas dataframe with DatetimeIndex

    time_series : (int)
        desired starting year of time_series

    n_years : (int)
        number of years to repeat time_series

    Returns:
    --------
    new_time_series : (pd.DataFrame)
        pandas dataframe repeated n_years
    """

    if len(time_series) % 8760 != 0:
        raise ValueError("Length of time_series must be a multiple of 8760")

    tz = time_series.index.tz
    time_series = time_series.tz_localize(
        None
    )  # timezone aware timeseries can cause problems, we'll make it tz-naive for now

    time_series.index = time_series.index.map(lambda dt: dt.replace(year=start_year))

    start = time_series.index[0]

    for year in range(start_year, start_year + n_years):
        if year == start_year:
            if is_leap_year(year):
                this_year = time_series.copy()
                this_year.index = time_series.index.map(
                    lambda dt: dt.replace(year=year)
                )
                this_year = pd.concat(
                    [
                        this_year[: str(year) + "-02-28"],
                        pd.DataFrame(
                            0,
                            index=pd.date_range(
                                start=datetime.datetime(
                                    year=year, month=2, day=29, minute=start.minute
                                ),
                                end=datetime.datetime(year=year, month=3, day=1),
                                freq="H",
                            ),
                            columns=time_series.columns,
                        ),
                        this_year[str(year) + "-03-01" :],
                    ]
                )
                new_time_series = this_year

            else:
                this_year = time_series.copy()
                this_year.index = time_series.index.map(
                    lambda dt: dt.replace(year=year)
                )
                new_time_series = this_year

        else:
            if is_leap_year(year):
                this_year = time_series.copy()
                this_year.index = time_series.index.map(
                    lambda dt: dt.replace(year=year)
                )
                this_year = pd.concat(
                    [
                        this_year[: str(year) + "-02-28"],
                        pd.DataFrame(
                            0,
                            index=pd.date_range(
                                start=datetime.datetime(
                                    year=year, month=2, day=29, minute=start.minute
                                ),
                                end=datetime.datetime(year=year, month=3, day=1),
                                freq="H",
                            ),
                            columns=time_series.columns,
                        ),
                        this_year[str(year) + "-03-01" :],
                    ]
                )
                new_time_series = pd.concat([new_time_series, this_year])

            else:
                this_year = time_series.copy()
                this_year.index = time_series.index.map(
                    lambda dt: dt.replace(year=year)
                )
                new_time_series = pd.concat([new_time_series, this_year])

    new_time_series.index = new_time_series.index.tz_localize(
        tz=tz
    )  # add back in the timezone

    return new_time_series


def is_leap_year(year):
    """Returns True if year is a leap year"""
    if year % 4 != 0:
        return False
    elif year % 100 != 0:
        return True
    elif year % 400 != 0:
        return False
    else:
        return True
