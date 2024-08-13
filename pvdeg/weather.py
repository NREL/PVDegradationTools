"""
Collection of classes and functions to obtain spectral parameters.
"""

from pvdeg import humidity
from pvdeg.utilities import nrel_kestrel_check

from pvlib import iotools
import os
import glob
import pandas as pd
from rex import NSRDBX, Outputs
from pvdeg import humidity
import datetime
import numpy as np
import h5py
import dask.dataframe as dd
import xarray as xr


def get(database, id=None, geospatial=False, **kwargs):
    """
    Load weather data directly from  NSRDB or through any other PVLIB i/o
    tools function

    Parameters:
    -----------
    database : (str)
        'NSRDB' or 'PVGIS'
    id : (int or tuple)
        If NSRDB, id is the gid for the desired location. 
        If PVGIS, id is a tuple of (latitude, longitude) for the desired location
    geospatial : (bool)
        If True, initialize weather data via xarray dataset and meta data via
        dask dataframe. This is useful for large scale geospatial analyses on
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

    META_MAP = {"elevation": "altitude", "Local Time Zone": "tz"}

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
        if database == "NSRDB":
            weather_df, meta = get_NSRDB(gid=gid, location=location, **kwargs)
        elif database == "PVGIS":
            URL = "https://re.jrc.ec.europa.eu/api/v5_2/"
            weather_df, _, meta, _ = iotools.get_pvgis_tmy(
                latitude=lat, longitude=lon, url=URL, **kwargs
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

        for key in [*meta.keys()]:
            if key in META_MAP.keys():
                meta[META_MAP[key]] = meta.pop(key)

        if database == "NSRDB" or database == "PSM3":
            meta["wind_height"] = 2
            meta["Source"] = "NSRDB"
        elif database == "PVGIS":
            meta["wind_height"] = 10
            meta["Source"] = "PVGIS"
        else:
            meta["wind_height"] = None

        # switch weather data headers and metadata to pvlib standard
        map_weather(weather_df)
        map_meta(meta)

        if "relative_humidity" not in weather_df.columns:
            print('\r','Column "relative_humidity" not found in DataFrame. Calculating...', end='')
            weather_df = humidity._ambient(weather_df)
            print('\r', '                                                                    ',end='')
            print('\r', end='')

        return weather_df, meta

    elif geospatial:
        if database == "NSRDB":
            nrel_kestrel_check()

            weather_ds, meta_df = get_NSRDB(geospatial=geospatial, **kwargs)
            meta_df["wind_height"] = 2
        elif database == "local":
            fp = kwargs.pop("file")
            weather_ds, meta_df = ini_h5_geospatial(fp)
        else:
            raise NameError(f"Geospatial analysis not implemented for {database}.")

        return weather_ds, meta_df


def read(file_in, file_type, map_variables=True, **kwargs):
    """
    Read a locally stored weather file of any PVLIB compatible type

    #TODO: add error handling

    Parameters:
    -----------
    file_in : (path)
        full file path to the desired weather file
    file_type : (str)
        type of weather file from list below (verified)
        [psm3, tmy3, epw, h5, csv]
    """

    META_MAP = {"elevation": "altitude", "Local Time Zone": "tz"}

    supported = ["psm3", "tmy3", "epw", "h5", "csv"]
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
    elif file_type == "CSV":
        weather_df, meta = csv_read(filename=file_in)
    else:
        print(f"File-Type not recognized. supported types:\n{supported}")

    if not isinstance(meta, dict):
        meta = meta.to_dict()

    # map meta-names as needed
    if map_variables == True:
        map_weather(weather_df)
        map_meta(meta)

    if weather_df.index.tzinfo is None:
        tz = "Etc/GMT%+d" % -meta["tz"]
        weather_df = weather_df.tz_localize(tz)

    return weather_df, meta


def csv_read(filename):
    """
    Read a locally stored csv weather file. The first line contains the meta data
    variable names, and the second line contains the meta data values. This is followed
    by the meterological data.


    Parameters:
    -----------
    file_path : (str)
        file path and name of h5 file to be read

    Returns:
    --------
    weather_df : (pd.DataFrame)
        DataFrame of weather data
    meta : (dict)
        Dictionary of metadata for the weather data
    """

    file1 = open(filename, "r")
    # get the meta data from the first two lines
    metadata_fields = file1.readline().split(",")
    metadata_fields[-1] = metadata_fields[-1].strip()  # strip trailing newline
    metadata_values = file1.readline().split(",")
    metadata_values[-1] = metadata_values[-1].strip()  # strip trailing newline
    meta = dict(zip(metadata_fields, metadata_values))
    for (
        key
    ) in meta:  # converts everything to a float that is possible to convert to a float
        try:
            meta[key] = float(meta[key])
        except:
            pass
    # get the column headers
    columns = file1.readline().split(",")
    columns[-1] = columns[-1].strip()  # strip trailing newline
    # remove blank columns if they are there
    columns = [col for col in columns if col != ""]
    dtypes = dict.fromkeys(columns, float)  # all floats except datevec
    dtypes.update(Year=int, Month=int, Day=int, Hour=int, Minute=int)
    dtypes["Cloud Type"] = int
    dtypes["Fill Flag"] = int
    weather_df = pd.read_csv(
        file1,
        header=None,
        names=columns,
        usecols=columns,
        dtype=dtypes,
        delimiter=",",
        lineterminator="\n",
    )
    try:
        dtidx = pd.to_datetime(
            weather_df[["Year", "Month", "Day", "Hour", "Minute", "Second"]]
        )
    except:
        try:
            dtidx = pd.to_datetime(
                weather_df[["Year", "Month", "Day", "Hour", "Minute"]]
            )
        except:
            try:
                dtidx = pd.to_datetime(weather_df[["Year", "Month", "Day", "Hour"]])
            finally:
                dtidx = print(
                    "Your data file should have columns for Year, Month, Day, and Hour"
                )
    weather_df.index = pd.DatetimeIndex(dtidx)
    file1.close()

    return weather_df, meta


def map_meta(meta):
    """ "
    This will update the headings for meterological data to standard forms
    as outlined in https://github.com/DuraMAT/pv-terms.

    Returns:
    --------
    meta : dictionary
        DataFrame of weather data with modified column headers.
    """

    META_MAP = {
        "elevation": "altitude",
        "Elevation": "altitude",
        "Local Time Zone": "tz",
        "Time Zone": "tz",
        "timezone": "tz",
        "Dew Point": "dew_point",
        "Longitude": "longitude",
        "Latitude": "latitude",
    }

    # map meta-names as needed
    for key in [*meta.keys()]:
        if key in META_MAP.keys():
            meta[META_MAP[key]] = meta.pop(key)

    return meta


def map_weather(weather_df):
    """ "
    This will update the headings for meterological data to standard forms
    as outlined in https://github.com/DuraMAT/pv-terms.

    Returns:
    --------
    weather_df : (pd.DataFrame)
        DataFrame of weather data with modified column headers.
    """

    DSET_MAP = {
        "year": "Year",
        "month": "Month",
        "day": "Day",
        "hour": "Hour",
        "minute": "Minute",
        "second": "Second",
        "GHI": "ghi",
        "DHI": "dhi",
        "DNI": "dni",
        "Clearsky GHI": "ghi_clear",
        "Clearsky DHI": "dhi_clear",
        "Clearsky DNI": "dni_clear",
        "Solar Zenith Angle": "solar_zenith",
        "Temperature": "temp_air",
        "air_temperature": "temp_air",
        "Relative Humidity": "relative_humidity",
        "Dew Point": "dew_point",
        "temp_dew": "dew_point",
        "Pressure": "pressure",
        "Wind Speed": "wind_speed",
        "Wind Direction": "wind_direction",
        "Surface Albedo": "albedo",
        "Precipitable Water": "precipitable_water",
        "Module_Temperature": "module_temperature",
    }

    for column_name in weather_df.columns:
        if column_name in [*DSET_MAP.keys()]:
            weather_df.rename(
                columns={column_name: DSET_MAP[column_name]}, inplace=True
            )

    return weather_df


def read_h5(gid, file, attributes=None, **_):
    """
    Read a locally stored h5 weather file that follows NSRDB conventions.


    Parameters:
    -----------
    file : (str)
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
    drop_variables = ["meta", "time_index", "tmy_year", "tmy_year_short", "coordinates"]

    for i, fp in enumerate(fps):
        hf = h5py.File(fp, "r")
        attr = list(hf)
        attr_to_read = [elem for elem in attr if elem not in drop_variables]

        chunks = []
        shapes = []
        for var in attr_to_read:
            chunks.append(
                hf[var].chunks if hf[var].chunks is not None else (np.nan, np.nan)
            )
            shapes.append(
                hf[var].shape if hf[var].shape is not None else (np.nan, np.nan)
            )
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
            drop_variables=drop_variables,
            mask_and_scale=False,
            decode_cf=True,
        )

        for var in ds.data_vars:
            if hasattr(getattr(ds, var), "psm_scale_factor"):
                scale_factor = 1 / ds[var].psm_scale_factor
                getattr(ds, var).attrs["scale_factor"] = scale_factor

        # TODO: delete
        # if tuple(coords_len.values()) == (
        #     ds.sizes["phony_dim_0"],
        #     ds.sizes["phony_dim_1"],
        # ):
        #     rename = {"phony_dim_0": "time", "phony_dim_1": "gid"}
        # elif tuple(coords_len.values()) == (
        #     ds.sizes["phony_dim_1"],
        #     ds.sizes["phony_dim_0"],
        # ):
        #     rename = {"phony_dim_0": "gid", "phony_dim_1": "time"}
        # else:
        #     raise ValueError("Dimensions do not match for {}".format(var))
        rename = {}
        for (
            phony,
            length,
        ) in ds.sizes.items():
            if length == coords_len["time"]:
                rename[phony] = "time"
            elif length == coords_len["gid"]:
                rename[phony] = "gid"
        ds = ds.rename(rename)
        ds = ds.assign_coords(coords)

        # TODO: In case re-chunking becomes necessary
        # ax0 = list(ds.sizes.keys())[list(ds.sizes.values()).index(shapes[0])]
        # ax1 = list(ds.sizes.keys())[list(ds.sizes.values()).index(shapes[1])]
        # ds = ds.chunk(chunks={ax0:chunks[0], ax1:chunks[1]})
        dss.append(ds)

    ds = xr.merge(dss)
    ds = xr.decode_cf(ds)

    # Rechunk time axis
    ds = ds.unify_chunks()
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
        hpc_fp = "/datasets/NSRDB/"
        hsds = False
    else:
        hpc_fp = "/nrel/nsrdb/v3/"
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
    satellite=None,
    names="TMY",
    NREL_HPC=False,
    gid=None,
    location=None,
    geospatial=False,
    attributes=None,
    **_,
):
    """
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
    META_MAP = {"elevation": "altitude", "Local Time Zone": "tz", "timezone": "tz"}

    if (
        satellite == None
    ):  # TODO: This function is not fully written as of January 3, 2024
        satellite, gid = get_satellite(location)
    if not geospatial:
        nsrdb_fnames, hsds = get_NSRDB_fnames(
            satellite=satellite, names=names, NREL_HPC=NREL_HPC
        )

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


def get_satellite(location):
    """
    identify a satellite to use for a given lattitude and longitude. This is to provide default values worldwide, but a more
    experienced user may want to specify a specific satellite to get better data.

    Provide a location tuple.

    Parameters:
    -----------
    location : (tuple)
        (latitude, longitude) for the desired location

    Returns:
    --------
    satellite : (str)
        'GOES', 'METEOSAT', 'Himawari', 'SUNY', 'CONUS', 'Americas'
    gid : (int)
        gid for the desired location
    """

    # this is just a placeholder till the actual code gets programmed.
    satellite = "PSM3"

    # gid = f.lat_lon_gid(lat_lon=location) # I couldn't get this to work
    gid = None
    return satellite, gid


def write(data_df, metadata, savefile="WeatherFile.csv"):
    """
    Saves dataframe with weather data and any associated meta data in an *.csv format.
    The metadata will be formatted on the first two lines with the first being the descriptor
    and the second line being the value. Then the meterological, time and other data series
    headers on on the third line with all the subsequent data on the remaining lines. This
    format can be read by the PVDeg software.

    Parameters
    -----------
    data_df : pandas.DataFrame
        timeseries data.
    metdata : dictionary
        Dictionary with 'latitude', 'longitude', 'altitude', 'source',
        'tz' for timezone, and other meta data.
    savefile : str
        Name of file to save output as.
        Name of file to save output as.
    standardSAM : boolean
        This checks the dataframe to avoid having a leap day, then averages it
        to SAM style (closed to the right),
        and fills the years so it starst on YEAR/1/1 0:0 and ends on
        YEAR/12/31 23:00.
    includeminute ; Bool
        For hourly data, if SAM input does not have Minutes, it calculates the
        sun position 30 minutes prior to the hour (i.e. 12 timestamp means sun
         position at 11:30).
        If minutes are included, it will calculate the sun position at the time
        of the timestamp (12:00 at 12:00)
        Set to true if resolution of data is sub-hourly.
        Name of file to save output as.
    standardSAM : boolean
        This checks the dataframe to avoid having a leap day, then averages it
        to SAM style (closed to the right),
        and fills the years so it starst on YEAR/1/1 0:0 and ends on
        YEAR/12/31 23:00.
    includeminute ; Bool
        For hourly data, if SAM input does not have Minutes, it calculates the
        sun position 30 minutes prior to the hour (i.e. 12 timestamp means sun
         position at 11:30).
        If minutes are included, it will calculate the sun position at the time
        of the timestamp (12:00 at 12:00)
        Set to true if resolution of data is sub-hourly.

    Returns
    -------
    Nothing, it just writes the file.

    """

    meta_string = (
        ", ".join(str(key) for key, value in metadata.items())
        + "\n"
        + ", ".join(str(value) for key, value in metadata.items())
    )

    result_df = pd.concat([data_df], axis=1).reindex()

    savedata = result_df.to_string(index=False).split("\n")
    savedata.pop(0)
    savedata = [",".join(ele.split()) for ele in savedata]
    savedata = "\n".join(savedata)
    columns = list(
        data_df.columns
    )  # This had to be pulled out separately because spaces can get turned into commas in the header names.
    str1 = ""
    for ele in columns:
        str1 = str1 + ele + ","
    savedata = meta_string + "\n" + str1 + "\n" + savedata

    file1 = open(savefile, "w")
    file1.writelines(savedata)
    file1.close()

def get_anywhere(database = "PSM3", id=None, **kwargs):
    """
    Load weather data directly from  NSRDB or through any other PVLIB i/o
    tools function. Only works for a single location look-up, not for geospatial analysis.

    Parameters:
    -----------
    database : (str)
        'PSM3' or 'PVGIS'
        Indicates the first database to try. PSM3 is for the NSRDB
    id : (int or tuple)
        The gid or tuple with latitude and longitude for the desired location. 
        Using a gid is not recommended because it is specific to one database.
    API_KEY : (str)
        This is used to access the NSRDB without limitation if a custom key is supplied.
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

    weather_arg = {'api_key': 'DEMO_KEY',  #Pass in a custom key to avoid access limitations.
            'email': 'user@mail.com',
            'names': 'tmy',
            'attributes': [],
            'map_variables': True,
            'geospatial': False}
    weather_arg.update(kwargs) #Will default to the kwargs passed to the function.

    if database == "PSM3":
        try:
            weather_db, meta = get(database='PSM3', id=id, **weather_arg)
        except:
            try:
                weather_db, meta = get(database='PVGIS', id=id, **{'map_variables': True} )
            except:
                meta = {'result': 'This location was not found in either the NSRDB or PVGIS'}
                weather_db = {'result': 'NA'}
    else:
        try:
            weather_db, meta = get(database='PVGIS', id=id, **{'map_variables': True})
        except:
            try:
                weather_db, meta = get(database='PSM3', id=id, **weather_arg)
            except:
                meta = {'result': 'This location was not found in either the NSRDB or PVGIS'}
                weather_db = {'result': 'NA'}

    return weather_db, meta 