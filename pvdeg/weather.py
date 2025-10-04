"""Collection of classes and functions to obtain spectral parameters."""

from pvdeg import humidity
from pvdeg.utilities import nrel_kestrel_check

from pvlib import iotools
import os
import glob
import pandas as pd
from rex import NSRDBX, Outputs
import datetime
import numpy as np
import h5py
from dask.delayed import delayed
import xarray as xr
from geopy.geocoders import Nominatim


# Global dataset mapping for standardizing weather variable names across different
# weather data sources
META_MAP = {
    "elevation": "altitude",
    "Elevation": "altitude",
    "Local Time Zone": "tz",
    "Time Zone": "tz",
    "timezone": "tz",
    "Longitude": "longitude",
    "Latitude": "latitude",
    "state": "State",
    "county": "County",
    "country": "Country",
    "Neighborhood": "neighbourhood",
    "country_code": "Country Code",
    "postcode": "Zipcode",
    "road": "Street",
    "village": "City",
    "city": "City",
    "town": "City",
}

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
    "surface_albedo": "albedo",
    "Precipitable Water": "precipitable_water",
    "Module_Temperature": "module_temperature",
}


TIME_PERIODICITY_MAP = {
    # pandas time freq string arg
    # ideally these should be the same
    "h": 8760,
    "1h": 8760,
    "30min": 17520,
    "15min": 35040,
}

ENTRIES_PERIODICITY_MAP = {
    # pandas time freq string arg
    # ideally these should be the same
    8760: "1h",
    17520: "30min",
    35040: "15min",
}


def get(
    database,
    id=None,
    geospatial=False,
    find_meta=False,
    **kwargs,
):
    """
    Load weather data directly from  NSRDB or through any other PVLIB i/o
    tools function.

    Parameters
    ----------
    database : (str)
        'NSRDB' or 'PVGIS'. Use "PSM3" for tmy NSRDB data.
    id : (int or tuple)
        If NSRDB, id is the gid for the desired location.
        If PVGIS, id is a tuple of (latitude, longitude) for the desired location
    geospatial : (bool)
        If True, initialize weather data via xarray dataset and meta data via
        dask dataframe. This is useful for large scale geospatial analyses on
        distributed compute systems. Geospatial analyses are only supported for
        NSRDB data and locally stored h5 files that follow pvlib conventions.
    find_meta : (bool)
        If true, this instructs the code to look up additional meta data.
        This only works for single locations and not for distributed data download or
        geospatial analysis. The default is False.
    **kwargs :
        Additional keyword arguments to pass to the get_weather function
        (see pvlib.iotools.get_psm3 for NSRDB)

    Returns
    -------
    weather_df : (pd.DataFrame)
        DataFrame of weather data
    meta : (dict)
        Dictionary of metadata for the weather data

    Example
    -------
    Collecting a single site of PSM3 NSRDB data. *Api key and email must be replaced
    with your personal api key and email*.
    [Request a key!](https://developer.nrel.gov/signup/)

    .. code-block:: python

        weather_arg = {
            'api_key': <api_key>,
            'email': <email>,
            'names': 'tmy',
            'attributes': [],
            'map_variables': True
        }

        weather_df, meta_dict =
        pvdeg.weather.get(database="PSM3",id=(25.783388, -80.189029), **weather_arg)

    Collecting a single site of PVGIS TMY data

    .. code-block:: python

        weather_df, meta_dict = pvdeg.weather.get(database="PVGIS", id=(49.95, 1.5))

    Collecting geospatial data from NSRDB on Kestrel (NREL INTERNAL USERS ONLY)

    satellite options:
        ``"GOES", "METEOSAT", "Himawari", "SUNY", "CONUS", "Americas"``


    .. code-block:: python

        weather_db = "NSRDB"
        weather_arg = {
            "satellite": "Americas",
            "names": "TMY",
            "NREL_HPC": True,
            "attributes": [
                    "air_temperature",
                    "wind_speed",
                    "dhi",
                    "ghi",
                    "dni",
                    "relative_humidity",
                ],
        }

        geo_weather, geo_meta = pvdeg.weather.get(
            weather_db, geospatial=True, **weather_arg
        )
    """

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
            weather_df, meta = iotools.get_pvgis_tmy(
                latitude=lat, longitude=lon, url=URL, **kwargs
            )
            inputs = meta["inputs"]
            meta = inputs["location"]
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
        if find_meta:
            meta = find_metadata(meta)

        if "relative_humidity" not in weather_df.columns:
            print(
                "\r",
                'Column "relative_humidity" not found in DataFrame. Calculating...',
                end="",
            )
            temp_air = weather_df["temp_air"]
            dew_point = weather_df.get("dew_point")
            if dew_point is None or temp_air is None:
                raise ValueError(
                    'Cannot calculate "relative_humidity": one of'
                    '"dew_point" or "temp_air" column not found in'
                    "DataFrame."
                )
            weather_df["relative_humidity"] = humidity.relative(temp_air, dew_point)
            print(
                "\r",
                "                                                                    ",
                end="",
            )
            print("\r", end="")

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


def read(file_in, file_type, map_variables=True, find_meta=False, **kwargs):
    """
    Read a locally stored weather file of any PVLIB compatible type

    #TODO: add error handling

    Parameters
    ----------
    file_in : (path)
        full file path to the desired weather file
    file_type : (str)
        type of weather file from list below (verified)
        [psm3, tmy3, epw, h5, csv]
    """

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
        print(f"File-Type not recognized. supported types: \n{supported}")

    if not isinstance(meta, dict):
        meta = meta.to_dict()

    # map meta-names as needed
    if map_variables is True:
        map_weather(weather_df)
        map_meta(meta)
    if find_meta:
        meta = find_metadata(meta)

    if weather_df.index.tzinfo is None:
        tz = "Etc/GMT%+d" % -meta["tz"]
        weather_df = weather_df.tz_localize(tz)

    return weather_df, meta


def csv_read(filename):
    """Read a locally stored csv weather file.

    The first line contains the meta data variable names, and the second line contains
    the meta data values. This is followed by the meterological data.

    Parameters
    ----------
    file_path : (str)
        file path and name of h5 file to be read

    Returns
    -------
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
        except Exception:
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
    except Exception:
        try:
            dtidx = pd.to_datetime(
                weather_df[["Year", "Month", "Day", "Hour", "Minute"]]
            )
        except Exception:
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
    """
    Update meteorological metadata keys/columns to standard forms as outlined in
    https://github.com/DuraMAT/pv-terms.

    Parameters
    ----------
    meta : dict or pandas.DataFrame
        Single-site metadata (dict) or multi-site geospatial metadata (DataFrame).

    Returns
    -------
    meta : dict or pandas.DataFrame
         Metadata with standardized keys/column names.
    """

    # Rename keys in dict
    if isinstance(meta, dict):
        for key in [*meta.keys()]:
            if key in META_MAP.keys():
                meta[META_MAP[key]] = meta.pop(key)
        if "Country Code" in meta.keys():
            meta["Country Code"] = meta["Country Code"].upper()
        return meta

    # Rename columns in DataFrame
    elif isinstance(meta, pd.DataFrame):
        rename_map = {k: v for k, v in META_MAP.items() if k in meta.columns}
        return meta.rename(columns=rename_map)

    else:
        raise TypeError(f"Input must be dict or pandas.DataFrame, got {type(meta)}")


def map_weather(weather_df):
    """

    Update the headings for meterological data to standard forms.

    Standard outlined in https://github.com/DuraMAT/pv-terms.

    Returns
    -------
    weather_df : (pd.DataFrame)
        DataFrame of weather data with modified column headers.
    """

    if isinstance(weather_df, pd.DataFrame):
        for column_name in weather_df.columns:
            if column_name in [*DSET_MAP.keys()]:
                weather_df.rename(
                    columns={column_name: DSET_MAP[column_name]}, inplace=True
                )

        return weather_df

    elif isinstance(weather_df, xr.Dataset):
        weather_df = weather_df.rename(
            {
                key: value
                for key, value in DSET_MAP.items()
                if key in weather_df.data_vars
            }
        )

        return weather_df

    else:
        raise TypeError("input must be pd.DataFrame or xr.Dataset")


def read_h5(gid, file, attributes=None, **_):
    """Read a locally stored h5 weather file that follows NSRDB conventions.

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
    if attributes is None:
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
    Initialize h5 weather file that follows NSRDB conventions for geospatial analyses.

    Parameters
    ----------
    file_path : (str)
        file path and name of h5 file to be read
    gid : (int)
        gid for the desired location
    attributes : (list)
        List of weather attributes to extract from NSRDB

    Returns
    -------
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
    """Get a sorted list of NSRDB files for a given satellite and year.

    Parameters
    ----------
    satellite : (str)
        'GOES', 'METEOSAT', 'Himawari', 'SUNY', 'CONUS', 'Americas'
    names : (int or str)
        PVLIB naming convention year or 'TMY':
        If int, year of desired data
        If str, 'TMY' or 'TMY3'
    NREL_HPC : (bool)
        If True, use NREL HPC path
        If False, use AWS path

    Returns
    -------
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

    nsrdb_fnames = sorted(nsrdb_fnames)
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
    """Get NSRDB weather data from different satellites and years.

    Provide either gid or location tuple.

    Parameters
    ----------
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

    Returns
    -------
    weather_df : (pd.DataFrame)
        DataFrame of weather data
    meta : (dict)
        Dictionary of metadata for the weather data
    """

    if (
        satellite is None
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
                    if gid is None:  # TODO: add exception handling
                        gid = f.lat_lon_gid(location)
                    meta = f["meta", gid].iloc[0]
                    index = f.time_index

                lattr = f.datasets
                for attr in lattr:
                    dattr[attr] = file

        if attributes is None:
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
        # new versions have multiple files per satellite-year to reduce filesizes
        # this is great for yearly data but TMY has multiple files
        # the year attached to the TMY file in the filesystem/name is seemingly
        # the year it was created. this creates problems, we only want to combine the
        # files if they are NOT TMY

        nsrdb_fnames, hsds = get_NSRDB_fnames(satellite, names, NREL_HPC)

        if isinstance(names, str) and names.lower() in ["tmy", "tmy3"]:
            # maintain as list with last element of sorted list
            nsrdb_fnames = nsrdb_fnames[-1:]

        weather_ds, meta_df = ini_h5_geospatial(nsrdb_fnames)

        weather_ds = weather_ds.assign_attrs({"kestrel_nsrdb_fnames": nsrdb_fnames})

        # select desired weather attributes
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
    """Repeat a pandas time series dataframe containing annual data.

    For example, repeat
    TMY data by n_years, adding in leap days as necessary. For now, this function
    requires 1 or more full years of uniform interval (non-leap year) data, i.e. length
    must be a multiple of 8760. On leap days, all data is set to 0.

    TODO: make it possible to have weirder time series, e.g. non uniform intervals.
    Include option for synthetic leap day data

    Parameters
    ----------
    time_series : (pd.DataFrame)
        pandas dataframe with DatetimeIndex

    time_series : (int)
        desired starting year of time_series

    n_years : (int)
        number of years to repeat time_series

    Returns
    -------
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
    """Return True if year is a leap year."""
    if year % 4 != 0:
        return False
    elif year % 100 != 0:
        return True
    elif year % 400 != 0:
        return False
    else:
        return True


def get_satellite(location):
    """Identify a satellite to use for a given lattitude and longitude.

    This is to
    provide default values worldwide, but a more experienced user may want to specify a
    specific satellite to get better data.

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
    """Save dataframe with weather data and any associated meta data in an *.csv format.

    The metadata will be formatted on the first two lines with the first being
    the descriptor and the second line being the value. Then the meterological, time and
    other data series headers on on the third line with all the subsequent data on the
    remaining lines. This format can be read by the PVDeg software.

    Parameters
    ----------
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
    )  # pulled out separately as spaces can get turned into commas in the header names.
    str1 = ""
    for ele in columns:
        str1 = str1 + ele + ","
    savedata = meta_string + "\n" + str1 + "\n" + savedata

    file1 = open(savefile, "w")
    file1.writelines(savedata)
    file1.close()


def get_anywhere(database="PSM3", id=None, **kwargs):
    """
    Load weather data directly from NSRDB or through any other PVLIB i/o tools.

    Only works for a single location look-up, not for geospatial analysis.

    Parameters:
    -----------
    database : (str)
        'PSM3' or 'PVGIS'
        Indicates the first database to try. PSM3 is for the NSRDB
    id : (int or tuple)
        The gid or tuple with latitude and longitude for the desired location.
        Using a gid is not recommended because it is specific to one database.
    API_KEY : (str)
        This is used to access the NSRDB without limitation if a custom key
        is supplied.
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

    weather_arg = {
        "api_key": "DEMO_KEY",  # Pass in a custom key to avoid access limitations.
        "email": "user@mail.com",
        "names": "tmy",
        "attributes": [],
        "map_variables": True,
        "geospatial": False,
        "find_meta": True,
    }
    weather_arg.update(kwargs)  # Will default to the kwargs passed to the function.

    if database == "PSM3":
        try:
            weather_db, meta = get(database="PSM3", id=id, **weather_arg)
        except Exception:
            try:
                weather_db, meta = get(
                    database="PVGIS", id=id, **{"map_variables": True}
                )
            except Exception:
                meta = {
                    "result": "This location was not found in either the NSRDB or PVGIS"
                }
                weather_db = {"result": "NA"}
    else:
        try:
            weather_db, meta = get(database="PVGIS", id=id, **{"map_variables": True})
        except Exception:
            try:
                weather_db, meta = get(database="PSM3", id=id, **weather_arg)
            except Exception:
                meta = {
                    "result": "This location was not found in either the NSRDB or PVGIS"
                }
                weather_db = {"result": "NA"}

    return weather_db, meta


def roll_tmy(weather_df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Wrap ends of TMY UTC DataFrame to align with local time.

    Aligns with local time based on timezone offset.

    Parameters:
    ----------
    weather_df : pd.DataFrame
        The input DataFrame containing TMY data with a UTC datetime index.
    meta : dict
        Metadata dictionary containing at least the 'tz' key, representing
        timezone offset in hours (e.g., -8 for UTC-8).

    Returns:
    -------
    pd.DataFrame
        The rolled DataFrame aligned to local times with a new datetime index
        spanning a typical year.

    Raises:
    ------
    ValueError
        If the timezone offset is not a multiple of the data frequency or if
        the frequency cannot be inferred.
    """

    # Extract timezone offset in hours
    tz_offset = meta.get("tz", 0)  # Default to UTC if not specified

    # Step 1: Localize the index to UTC
    weather_df_local = weather_df.copy()
    weather_df_local.index = pd.to_datetime(weather_df_local.index)
    weather_df_local = weather_df_local.tz_localize("UTC")

    # Step 2: Convert to desired local timezone
    # 'Etc/GMT+X' corresponds to UTC-X
    if tz_offset >= 0:
        local_tz = f"Etc/GMT-{tz_offset}"
    else:
        local_tz = f"Etc/GMT+{abs(tz_offset)}"

    try:
        weather_df_local = weather_df_local.tz_convert(local_tz)
    except Exception as e:
        raise ValueError(f"Invalid timezone offset: {tz_offset}. Error: {e}")

    # Step 3: Make timezone-naive
    weather_df_naive = weather_df_local.tz_localize(None)

    # Step 4: Determine frequency
    freq = pd.infer_freq(weather_df_naive.index)
    if freq is None:
        raise ValueError(
            "Cannot infer frequency of the DataFrame index. Ensure it is regular."
        )

    # Step 5: Calculate the shift amount
    # To align local time to start at 00:00, shift by -tz_offset hours
    # For example, tz_offset = -8 (UTC-8) => shift by +8 hours
    total_shift = pd.Timedelta(hours=-tz_offset)

    if freq.isalpha():
        freq = "1" + freq

    row_timedelta = pd.to_timedelta(
        freq
    )  # this probably broke because it was a string without an hourly frequency

    if total_shift % row_timedelta != pd.Timedelta(0):
        raise ValueError("Timezone offset must be a multiple of the data frequency.")

    num_shift = int(total_shift / row_timedelta)

    # Step 6: Perform the shift (roll the DataFrame)
    if num_shift > 0:
        rearranged = pd.concat(
            [weather_df_naive.iloc[num_shift:], weather_df_naive.iloc[:num_shift]]
        )
    elif num_shift < 0:
        rearranged = pd.concat(
            [weather_df_naive.iloc[num_shift:], weather_df_naive.iloc[:num_shift]]
        )
    else:
        rearranged = weather_df_naive.copy()

    # Step 7: Assign a new datetime index spanning a typical non-leap year
    # Preserve the original start time's hour, minute, second, etc.
    # Using year 2001 as it is not a leap year

    # Extract the time component from the first timestamp
    original_start_time = rearranged.index[0].time()
    start_time = pd.Timestamp("2001-01-01") + pd.Timedelta(
        hours=0,
        minutes=original_start_time.minute,
    )

    expected_num_rows = rearranged.shape[0]

    # Create the new datetime index with the preserved start time
    new_index = pd.date_range(start=start_time, periods=expected_num_rows, freq=freq)

    # Handle potential leap day if present in new_index
    # Since 2001 is not a leap year, ensure no Feb 29 exists
    new_index = new_index[~((new_index.month == 2) & (new_index.day == 29))]

    # Assign the new index to the rearranged DataFrame
    rearranged = rearranged.iloc[: len(new_index)]  # Ensure lengths match
    rearranged.index = new_index

    return rearranged


# RENAME, THIS SHOULD NOT REFERENCE PVGIS
def _process_weather_result_distributed(weather_df):
    """Create xarray.Dataset using np.array backend from pvgis weather dataframe."""
    import dask.array as da

    weather_df.index.rename("time", inplace=True)
    weather_ds = weather_df.to_xarray().drop_vars("time").copy()

    for var in weather_ds.data_vars:
        dask_array = da.from_array(weather_ds[var].values, chunks="auto")

        weather_ds[var] = (weather_ds[var].dims, dask_array)

    return weather_ds


@delayed
def _weather_distributed_vec(
    database: str,
    coord: float,
    api_key: str,  # NSRDB api key
    email: str,  # NSRDB developer email
):
    """
    Distributed weather calculation for use with dask futures/delayed

    Parameters
    ----------
    database: str
        database/source from `pvdeg.weather.get`
    coord: tuple[float]
        (latitude, longitude) coordinate pair. (`pvdeg.weather.get` id)
    api_key: str
        NSRDB developer api key (see `pvdeg.weather.get`)
    email: str
        NSRDB developer email (see `pvdeg.weather.get`)

    Returns
    --------
        Returns ds, dict, None if unsucessful
        Returns None, None, Exception if unsucessful
    """

    # we want to fail loudly, quickly
    if database == "PVGIS":  # does not need api key
        weather_df, meta_dict = get(database=database, id=coord)
    elif database == "PSM3":
        weather_df, meta_dict = get(
            database=database, id=coord, api_key=api_key, email=email
        )
    else:
        raise NotImplementedError(
            f'database {database} not implemented, options: "PVGIS", "PSM3"'
        )

    # convert single location dataframe to xarray dataset
    weather_ds = _process_weather_result_distributed(weather_df=weather_df)

    return weather_ds, meta_dict, None


# THE NSRDB shapes could be moved to their own definition
# organization style question?
def empty_weather_ds(gids_size, periodicity, database) -> xr.Dataset:
    """
    Create an empty weather dataframe for generalized input.

    Parameters
    ---------
    gids_size: int
        number of entries to create along gid axis
    periodicity: str
        freqency, pandas `freq` string arg from `pd.date_range`.

        .. code-block:: python
            "1h"
            "30min"
            "15min"
    database: str
        database from `pvdeg.weather.get`

    Returns
    -------
    weather_ds: xarray.Dataset
        Weather dataset of the same format/shapes given by a
        `pvdeg.weather.get` geospatial call or
        `pvdeg.weather.weather_distributed` call or
        GeosptialScenario.get_geospatial_data`.
    """
    import dask.array as da

    pvgis_shapes = {
        "temp_air": ("gid", "time"),
        "relative_humidity": ("gid", "time"),
        "ghi": ("gid", "time"),
        "dni": ("gid", "time"),
        "dhi": ("gid", "time"),
        "IR(h)": ("gid", "time"),
        "wind_speed": ("gid", "time"),
        "wind_direction": ("gid", "time"),
        "pressure": ("gid", "time"),
    }

    nsrdb_shapes = {
        "Year": ("gid", "time"),
        "Month": ("gid", "time"),
        "Day": ("gid", "time"),
        "Hour": ("gid", "time"),
        "Minute": ("gid", "time"),
        "temp_air": ("gid", "time"),
        "dew_point": ("gid", "time"),
        "dhi": ("gid", "time"),
        "dni": ("gid", "time"),
        "ghi": ("gid", "time"),
        "albedo": ("gid", "time"),
        "pressure": ("gid", "time"),
        "wind_direction": ("gid", "time"),
        "wind_speed": ("gid", "time"),
        "relative_humidity": ("gid", "time"),
    }

    attrs = {}
    global_attrs = {}

    dims_size = {"time": TIME_PERIODICITY_MAP[periodicity], "gid": gids_size}

    if database == "NSRDB" or database == "PSM3":
        # shapes = shapes | nsrdb_extra_shapes
        shapes = nsrdb_shapes
    elif database == "PVGIS":
        shapes = pvgis_shapes
    else:
        raise ValueError(f"database must be PVGIS, NSRDB, PSM3 not {database}")

    weather_ds = xr.Dataset(
        data_vars={
            var: (dim, da.empty([dims_size[d] for d in dim]), attrs.get(var))
            for var, dim in shapes.items()
        },
        coords={
            "time": pd.date_range(
                "2022-01-01",
                freq=periodicity,
                periods=TIME_PERIODICITY_MAP[periodicity],
            ),
            "gid": np.linspace(0, gids_size - 1, gids_size, dtype=int),
        },
        attrs=global_attrs,
    )

    return weather_ds


# add some check to see if a dask client exists
# can force user to pass dask client to ensure it exists
# if called without dask client we will return a xr.Dataset
# with dask backend that does not appear as if it failed until we compute it


# TODO: implement rate throttling so we do not make too many requests.
# TODO: multiple API keys to get around NSRDB key rate limit. 2 key, email pairs means
# twice the speed ;)
# TODO: this overwrites NSRDB GIDS when database == "PSM3"


def weather_distributed(
    database: str,
    coords: list[tuple],
    api_key: str = "",
    email: str = "",
):
    """
    Grab weather using pvgis for all locations using dask for parallelization.

    You must create a dask client with multiple processes before calling this
    function, otherwise results will not be properly calculated.

    PVGIS supports up to 30 requests per second so your dask client should not
    have more than $x$ workers/threads that would put you over this limit.

    NSRDB (including `database="PSM3"`) is rate limited and your key will face
    restrictions after making too many requests.
    See rates [here](https://developer.nrel.gov/docs/solar/nsrdb/guide/).

    Parameters
    ----------
    database : (str)
        'PVGIS' or 'NSRDB' (not implemented yet)
    coords: list[tuple]
        list of tuples containing (latitude, longitude) coordinates

        .. code-block:: python

            coords_example = [
                (49.95, 1.5),
                (51.95, -9.5),
                (51.95, -8.5),
                (51.95, -4.5),
                (51.95, -3.5)]

    api_key: str
        Only required when making NSRDB requests using "PSM3".
        [NSRDB developer API key](https://developer.nrel.gov/signup/)

    email: str
        Only required when making NSRDB requests using "PSM3".
        [NSRDB developer account email associated with
        `api_key`](https://developer.nrel.gov/signup/)

    Returns
    -------
    weather_ds : xr.Dataset
        Weather data for all locations requested in an xarray.Dataset using a
        dask array backend.
    meta_df : pd.DataFrame
        Pandas DataFrame containing metadata for all requested locations. Each
        row maps to a single entry in the weather_ds.
    gids_failed: list
        list of index failed coordinates in input `coords`
    """

    import dask.delayed
    import dask.distributed

    try:
        client = dask.distributed.get_client()
        print("Connected to a Dask scheduler | Dashboard:", client.dashboard_link)
    except ValueError:
        raise RuntimeError("No Dask scheduler found. Ensure a dask client is running.")

    if database != "PVGIS" and database != "PSM3":
        raise NotImplementedError(
            f"Only 'PVGIS' and 'PSM3' are implemented, you entered {database}"
        )

    delays = [
        _weather_distributed_vec(database, coord, api_key, email) for coord in coords
    ]

    futures = client.compute(delays)
    results = client.gather(futures)

    # results is a 2d list
    # results[0] is the weather_ds with dask backend
    # results[1] is meta_dict
    weather_ds_collection = [row[0] for row in results]
    meta_dict_collection = [row[1] for row in results]

    indexes_failed = []

    time_length = weather_ds_collection[0].sizes["time"]
    periodicity = ENTRIES_PERIODICITY_MAP[time_length]

    # weather_ds = pvgis_hourly_empty_weather_ds(len(results)) # create empty weather
    # xr.dataset
    weather_ds = empty_weather_ds(
        gids_size=len(results),
        periodicity=periodicity,
        database=database,
    )

    meta_df = pd.DataFrame.from_dict(
        meta_dict_collection
    )  # create populated meta pd.DataFrame

    # gids are spatially meaningless if data is from PVGIS, they will only show
    # corresponding entries between weather_ds and meta_df
    # only meaningfull if data is from NSRDB
    # this loop can be refactored, it is a little weird
    for i, row in enumerate(results):
        if row[2]:
            indexes_failed.append(i)
            continue

        weather_ds[dict(gid=i)] = weather_ds_collection[i]

    return weather_ds, meta_df, indexes_failed


def find_metadata(meta):
    """
    Fills in missing meta data for a geographic location.
    The meta dictionary must have longitude and latitude information.
    Make sure meta_map has been run first to eliminate the creation of duplicate entries
    with different names. It will only replace empty keys and those with one character
    of length.

    Parameters:
    -----------
    meta : (dict)
        Dictionary of metadata for the weather data

    Returns:
    --------
    meta : (dict)
        Dictionary of metadata for the weather data
    """
    geolocator = Nominatim(user_agent="geoapiexercises")
    location = geolocator.reverse(
        str(meta["latitude"]) + "," + str(meta["longitude"])
    ).raw["address"]
    map_meta(location)

    for key in [*location.keys()]:
        if key in meta.keys():
            if len(meta[key]) < 2:
                meta[key] = location[key]
        else:
            meta[key] = location[key]

    return meta


# def _nsrdb_to_uniform(weather_df: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, dict]:  # noqa

#     map_weather(weather_df=weather_df)
#     map_meta(meta)

# check if weather is localized, convert to GMT (like pvgis)
# check if time index is on the hour or 30 minutes
# weather_df.index - pd.Timedelta("30m")

# NSRDB datavars
# Year  Month  Day  Hour  Minute  dew_point  dhi
# dni  ghi  albedo  pressure  temp_air
# wind_direction  wind_speed  relative_humidity

# weather_dropables = ['Year',  'Month',  'Day',  'Hour',  'Minute',  'dew_point']
# meta_dropables = [...]

# NSRDB meta
# {'Source': 'NSRDB',
#  'Location ID': '145809',
#  'City': '-',
#  'State': '-',
#  'Country': '-',
#  'Dew Point Units': 'c',
#  'DHI Units': 'w/m2',
#  'DNI Units': 'w/m2',
#  'GHI Units': 'w/m2',
#  'Temperature Units': 'c',
#  'Pressure Units': 'mbar',
#  'Wind Direction Units': 'Degrees',
#  'Wind Speed Units': 'm/s',
#  'Surface Albedo Units': 'N/A',
#  'Version': '3.2.0',
#  'latitude': 39.73,
#  'longitude': -105.18,
#  'altitude': 1820,
#  'tz': -7,
#  'wind_height': 2}
# ...

# def _pvgis_to_uniform(
# weather_df: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, dict]:

# map_weather(weather_df=weather_df)
# map_meta(meta)

# drop meaningless variables

# pvgis datavars
# temp_air  relative_humidity   ghi  dni   dhi
# IR(h)  wind_speed  wind_direction  pressure

# weather_dropables = ["IR(h)"]
# meta_dropables = ['irradiance_time_offset', ...]

# pvgis meta
# {'latitude': 24.7136,
#   'longitude': 46.6753,
#   'irradiance_time_offset': -0.1955,
#   'altitude': 646.0,
#   'wind_height': 10,
#   'Source': 'PVGIS'})
# ...
