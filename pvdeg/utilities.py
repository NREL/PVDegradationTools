"""utilities.py."""

import os
import json
import pandas as pd
import numpy as np
from rex import NSRDBX, Outputs
from pvdeg import DATA_DIR
from typing import Callable
from random import choices
from string import ascii_uppercase
from collections import OrderedDict
import xarray as xr
from subprocess import run
import cartopy.feature as cfeature


# A mapping to simplify access to files stored in `pvdeg/data`
pvdeg_datafiles = {
    "AApermeation": os.path.join(DATA_DIR, "AApermeation.json"),
    "H2Opermeation": os.path.join(DATA_DIR, "H2Opermeation.json"),
    "O2permeation": os.path.join(DATA_DIR, "O2permeation.json"),
    "DegradationDatabase": os.path.join(DATA_DIR, "DegradationDatabase.json"),
    "albedo.json": os.path.join(DATA_DIR, "albedo.json"),
}


def gid_downsampling(meta, n):
    """Downsample the NSRDB GID grid by a factor of n.

    Parameters:
    -----------
    meta : (pd.DataFrame)
        DataFrame of NSRDB meta data
    n : (int)
        Downsample factor

    Returns:
    --------
    meta_sub : (pd.DataFrame)
        DataFrame of NSRDB meta data
    gids_sub : (list)
        List of GIDs for the downsampled NSRDB meta data
    """
    if n == 0:
        gids_sub = meta.index.values
        return meta, gids_sub

    lon_sub = sorted(meta["longitude"].unique())[0 : -1 : max(1, 2 * n)]
    lat_sub = sorted(meta["latitude"].unique())[0 : -1 : max(1, 2 * n)]

    gids_sub = meta[
        (meta["longitude"].isin(lon_sub)) & (meta["latitude"].isin(lat_sub))
    ].index.values

    meta_sub = meta.loc[gids_sub]

    return meta_sub, gids_sub


def meta_as_dict(rec):
    """Turn a numpy recarray record into a dict.

    Parameters
    ----------
    rec : (np.recarray)
        numpy structured array with labels as dtypes

    Returns
    -------
     : (dict)
        dictionary of numpy structured array
    """
    return {name: rec[name].item() for name in rec.dtype.names}


def get_kinetics(name=None, fname="kinetic_parameters.json"):
    """Return a list of LETID/B-O LID kinetic parameters from kinetic_parameters.json.

    Parameters
    ----------
    name : str
        unique name of kinetic parameter set. If None, returns a list of the possible
        options.

    Returns
    -------
    parameter_dict : (dict)
        dictionary of kinetic parameters
    """
    fpath = os.path.join(DATA_DIR, fname)

    with open(fpath) as f:
        data = json.load(f)

    # TODO: rewrite to use exception handling
    if name is None:
        parameters_list = data.keys()
        return "Choose a set of kinetic parameters:", [*parameters_list]

    kinetic_parameters = data[name]
    return kinetic_parameters


def write_gids(
    nsrdb_fp,
    region="Colorado",
    region_col="state",
    lat_long=None,
    gids=None,
    out_fn="gids",
):
    """Generate a .CSV file containing the GIDs for the spatial test range.

    The .CSV  file will be saved to the working directory.

    TODO: specify output file name and directory?

    Parameters:
    -----------
    nsrdb_fp : (str, path_obj)
        full file path to the NSRDB h5 file containing the weather data and GIDs
    region : (str, default = "Colorado")
        Name of the NSRDB region you are filtering into the GID list
    region_col : (str, default = "Sate")
        Name of the NSRDB region type
    lat_long : (tuple)
        Either single (Lat, Long) or series of (lat,Long) pairs
    out_fd : (str, default = "gids")
        Name of data column you want to retrieve. Generally, this should be "gids"

    Return
    ------
    project_points_path : (str)
        File path to the newly created "gids.csv"
    """
    if not gids:
        with NSRDBX(nsrdb_fp, hsds=False) as f:
            if lat_long:
                gids = f.lat_lon_gid(lat_long)
                if isinstance(gids, int):
                    gids = [gids]
            else:
                gids = f.region_gids(region=region, region_col=region_col)

    file_out = f"{out_fn}.csv"
    df_gids = pd.DataFrame(gids, columns=["gid"])
    df_gids.to_csv(file_out, index=False)

    return file_out


def _get_state(id):
    """Return the full name of a state based on two-letter state code.

    Parameters
    ----------
    id : (str)
        two letter state code (example: CO, AZ, MD)

    Returns
    -------
    state_name : (str)
        full name of US state (example: Colorado, Arizona, Maryland)
    """
    state_dict = {
        "AK": "Alaska",
        "AL": "Alabama",
        "AR": "Arkansas",
        "AS": "American Samoa",
        "AZ": "Arizona",
        "CA": "California",
        "CO": "Colorado",
        "CT": "Connecticut",
        "DC": "District of Columbia",
        "DE": "Delaware",
        "FL": "Florida",
        "GA": "Georgia",
        "GU": "Guam",
        "HI": "Hawaii",
        "IA": "Iowa",
        "ID": "Idaho",
        "IL": "Illinois",
        "IN": "Indiana",
        "KS": "Kansas",
        "KY": "Kentucky",
        "LA": "Louisiana",
        "MA": "Massachusetts",
        "MD": "Maryland",
        "ME": "Maine",
        "MI": "Michigan",
        "MN": "Minnesota",
        "MO": "Missouri",
        "MP": "Northern Mariana Islands",
        "MS": "Mississippi",
        "MT": "Montana",
        "NA": "National",
        "NC": "North Carolina",
        "ND": "North Dakota",
        "NE": "Nebraska",
        "NH": "New Hampshire",
        "NJ": "New Jersey",
        "NM": "New Mexico",
        "NV": "Nevada",
        "NY": "New York",
        "OH": "Ohio",
        "OK": "Oklahoma",
        "OR": "Oregon",
        "PA": "Pennsylvania",
        "PR": "Puerto Rico",
        "RI": "Rhode Island",
        "SC": "South Carolina",
        "SD": "South Dakota",
        "TN": "Tennessee",
        "TX": "Texas",
        "UT": "Utah",
        "VA": "Virginia",
        "VI": "Virgin Islands",
        "VT": "Vermont",
        "WA": "Washington",
        "WI": "Wisconsin",
        "WV": "West Virginia",
        "WY": "Wyoming",
    }
    state_name = state_dict[id]
    return state_name


def get_state_bbox(
    abbr: str = None,
) -> np.ndarray:
    """Retrieve top left and bottom right coordinate pairs for state bounding boxes."""
    # can move to its own file in pvdeg.DATA_DIR
    bbox_dict = {
        "Alabama": [
            [-84.8882446289062, 35.0080299377441],
            [-88.4731369018555, 30.1375217437744],
        ],
        "Alaska": [
            [-129.9795, 71.4410],
            [-179.1505, 51.2097],
        ],
        "Arizona": [
            [-109.045196533203, 37.0042610168457],
            [-114.818359375, 31.3321762084961],
        ],
        "Arkansas": [
            [-89.6422424316406, 36.4996032714844],
            [-94.6178131103516, 33.0041046142578],
        ],
        "California": [
            [-114.13077545166, 42.0095024108887],
            [-124.482009887695, 32.5295219421387],
        ],
        "Colorado": [
            [-102.041580200195, 41.0023612976074],
            [-109.060256958008, 36.9924240112305],
        ],
        "Connecticut": [
            [-71.7869873046875, 42.0505905151367],
            [-73.7277755737305, 40.9667053222656],
        ],
        "Delaware": [
            [-74.9846343994141, 39.8394355773926],
            [-75.7890472412109, 38.4511260986328],
        ],
        "District Of Columbia": [
            [-76.8369, 39.1072],
            [-77.2369, 38.7072],
        ],
        "Florida": [
            [-79.9743041992188, 31.0009689331055],
            [-87.6349029541016, 24.3963069915771],
        ],
        "Georgia": [
            [-80.7514266967773, 35.0008316040039],
            [-85.6051712036133, 30.3557567596436],
        ],
        "Hawaii": [
            [-154.8066, 22.2356],
            [160.2471, 189117],
        ],
        "Idaho": [
            [-111.043563842773, 49.000846862793],
            [-117.243034362793, 41.9880561828613],
        ],
        "Illinois": [
            [-87.0199203491211, 42.5083045959473],
            [-91.513053894043, 36.9701309204102],
        ],
        "Indiana": [
            [-84.7845764160156, 41.7613716125488],
            [-88.0997085571289, 37.7717399597168],
        ],
        "Iowa": [
            [-90.1400604248047, 43.5011367797852],
            [-96.6397171020508, 40.3755989074707],
        ],
        "Kansas": [
            [-94.5882034301758, 40.0030975341797],
            [-102.0517578125, 36.9930801391602],
        ],
        "Kentucky": [
            [-81.9645385742188, 39.1474609375],
            [-89.5715103149414, 36.4967155456543],
        ],
        "Louisiana": [
            [-88.817008972168, 33.019458770752],
            [-94.0431518554688, 28.9210300445557],
        ],
        "Maine": [
            [-66.9250717163086, 47.4598426818848],
            [-71.0841751098633, 42.9561233520508],
        ],
        "Maryland": [
            [-75.0395584106445, 39.7229347229004],
            [-79.4871978759766, 37.8856391906738],
        ],
        "Massachusetts": [
            [-69.8615341186523, 42.8867149353027],
            [-73.5081481933594, 41.1863288879395],
        ],
        "Michigan": [
            [-82.122802734375, 48.3060646057129],
            [-90.4186248779297, 41.6960868835449],
        ],
        "Minnesota": [
            [-89.4833831787109, 49.3844909667969],
            [-97.2392654418945, 43.4994277954102],
        ],
        "Mississippi": [
            [-88.0980072021484, 34.9960556030273],
            [-91.6550140380859, 30.1477890014648],
        ],
        "Missouri": [
            [-89.0988388061523, 40.6136360168457],
            [-95.7741470336914, 35.9956817626953],
        ],
        "Montana": [
            [-104.039558410645, 49.0011100769043],
            [-116.050003051758, 44.3582191467285],
        ],
        "Nebraska": [
            [-95.3080520629883, 43.0017013549805],
            [-104.053520202637, 39.9999961853027],
        ],
        "Nevada": [
            [-114.039642333984, 42.0022087097168],
            [-120.005729675293, 35.0018730163574],
        ],
        "New Hampshire": [
            [-70.534065246582, 45.3057823181152],
            [-72.55712890625, 42.6970405578613],
        ],
        "New Jersey": [
            [-73.8850555419922, 41.3574256896973],
            [-75.5633926391602, 38.7887535095215],
        ],
        "New Mexico": [
            [-103.000862121582, 37.0001411437988],
            [-109.050178527832, 31.3323001861572],
        ],
        "New York": [
            [-71.8527069091797, 45.0158615112305],
            [-79.7625122070312, 40.4773979187012],
        ],
        "North Carolina": [
            [-75.4001159667969, 36.5880393981934],
            [-84.3218765258789, 33.7528762817383],
        ],
        "North Dakota": [
            [-96.5543899536133, 49.0004920959473],
            [-104.049270629883, 45.9350357055664],
        ],
        "Ohio": [
            [-80.5189895629883, 42.3232383728027],
            [-84.8203430175781, 38.4031982421875],
        ],
        "Oklahoma": [
            [-94.4312133789062, 37.0021362304688],
            [-103.002571105957, 33.6191940307617],
        ],
        "Oregon": [
            [-116.463500976562, 46.2991027832031],
            [-124.703544616699, 41.9917907714844],
        ],
        "Pennsylvania": [
            [-74.6894989013672, 42.5146903991699],
            [-80.5210876464844, 39.7197647094727],
        ],
        "Rhode Island": [
            [-71.1204681396484, 42.018856048584],
            [-71.9070053100586, 41.055534362793],
        ],
        "South Carolina": [
            [-78.4992980957031, 35.2155418395996],
            [-83.35400390625, 32.0333099365234],
        ],
        "South Dakota": [
            [-96.4363327026367, 45.9454536437988],
            [-104.05770111084, 42.4798889160156],
        ],
        "Tennessee": [
            [-81.6468963623047, 36.6781196594238],
            [-90.310302734375, 34.9829788208008],
        ],
        "Texas": [
            [-93.5078201293945, 36.5007057189941],
            [-106.645652770996, 25.8370609283447],
        ],
        "Utah": [
            [-109.041069030762, 42.0013885498047],
            [-114.053932189941, 36.9979667663574],
        ],
        "Vermont": [
            [-71.4653549194336, 45.0166664123535],
            [-73.437744140625, 42.7269325256348],
        ],
        "Virginia": [
            [-75.2312240600586, 39.4660148620605],
            [-83.6754150390625, 36.5407867431641],
        ],
        "Washington": [
            [-116.917427062988, 49.00244140625],
            [-124.836097717285, 45.5437202453613],
        ],
        "West Virginia": [
            [-77.7190246582031, 40.638801574707],
            [-82.6447448730469, 37.2014808654785],
        ],
        "Wisconsin": [
            [-104.052154541016, 45.0034217834473],
            [-111.05689239502, 40.9948768615723],
        ],
    }

    name = _get_state(abbr)
    return np.array(bbox_dict[name])


def convert_tmy(file_in, file_out="h5_from_tmy.h5"):
    """Read a older TMY-like weather file and convert to h5 for use in pvdeg.

    TODO: figure out scale_facator and np.int32 for smaller file
          expand for international locations?

    Parameters:
    -----------
    file_in : (str, path_obj)
        full file path to existing weather file
    file_out : (str, path_obj)
        full file path and name of file to create.
    """
    from pvlib import iotools

    src_data, src_meta = iotools.tmy.read_tmy3(file_in, coerce_year=2023)

    save_cols = [
        "dni",
        "dhi",
        "ghi",
        "temp_air",
        "relative_humidity",
        "wind_speed",
        "albedo",
    ]

    df_new = src_data[save_cols].copy()
    time_index = df_new.index

    meta = {
        "latitude": [src_meta["latitude"]],
        "longitude": [src_meta["longitude"]],
        "elevation": [src_meta["altitude"]],
        "timezone": [src_meta["TZ"]],
        "country": ["United States"],
        "state": [_get_state(src_meta["State"])],
    }
    meta = pd.DataFrame(meta)

    with Outputs(file_out, "w") as f:
        f.meta = meta
        f.time_index = time_index

    for col in df_new.columns:
        Outputs.add_dataset(
            h5_file=file_out,
            dset_name=col,
            dset_data=df_new[col].values,
            attrs={"scale_factor": 100},
            dtype=np.int64,
        )


# currently this is only designed for Oxygen Permeation. It could easily be adapted for
# all permeation data.
def _add_material(
    name,
    alias,
    Ead,
    Eas,
    So,
    Do=None,
    Eap=None,
    Po=None,
    fickian=True,
    fp=DATA_DIR,
    fname="O2permeation.json",
):
    """Add a new material to the materials database.

    Check the parameters for
    specific units. If material already exists, parameters will be updated.

    TODO: check if material is already existing

    Parameters:
    -----------
    name : (str)
        Unique material name
    alias : (str)
        Material alias (ex: PET1, EVA)
    Ead : (float)
        Diffusivity Activation Energy [kJ/mol]
    Eas : (float)
        Solubility Activation Energy [kJ/mol]
    So : (float)
        Solubility Prefactor [g/cm^3]
    Do : (float)
        Diffusivity Prefactor [cm^2/s] (unused)
    Eap : (float)
        Permeability Activation Energy [kJ/mol] (unused)
    Po : (float)
        Permeability Prefactor [g*mm/m^2/day] (unused)
    fickian : (boolean)
        I have no idea what this means (unused)
    fp : (str)
        file path to the json materials file
    fname : (str)
        name of the json materials file
    """

    fpath = os.path.join(fp, fname)

    material_dict = {
        "alias": alias,
        "Fickian": fickian,
        "Ead": Ead,
        "Do": Do,
        "Eas": Eas,
        "So": So,
        "Eap": Eap,
        "Po": Po,
    }

    with open(fpath) as f:
        data = json.load(f)
    data.update({name: material_dict})

    with open(fpath, "w") as f:
        json.dump(data, f, indent=4)


def quantile_df(file, q):
    """Calculate the quantile of each parameter at each location.

    Parameters
    ----------
    file : (str)
        Filepath to h5 results file containing timeseries and location data.
    q : (float)
        quantile to calculate

    Returns:
    --------
    res : (pd.DataFrame)
        dataframe containing location coordinates and quantile values of
        each parameter.
    """
    with Outputs(file, mode="r") as out:
        res = out["meta"][["latitude", "longitude"]]
        for key in out.attrs.keys():
            if key not in ["meta", "time_index"]:
                for i, cor in res.iterrows():
                    quantile = np.quantile(out[key, :, i], q=q, interpolation="linear")
                    res.loc[i, key] = quantile
    return res


def ts_gid_df(file, gid):
    """Extract the time series of each parameter for given location.

    Parameters
    ----------
    file : (str)
        Filepath to h5 results file containing timeseries and location data.
    gid : (int)
        geographical id of location

    Returns
    -------
    res : (pd.DataFrame)
        dataframe containing time series data for given location.
    """
    with Outputs(file, mode="r") as out:
        res = pd.DataFrame(index=out["time_index"])
        meta = out["meta"][["latitude", "longitude"]]
        for key in out.attrs.keys():
            if key not in ["meta", "time_index"]:
                res[key] = out[key, :, gid]
                res.gid = gid
                res.lat = meta.latitude[gid]
                res.lon = meta.longitude[gid]
    return res


def tilt_azimuth_scan(
    weather_df=None, meta=None, tilt_step=5, azimuth_step=5, func=Callable, **kwarg
):
    """Calculate minimum standoff distance for roof-mounted PV systems.

    Standoff calculated as a function of tilt and azimuth.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    tilt_step : integer
        Step in degrees of change in tilt angle of PV system between calculations.
        Will scan from 0 to 90 degrees.
    azimuth_step : integer
        Step in degrees of change in Azimuth angle of PV system relative to north.
        Will scan from 0 to 180 degrees.
    kwarg : dict
        All the keywords in a dictionary form that are needed to run the function.
    calc_function : string
        The name of the function that will be calculated.
    Returns
        standoff_series : 2-D array with each row consiting of tilt, azimuth, then
        standoff
    """
    total_count = (np.ceil(360 / azimuth_step) + 1) * (np.ceil(90 / tilt_step) + 1)
    tilt_azimuth_series = np.zeros((int(total_count), 3))
    count = 0
    azimuth = -azimuth_step
    while azimuth < 360:
        tilt = -tilt_step
        azimuth = azimuth + azimuth_step
        if azimuth > 360:
            azimuth = 360
        while tilt < 90:
            tilt = tilt + tilt_step
            if tilt > 90:
                tilt = 90
            tilt_azimuth_series[count][0] = tilt
            tilt_azimuth_series[count][1] = azimuth
            tilt_azimuth_series[count][2] = func(
                weather_df=weather_df, meta=meta, tilt=tilt, azimuth=azimuth, **kwarg
            )
            count = count + 1
            print(
                "\r", "%.1f" % (100 * count / total_count), "% complete", sep="", end=""
            )

    print("\r                     ", end="")
    print("\r", end="")
    return tilt_azimuth_series


def _meta_df_from_csv(file_paths: list[str]):
    """
    Create csv dataframe from list of files in string form, helper function.

    Also warns if d.irectory not functional yet.

    Parameters
    ----------
    file_paths : list[str]
        List of local weather csv files to strip metadata from.
        For example: download a collection of weather files from the NSRDB web viewer.

    Returns
    -------
    metadata_df : pandas.DataFrame
        Dataframe of stripped metadata from csv.
        Columns represent attribute names while rows represent a unique file.
    """
    # TODO: functionality
    # list[path] instead of just string
    # or a directory, just use csv from provided directory

    def read_meta(path):
        df = pd.read_csv(path, nrows=1)
        listed = df.to_dict(orient="list")
        stripped = {key: value[0] for key, value in listed.items()}

        return stripped

    metadata_df = pd.DataFrame()

    for i in range(len(file_paths)):
        metadata_df[i] = read_meta(file_paths[i])

    metadata_df = metadata_df.T

    # correct level of precision??
    conversions = {
        "Location ID": np.int32,
        "Latitude": np.double,
        "Longitude": np.double,
        "Time Zone": np.int8,
        "Elevation": np.int16,
        "Local Time Zone": np.int8,
    }

    metadata_df = metadata_df.astype(conversions)
    return metadata_df


def _weather_ds_from_csv(
    file_paths: list[str],
    year: int,
    # select year, should be able to provide single year, or list of years
):
    """Create a geospatial xarray dataset from local csv files, helper function."""
    #  ds = xr.open_dataset(
    #             fp,
    #             engine="h5netcdf",
    #             phony_dims="sort",
    #             chunks={"phony_dim_0": chunks[0], "phony_dim_1": chunks[1]},
    #             drop_variables=drop_variables,
    #             mask_and_scale=False,
    #             decode_cf=True,
    #         )

    # PROBLEM: all csv do not contain all years but these all appear to have 2004
    # when missing years, xarray will see mismatched coordinates and populate all these
    # values with nan this is wrong we are using tmy so we ignore the year as it
    # represents a typical meteorological year

    # Prepare a list to hold the DataFrames
    dataframes = []

    # Process each file
    for file_path in file_paths:
        # Extract GID from the filename
        header = pd.read_csv(file_path, nrows=1)
        gid = header["Location ID"][0]

        # Read the CSV, skipping rows to get to the relevant data
        df = pd.read_csv(file_path, skiprows=2)

        # Add GID and Time columns
        df["gid"] = gid

        df["time"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])

        # make allow this to take list of years
        df = df[df["time"].dt.year == year]

        # add generic approach, dont manually do this, could change based on user
        # selections

        # Select relevant columns and append to the list
        # df = df[['gid', 'time', 'GHI', 'Temperature', 'DHI', 'DNI', 'Surface Albedo',
        # 'Wind Direction', 'Wind Speed']]
        df = df[
            [
                "gid",
                "time",
                "GHI",
                "Temperature",
                "DHI",
                "DNI",
                "Surface Albedo",
                "Wind Speed",
            ]
        ]
        dataframes.append(df)

    # Combine all DataFrames into one
    combined_df = pd.concat(dataframes)

    # Convert the combined DataFrame to an xarray Dataset
    weather_ds = combined_df.set_index(["time", "gid"]).to_xarray()

    # combined_df = combined_df.set_index(['time', 'gid']).sort_index()
    # weather_ds = combined_df.set_index(['time', 'gid']).to_xarray()

    # GHI             (gid, time) int64 12kB 0 0 0 0 0 0 ... 507 439 393 238 54 20
    # Temperature     (gid, time) float64 12kB -12.6 -13.3 -13.6 ... -3.4 -4.6
    # DHI             (gid, time) int64 12kB 0 0 0 0 0 0 0 ... 56 113 94 129 54 20
    # DNI             (gid, time) int64 12kB 0 0 0 0 0 0 ... 1004 718 728 337 0 0
    # Surface Albedo  (gid, time) float64 12kB 0.8 0.8 0.8 0.8 ... 0.8 0.8 0.8 0.8
    # Wind Speed

    weather_ds = weather_ds.rename_vars(
        {
            "GHI": "ghi",
            "Temperature": "temp_air",
            "DHI": "dhi",
            "DNI": "dni",
            "Wind Speed": "wind_speed",
        }
    )

    return weather_ds


# not functional
def geospatial_from_csv(
    file_path: list[str],
    year: int,  # should be able to take a range of years
):
    """Create an xarray dataset contaning aeospatial and geospatial weather/meta data.

    Creates an xarray dataset contaning aeospatial weather data and a pandas dataframe
    containing geospatial metadata from a list of local csv files.

    Useful for importing data from NSRDB api viewer https://nsrdb.nrel.gov/data-viewer
    when downloaded locally as csv

    Parameters
    ----------
    file_path : list[str]
        List of absolute paths to csv files in string form.
    year : int
        Single year of data to use from local csv files.
    """
    weather_ds, meta_df = (
        _weather_ds_from_csv(file_path, year),
        _meta_df_from_csv(file_path),
    )

    # only want to keep meta from given file using GIDs from DS
    # gather included files' gids from xarray
    included_gids = weather_ds.coords["gid"].values

    # filter the metadate to only include gid values found above
    filtered_meta = meta_df[meta_df["Location ID"].isin(included_gids)]

    # reset the indecies of updated dataframe (might not be nessecary)
    filtered_meta = filtered_meta.reset_index(drop=True)

    # rename Location ID column to gid
    filtered_meta = filtered_meta.rename({"Location ID": "gid"}, axis="columns")

    return weather_ds, filtered_meta


def strip_normalize_tmy(df, start_time, end_time):
    """Normalize the DataFrame, extract data between start and end times.

    Dataframe is noramlized to start at 00:00 and the data between the
    specified start and end times is extracted. Data are then shifted back to the
    original indexes.

    Parameters
    ----------
    df : pd.Dataframe
        dataframe with a datetime index and tmy data
    start_time : datetime.datetime
        start time
    end_time : datetime.datetime
        end time

    Returns
    -------
    sub_results : pd.DataFrame
        extracted subset of tmy data
    """
    tz = df.index.tz
    start_time = start_time.replace(tzinfo=tz)
    end_time = end_time.replace(tzinfo=tz)

    initial_time = df.index[0]
    shifted_index = df.index - pd.DateOffset(
        hours=initial_time.hour,
        minutes=initial_time.minute,
        seconds=initial_time.second,
    )
    df.index = shifted_index

    mask = (df.index >= start_time) & (df.index <= end_time)
    sub_results = df.loc[mask]

    sub_results.index = sub_results.index + pd.DateOffset(
        hours=initial_time.hour,
        minutes=initial_time.minute,
        seconds=initial_time.second,
    )

    return sub_results


def new_id(collection):
    """Generate a 5 uppercase letter string unqiue from all keys in a dictionary.

    Parameters
    ----------
    Collection : dict, ordereddict
        dictionary with keys as strings

    Returns : str
    -------------
        Unique 5 letter string of uppercase characters.
    """
    if not isinstance(collection, (dict, OrderedDict)):
        raise TypeError(f"{collection.__name__} type {type(collection)} expected dict")

    def gen():
        return "".join(choices(ascii_uppercase, k=5))

    id = gen()
    while id in collection.keys():
        id = gen()

    return id


def restore_gids(
    original_meta_df: pd.DataFrame, analysis_result_ds: xr.Dataset
) -> xr.Dataset:
    """Restore gids to results dataset.

    For desired behavior output data must have
    identical ordering to input data, otherwise will fail silently by misassigning gids
    to lat-long coordinates in returned dataset.

    Parameters
    ----------
    original_meta_df : pd.DataFrame
        Metadata dataframe as returned by geospatial ``pvdeg.weather.get``
    analysis_result_ds : xr.Dataset
        geospatial result data as returned by ``pvdeg.geospatial.analysis``

    Returns
    -------
    restored_gids_ds : xr.Dataset
        dataset like ``analysis_result_ds`` with new datavariable, ``gid``
        holding the original gids of each result from the input metadata.
        Warning: if meta order is different than result ordering gids will
        be assigned incorrectly.
    """
    flattened = analysis_result_ds.stack(points=("latitude", "longitude"))

    gids = original_meta_df.index.values

    # Create a DataArray with the gids and assign it to the Dataset
    gids_da = xr.DataArray(gids, coords=[flattened["points"]], name="gid")

    # Unstack the DataArray to match the original dimensions of the Dataset
    gids_da = gids_da.unstack("points")

    restored_gids_ds = analysis_result_ds.assign(gid=gids_da)

    return restored_gids_ds


def _find_bbox_corners(coord_1=None, coord_2=None, coords=None):
    """Find min/max latitude and longitude.

       Find min and max latitude and longitude coordinates from two lists or a tall
       numpy array of the shape [[lat, long], ...]

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
    lats, longs : tuple(list)
        min and max latitude and longitudes. Minimum latitude at lats[0].
        Maximum latitude at lats[1]. Same pattern for longs.
    """
    if coord_1 is not None and coord_2 is not None:
        lats = [coord_1[0], coord_2[0]]
        longs = [coord_1[1], coord_2[1]]
    elif coords.any():
        lats = coords[:, 0]
        longs = coords[:, 1]

    min_lat, max_lat = np.min(lats), np.max(lats)
    min_long, max_long = np.min(longs), np.max(longs)

    lats = [min_lat, max_lat]
    longs = [min_long, max_long]

    return lats, longs


def _plot_bbox_corners(ax, coord_1=None, coord_2=None, coords=None):
    """Set matplotlib axis limits to the values from a bounding box.

    See Also:
    --------
    pvdeg.utilities._find_bbox_corners for more information
    """
    lats, longs = _find_bbox_corners(coord_1, coord_2, coords)

    ax.set_xlim([longs[0], longs[1]])
    ax.set_ylim([lats[0], lats[1]])
    return


def _add_cartopy_features(
    ax,
    features=[
        cfeature.BORDERS,
        cfeature.COASTLINE,
        cfeature.LAND,
        cfeature.OCEAN,
        cfeature.LAKES,
        cfeature.RIVERS,
    ],
):
    """Add cartopy features to an existing matplotlib.pyplot axis."""
    for i in features:
        if i == cfeature.BORDERS:
            ax.add_feature(i, linestyle=":")
        else:
            ax.add_feature(i)


def linear_normalize(array: np.ndarray) -> np.ndarray:
    """Normalize a non-negative input array."""
    return np.divide(
        np.subtract(array, np.min(array)),
        np.subtract(np.max(array), np.min(array)),
    )


def _calc_elevation_weights(
    elevations: np.array,
    coords: np.array,
    k_neighbors: int,
    method: str,
    normalization: str,
    kdtree,
) -> np.array:
    """Calculate elevation weights, utility function.

    Caluclate a weight for each point in a dataset to use for probabalistic
    downselection.

    Parameters
    ----------
    elevations : np.ndarray
        one dimensional numpy array of elevations at each gid in the metadata
    coords : np.ndarray
        tall 2d numpy array of lat-long pairs like [[lat, long], ...]
    k_neighbors : int
        number of neighbors to use in local elevation calculation at each point
    method : str, (default = 'mean')
        method to calculate elevation weights for each point.
        Options : `'mean'`, `'sum'`, `'median'`
    normalization : str, (default = 'linear')
        function to apply when normalizing weights. Logarithmic uses log_e/ln
        options : `'linear'`, `'log'`, '`exp'`, `'invert-linear'`
    kdtree : sklearn.neighbors.KDTree or str
        kdtree containing latitude-longitude pairs for quick lookups
        Generate using ``pvdeg.geospatial.meta_KDTree``. Can take a pickled
        kdtree as a path to the .pkl file.

    Returns
    -------
    gids : np.array
        1d numpy array of weights corresponding to each lat-long pair
        in coordinates and respectively in metadata.
    """
    weights = np.empty_like(elevations)

    for i, coord in enumerate(coords):
        indicies = kdtree.query(coord.reshape(1, -1), k=k_neighbors + 1)[1][
            0
        ]  # +1 to include current point
        delta_elevation = np.abs(elevations[indicies[1:]] - elevations[i])

        if method == "mean":
            delta = np.mean(delta_elevation)
        elif method == "sum":
            delta = np.sum(delta_elevation)
        elif method == "median":
            delta = np.median(delta_elevation)
        weights[i] = delta

    linear_weights = linear_normalize(weights)

    if normalization == "linear":
        return linear_weights

    if normalization == "invert-linear":
        return 1 - linear_weights

    elif normalization == "exp":
        return linear_normalize(np.exp(linear_weights))

    elif normalization == "log":
        # add 1 to shift the domain right so results of log will be positive
        # may be a better way, value wont be properly normalized between 0 and 1
        return linear_normalize(np.log(linear_weights + 1))

    raise ValueError(
        f"""
        normalization method: {normalization} does not exist.
        must be: "linear", "exp", "log"
        """
    )


def fix_metadata(meta):
    """Meta gid was appearing with ('lat' : {gid: lat}, 'long' : {gid: long}), ...

    remove each subdict and replace with value for each key.

    Parameters:
    -----------
    meta : dict
        dictionary of metadata with key : dict pairs

    Returns
    fixed_meta : dict
        dictionary of metadata with key : value pairs
    """
    fixed_metadata = {key: list(subdict.values())[0] for key, subdict in meta.items()}
    return fixed_metadata


# we want this to only exist for things that can be run on kestrel
# moving away from hpc tools so this may not be useful in the future
def nrel_kestrel_check():
    """Check if the user is on Kestrel HPC environment.

    Passes silently or raises a
    ConnectionError if not running on Kestrel. This will fail on AWS.

    Returns
    -------
    None

    See Also
    --------
    NREL HPC : https://www.nrel.gov/hpc/
    Kestrel Documentation : https://nrel.github.io/HPC/Documentation/
    """
    kestrel_hostname = "kestrel.hpc.nrel.gov"

    host = run(args=["hostname", "-f"], shell=False, capture_output=True, text=True)
    device_domain = ".".join(host.stdout.split(".")[-4:])[:-1]

    if kestrel_hostname != device_domain:
        raise ConnectionError(
            f"""
            connected to {device_domain} not a node of {kestrel_hostname}")
            """
        )


def remove_scenario_filetrees(fp, pattern="pvd_job_*"):
    """Move `cwd` to fp and remove all scenario file trees from fp directory.

    Permanently deletes all scenario file trees. USE WITH CAUTION.

    Parameters:
    -----------
    fp : string
        file path to directory where all scenario files should be removed
    pattern : str
        pattern to search for using glob. Default value of `pvd_job_` is
        equvilent to `pvd_job_*` in bash.

    Returns:
    --------
    None
    """
    import shutil
    import glob

    os.chdir(fp)

    items = glob.glob(pattern)
    for item in items:
        if os.path.isdir(item):
            shutil.rmtree(item)


def _update_pipeline_task(task):
    """
    Convert qualified name to callable function reference, mantain odict items ordering.

    Use to restore scenario from json.
    """
    from importlib import import_module

    module_name, func_name = task["qualified_function"].rsplit(".", 1)
    params = task["params"]  # need to do this to maintain ordering
    module = import_module(module_name)
    func = getattr(module, func_name)
    task["job"] = func
    del task["qualified_function"]
    del task["params"]  # maintain ordering,
    task["params"] = params


def compare_templates(
    ds1: xr.Dataset, ds2: xr.Dataset, atol=1e-10, consider_nan_equal=True
) -> bool:
    """Compare loaded datasets with "empty-like" values."""
    if ds1.dims != ds2.dims:
        return False

    if set(ds1.coords.keys()) != set(ds2.coords.keys()):
        return False

    for coord in ds1.coords:
        if ds1.coords[coord].dtype.kind in {"i", "f"}:
            if not np.allclose(ds1.coords[coord], ds2.coords[coord], atol=atol):
                return False
        elif ds1.coords[coord].dtype.kind == "M":
            if not np.array_equal(ds1.coords[coord], ds2.coords[coord]):
                return False
        else:
            if not np.array_equal(ds1.coords[coord], ds2.coords[coord]):
                return False

    if set(ds1.data_vars.keys()) != set(ds2.data_vars.keys()):
        return False

    for dim in ds1.dims:
        if not ds1.indexes[dim].equals(ds2.indexes[dim]):
            return False

    return True


def add_time_columns_tmy(weather_df, coerce_year=1979):
    """Add time columns to a tmy weather dataframe.

    Parameters
    ----------
    weather_df: pd.DataFrame
        tmy weather dataframe containing 8760 rows.
    coerce_year: int
        year to set the dataframe to.

    Returns
    -------
    weather_df: pd.DataFrame
        dataframe with columns added new columns will be

        ``'Year', 'Month', 'Day', 'Hour', 'Minute'``
    """
    weather_df = weather_df.reset_index(drop=True)

    if len(weather_df) == 8760:
        freq = "h"
    elif len(weather_df) == 17520:
        freq = "30min"
    else:
        raise ValueError("weather df must be in 1 hour or 30 minute intervals")

    date_range = pd.date_range(
        start=f"{coerce_year}-01-01 00:00:00",  # noqa: E231
        end=f"{coerce_year}-12-31 23:45:00",  # noqa: E231
        freq=freq,
    )

    df = pd.DataFrame(
        {
            "Year": date_range.year,
            "Month": date_range.month,
            "Day": date_range.day,
            "Hour": date_range.hour,
            "Minute": date_range.minute,
        }
    )

    weather_df = pd.concat([weather_df, df], axis=1)
    return weather_df


def merge_sparse(files: list[str]) -> xr.Dataset:
    """Merge an arbitrary number of geospatial analysis results.

    Creates monotonically
    increasing indicies.

    Uses `engine='h5netcdf'` for reliability, use h5netcdf to save your results to
    netcdf files.

    Parameters
    -----------
    files: list[str]
        A list of strings representing filepaths to netcdf (.nc) files.
        Each file must have the same coordinates, `['latitude','longitude']` and
        identical datavariables.

    Returns
    -------
    merged_ds: xr.Dataset
        Dataset (in memory) with `coordinates = ['latitude','longitude']` and
        datavariables matching files in filepaths list.
    """
    datasets = [xr.open_dataset(fp, engine="h5netcdf").compute() for fp in files]

    latitudes = np.concatenate([ds.latitude.values for ds in datasets])
    longitudes = np.concatenate([ds.longitude.values for ds in datasets])
    unique_latitudes = np.sort(np.unique(latitudes))
    unique_longitudes = np.sort(np.unique(longitudes))

    data_vars = datasets[0].data_vars

    merged_ds = xr.Dataset(
        {
            var: (
                ["latitude", "longitude"],
                np.full((len(unique_latitudes), len(unique_longitudes)), np.nan),
            )
            for var in data_vars
        },
        coords={"latitude": unique_latitudes, "longitude": unique_longitudes},
    )

    for ds in datasets:
        lat_inds = np.searchsorted(unique_latitudes, ds.latitude.values)
        lon_inds = np.searchsorted(unique_longitudes, ds.longitude.values)

        for var in ds.data_vars:
            merged_ds[var].values[np.ix_(lat_inds, lon_inds)] = ds[var].values

    return merged_ds


def display_json(
    pvdeg_file: str = None,
    fp: str = None,
) -> None:
    """Interactively view a 2 level JSON file in a JupyterNotebook.

    Parameters:
    ------------
    pvdeg_file: str
        keyword for material json file in `pvdeg/data`. Options:
        >>> "AApermeation", "H2Opermeation", "O2permeation", "DegradationDatabase"
    fp: str
        file path to material parameters json with same schema as material parameters
        json files in `pvdeg/data`.  `pvdeg_file` will override `fp` if both are
        provided.
    """
    from IPython.display import display, HTML

    if pvdeg_file:
        try:
            fp = pvdeg_datafiles[pvdeg_file]
        except KeyError:
            raise KeyError(
                f"{pvdeg_file} is not in pvdeg/data. Options are "
                f"{pvdeg_datafiles.keys()}"
            )

    with open(fp, "r") as file:
        data = json.load(file)

    def json_to_html(data):
        json_str = json.dumps(data, indent=2)
        for key in data.keys():
            json_str = json_str.replace(
                f'"{key}":',  # noqa: E702,E231, E501
                f'<span style="color: plum;">"{key}":</span>',  # noqa: E702,E231, E501
            )
        indented_html = "<br>".join([" " * 4 + line for line in json_str.splitlines()])
        return f'<pre style="color: white; background-color: black; padding: 10px; border-radius: 5px;">{indented_html}</pre>'  # noqa: E702,E231, E501

    html = f'<h2 style="color: white;">JSON Output at fp: {fp}</h2><div>'  # noqa
    for key, value in data.items():
        html += (
            f"<div>"
            f'<strong style="color: white;">{key}:</strong> '  # noqa
            f"<span onclick=\"this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'\" style=\"cursor: pointer; color: white;\">&#9660;</span>"  # noqa: E702,E231,E501,W505
            f'<div style="display: none;">{json_to_html(value)}</div>'  # noqa
            f"</div>"
        )
    html += "</div>"

    # Display the HTML
    display(HTML(html))
    print(html)


def search_json(
    pvdeg_file: str = None,
    fp: str = None,
    name_or_alias: str = None,
) -> str:
    """Search through 2 level JSON.

       Search through 2 level JSON with arbitrary key names for subkeys with matching
       attributes of name or alias.

    Parameters
    ----------
    pvdeg_file: str
        keyword for material json file in `pvdeg/data`. Options:
        >>> "AApermeation", "H2Opermeation", "O2permeation"
    fp: str
        file path to material parameters json with same schema as material parameters
        json files in `pvdeg/data`. `pvdeg_file` will override `fp` if both are
        provided.
    name_or_alias: str
        searches for matching subkey value in either `name` or `alias` attributes.
        Exits on the first matching instance.

    Returns
    -------
    jsonkey: str
        arbitrary key from json that owns the matching subattribute of `name` or
        `alias`.
    """
    if pvdeg_file:
        try:
            fp = pvdeg_datafiles[pvdeg_file]
        except KeyError:
            raise KeyError(
                rf"{pvdeg_file} is not exist in pvdeg/data. Options are: "
                " {pvdeg_datafiles.keys()}"
            )

    with open(fp, "r") as file:
        data = json.load(file)

    for key, subdict in data.items():
        if "name" in subdict and "alias" in subdict:
            if subdict["name"] == name_or_alias or subdict["alias"] == name_or_alias:
                return key

    raise ValueError(rf"name_or_alias: {name_or_alias} not in JSON at {fp}")


def read_material(
    pvdeg_file: str = None,
    fp: str = None,
    key: str = None,
    encoding: str = "utf-8",
) -> dict:
    """Read material dictionary from a `pvdeg/data` file or JSON file path.

    Parameters
    ----------
    pvdeg_file: str
        keyword for material json file in `pvdeg/data`. Options:
        >>> "AApermeation", "H2Opermeation", "O2permeation"
    fp: str
        file path to material parameters json with same schema as material parameters
        json files in `pvdeg/data`. `pvdeg_file` will override `fp` if both are
        provided.
    key: str
        key corresponding to specific material in the file. In the pvdeg files these
        have arbitrary names. Inspect the files or use `display_json` or `search_json`
        to identify the key for desired material.
    encoding : (str)
        encoding to use when reading the JSON file, default is "utf-8"

    Returns
    -------
    material: dict
        dictionary of material parameters from the selected file at the index key.
    """
    if pvdeg_file:
        try:
            fp = pvdeg_datafiles[pvdeg_file]
        except KeyError:
            raise KeyError(
                f"{pvdeg_file} is not in pvdeg/data. Options are: "
                " {pvdeg_datafiles.keys()}"
            )

    with open(fp, "r", encoding=encoding) as file:
        data = json.load(file)

    material_dict = data[key]
    return material_dict


def read_material_property(
    pvdeg_file: str = None,
    filepath: str = None,
    key: str = None,
    parameters: list[str] = None,
) -> dict:
    """Read material parameters from a `pvdeg/data` file or JSON file path.

    Parameters
    ----------
    pvdeg_file: str
        keyword for material json file in `pvdeg/data`. Options:
        >>> "AApermeation", "H2Opermeation", "O2permeation"
    filepath: str
        file path to material parameters json with same schema as material parameters
        json files in `pvdeg/data`. `pvdeg_file` will override `fp` if both are
        provided.
    key: str
        key corresponding to specific material in the file. In the pvdeg files these
        have arbitrary names. Inspect the files or use `display_json` or `search_json`
        to identify the key for desired material.

    Returns
    -------
    parameters: dict
        dictionary of material parameters from the selected file at the index key.
    """
    material_dict = read_material(
        pvdeg_file=pvdeg_file,
        fp=filepath,
        key=key,
    )

    if parameters:
        material_dict = {
            k: (
                material_dict[k]["value"]
                if k in material_dict and isinstance(material_dict[k], dict)
                else material_dict[k] if k in material_dict else None
            )
            for k in parameters
        }
    else:
        material_dict = {
            k: v["value"] if isinstance(v, dict) else v
            for k, v in material_dict.items()
        }
    return material_dict
