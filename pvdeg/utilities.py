import os
import json
import pandas as pd
import numpy as np
from rex import NSRDBX, Outputs
from pvdeg import DATA_LIBRARY


def gid_downsampling(meta, n):
    """
    Downsample the NSRDB GID grid by a factor of n

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

    lon_sub = sorted(meta["longitude"].unique())[0 : -1 : max(1, 2 * n)]
    lat_sub = sorted(meta["latitude"].unique())[0 : -1 : max(1, 2 * n)]

    gids_sub = meta[
        (meta["longitude"].isin(lon_sub)) & (meta["latitude"].isin(lat_sub))
    ].index.values

    meta_sub = meta.loc[gids_sub]

    return meta_sub, gids_sub


def meta_as_dict(rec):
    """
    Turn a numpy recarray record into a dict.

    Parameters:
    -----------
    rec : (np.recarray)
        numpy structured array with labels as dtypes

    Returns:
    --------
     : (dict)
        dictionary of numpy structured array
    """

    return {name: rec[name].item() for name in rec.dtype.names}


def get_kinetics(name=None, fname="kinetic_parameters.json"):
    """
    Returns a list of LETID/B-O LID kinetic parameters from kinetic_parameters.json

    Parameters:
    -----------
    name : str
        unique name of kinetic parameter set. If None, returns a list of the possible options.

    Returns:
    --------
    parameter_dict : (dict)
        dictionary of kinetic parameters
    """
    fpath = os.path.join(DATA_LIBRARY, fname)

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
    """
    Generate a .CSV file containing the GIDs for the spatial test range.
    The .CSV file will be saved to the working directory

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

    Returns:
    -----------
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
    """
    Returns the full name of a state based on two-letter state code

    Parameters:
    -----------
    id : (str)
        two letter state code (example: CO, AZ, MD)

    Returns:
    -----------
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


def convert_tmy(file_in, file_out="h5_from_tmy.h5"):
    """
    Read a older TMY-like weather file and convert to h5 for use in pvdeg

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

    save_cols = {
        "DNI": "dni",
        "DHI": "dhi",
        "GHI": "ghi",
        "DryBulb": "temp_air",
        "DewPoint": "dew_point",
        "RHum": "relative_humidity",
        "Wspd": "wind_speed",
        "Alb": "albedo",
    }

    df_new = src_data[save_cols.keys()].copy()
    df_new.columns = save_cols.values()
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


def _read_material(name, fname="materials.json"):
    """
    read a material from materials.json and return the parameter dictionary

    Parameters:
    -----------
    name : (str)
        unique name of material

    Returns:
    --------
    mat_dict : (dict)
        dictionary of material parameters
    """
    # TODO: test then delete commented code
    # root = os.path.realpath(__file__)
    # root = root.split(r'/')[:-1]
    # file = os.path.join('/', *root, 'data', 'materials.json')
    fpath = os.path.join(DATA_LIBRARY, fname)
    with open(fpath) as f:
        data = json.load(f)

    if name is None:
        material_list = data.keys()
        return [*material_list]

    mat_dict = data[name]
    return mat_dict


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
    fname="materials.json",
):
    """
    Add a new material to the materials.json database. Check the parameters for specific units.
    If material already exists, parameters will be updated.

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
    """

    # TODO: test then delete commented code
    # root = os.path.realpath(__file__)
    # root = root.split(r'/')[:-1]
    # OUT_FILE = os.path.join('/', *root, 'data', 'materials.json')
    fpath = os.path.join(DATA_LIBRARY, fname)

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
    """
    Calculate the quantile of each parameter at each location.

    Parameters:
    -----------
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
    """
    Extract the time series of each parameter for given location.

    Parameters:
    -----------
    file : (str)
        Filepath to h5 results file containing timeseries and location data.
    gid : (int)
        geographical id of location

    Returns:
    --------
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
