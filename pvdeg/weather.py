"""
Collection of classes and functions to obtain spectral parameters.
"""

from pvlib import iotools
import os
import glob
import pandas as pd
from rex import NSRDBX, Outputs
from pvdeg import humidity

def get(database, id, **kwargs):
    """
    Load weather data directly from  NSRDB or through any other PVLIB i/o 
    tools function

    Parameters:
    -----------
    database : (str)
        'NSRDB' or 'PVGIS'
    id : (int or tuple)
        If NSRDB, id is the gid for the desired location
        If PVGIS, id is a tuple of (latitude, longitude) for the desired location
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
    if type(id) is tuple:
        location = id
        gid = None
        lat = location[0]
        lon = location[1]
    elif type(id) is int:
        gid = id
        location = None
    else:
        raise TypeError(
            'Project points needs to be either location tuple (latitude, longitude), or gid integer.')

    #TODO: decide wether to follow NSRDB or pvlib conventions...
    # e.g. temp_air vs. air_temperature
    # "map variables" will guarantee PVLIB conventions (automatic in coming update) which is "temp_air"
    if database == 'NSRDB':
        weather_df, meta = get_NSRDB(gid=gid, location=location, **kwargs)
    elif database == 'PVGIS':
        weather_df, _, _, meta = iotools.get_pvgis_tmy(latitude=lat, longitude=lon,
                                                             map_variables=True, **kwargs)
    elif database == 'PSM3':
        weather_df, meta = iotools.get_psm3(latitude=lat, longitude=lon, **kwargs)
    else:
        raise NameError('Weather database not found.')

    if 'relative_humidity' not in weather_df.columns:
        print('Column "relative_humidity" not found in DataFrame. Calculating...')
        weather_df = humidity._ambient(weather_df)
    
    return weather_df, meta


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

    supported = ['psm3','tmy3','epw','h5']
    file_type = file_type.upper()
    
    if file_type in ['PSM3','PSM']:
        weather_df, meta = iotools.read_psm3(filename=file_in, map_variables=True)
    elif file_type in ['TMY3','TMY']:
        weather_df, meta = iotools.read_tmy3(filename=file_in, map_variables=True)
    elif file_type == 'EPW':
        weather_df, meta = iotools.read_epw(filename=file_in)
    elif file_type == 'H5':
        weather_df, meta = read_h5(file=file_in, **kwargs)
    else:
        print(f'File-Type not recognized. supported types:\n{supported}')
    
    if not isinstance(meta, dict):
        meta = meta.to_dict()

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

    fp = os.path.join(os.path.dirname(__file__), file)

    with Outputs(fp, mode='r') as f:   
        meta = f.meta.loc[gid]
        index = f.time_index
        dattr = f.attrs

    #TODO: put into utilities
    if attributes == None:
        attributes = list(dattr.keys())
        try:
            attributes.remove('meta')
            attributes.remove('tmy_year_short')
        except ValueError:
            pass

    weather_df = pd.DataFrame(index=index, columns=attributes)
    for dset in attributes:
        with Outputs(fp, mode='r') as f:   
            weather_df[dset] = f[dset, :, gid]

    return weather_df, meta.to_dict()


def get_NSRDB_fnames(satellite, names, NREL_HPC = False, **_):
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

    sat_map = {'GOES' : 'full_disc',
               'METEOSAT' : 'meteosat',
               'Himawari' : 'himawari',
               'SUNY' : 'india',
               'CONUS' : 'conus',
               'Americas' : 'current'}

    if NREL_HPC:
        hpc_fp = '/datasets/NSRDB/'
        hsds = False
    else:
        hpc_fp = '/nrel/nsrdb/'
        hsds = True
        
    if type(names) == int:
        nsrdb_fp = os.path.join(hpc_fp, sat_map[satellite], '*_{}.h5'.format(names))
        nsrdb_fnames = glob.glob(nsrdb_fp)
    else:
        nsrdb_fp = os.path.join(hpc_fp, sat_map[satellite], '*_{}*.h5'.format(names.lower()))
        nsrdb_fnames = glob.glob(nsrdb_fp)
        
    if len(nsrdb_fnames) == 0:
        raise FileNotFoundError(
            "Couldn't find NSRDB input files! \nSearched for: '{}'".format(nsrdb_fp))
    
    return nsrdb_fnames, hsds


def get_NSRDB(satellite, names, NREL_HPC, gid=None, location=None, attributes=None, **_):
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

    DSET_MAP = {'air_temperature' : 'temp_air',
                'Relative Humidity' : 'relative_humidity'}

    META_MAP = {'elevation' : 'altitude'}

    nsrdb_fnames, hsds = get_NSRDB_fnames(satellite, names, NREL_HPC)

    dattr = {}
    for i, file in enumerate(nsrdb_fnames):
        with NSRDBX(file, hsds=hsds) as f:
            if i == 0:
                if gid == None: #TODO: add exception handling
                    gid = f.lat_lon_gid(location)
                meta = f['meta', gid].iloc[0]
                index = f.time_index

            lattr = f.datasets
            for attr in lattr:
                dattr[attr] = file

    if attributes == None:
        attributes = list(dattr.keys())
        try:
            attributes.remove('meta')
            attributes.remove('tmy_year_short')
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