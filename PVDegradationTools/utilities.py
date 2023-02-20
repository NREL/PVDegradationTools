import os
import pandas as pd
import numpy as np
from rex import NSRDBX, Outputs

def write_gids(nsrdb_fp, region='Colorado', region_col='state', out_fn='gids'):
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
    out_fd : (str, default = "gids")
        Name of data column you want to retrieve. Generally, this should be "gids"

    Returns:
    -----------
    None
    """

    with NSRDBX(nsrdb_fp, hsds=False) as f:
        gids = f.region_gids(region=region, region_col=region_col)   

    df_gids = pd.DataFrame(gids, columns=['gid'])
    df_gids.to_csv('{}.csv'.format(out_fn), index=False)


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
    state_dict = {'AK': 'Alaska',
            'AL': 'Alabama',
            'AR': 'Arkansas',
            'AS': 'American Samoa',
            'AZ': 'Arizona',
            'CA': 'California',
            'CO': 'Colorado',
            'CT': 'Connecticut',
            'DC': 'District of Columbia',
            'DE': 'Delaware',
            'FL': 'Florida',
            'GA': 'Georgia',
            'GU': 'Guam',
            'HI': 'Hawaii',
            'IA': 'Iowa',
            'ID': 'Idaho',
            'IL': 'Illinois',
            'IN': 'Indiana',
            'KS': 'Kansas',
            'KY': 'Kentucky',
            'LA': 'Louisiana',
            'MA': 'Massachusetts',
            'MD': 'Maryland',
            'ME': 'Maine',
            'MI': 'Michigan',
            'MN': 'Minnesota',
            'MO': 'Missouri',
            'MP': 'Northern Mariana Islands',
            'MS': 'Mississippi',
            'MT': 'Montana',
            'NA': 'National',
            'NC': 'North Carolina',
            'ND': 'North Dakota',
            'NE': 'Nebraska',
            'NH': 'New Hampshire',
            'NJ': 'New Jersey',
            'NM': 'New Mexico',
            'NV': 'Nevada',
            'NY': 'New York',
            'OH': 'Ohio',
            'OK': 'Oklahoma',
            'OR': 'Oregon',
            'PA': 'Pennsylvania',
            'PR': 'Puerto Rico',
            'RI': 'Rhode Island',
            'SC': 'South Carolina',
            'SD': 'South Dakota',
            'TN': 'Tennessee',
            'TX': 'Texas',
            'UT': 'Utah',
            'VA': 'Virginia',
            'VI': 'Virgin Islands',
            'VT': 'Vermont',
            'WA': 'Washington',
            'WI': 'Wisconsin',
            'WV': 'West Virginia',
            'WY': 'Wyoming'}
    state_name = state_dict[id]
    return state_name
    
def convert_tmy(file_in, file_out='h5_from_tmy.h5'):
    '''
    Read a older TMY-like weather file and convert to h5 for use in PVD

    TODO: figure out scale_facator and np.int32 for smaller file
          expand for international locations?

    Parameters:
    -----------
    file_in : (str, path_obj)
        full file path to existing weather file
    file_out : (str, path_obj)
        full file path and name of file to create.
    '''
    from pvlib import iotools

    src_data, src_meta = iotools.tmy.read_tmy3(file_in, coerce_year=2023)

    save_cols = {'DNI':'dni',
            'DHI':'dhi',
            'GHI':'ghi',
            'DryBulb':'air_temperature',
            'DewPoint':'dew_point',
            'RHum':'relative_humidity',
            'Wspd':'wind_speed',
            'Alb':'albedo'}

    df_new = src_data[save_cols.keys()].copy()
    df_new.columns = save_cols.values()
    time_index = df_new.index

    meta = {'latitude':[src_meta['latitude']],
            'longitude':[src_meta['longitude']],
            'elevation':[src_meta['altitude']],
            'timezone':[src_meta['TZ']],
            'country':['United States'],
            'state':[_get_state(src_meta['State'])]}
    meta = pd.DataFrame(meta)

    with Outputs(file_out, 'w') as f:
        f.meta = meta
        f.time_index = time_index
    
    for col in df_new.columns:
        Outputs.add_dataset(h5_file=file_out, dset_name=col,
                            dset_data=df_new[col].values,
                            attrs={'scale_factor':100},
                            dtype=np.int64)

def get_poa_irradiance(
    nsrdb_fp, 
    gid,
    solar_position,
    tilt=None, 
    azimuth=180, 
    sky_model='isotropic'):

    """
    Calculate plane-of-array (POA) irradiance using pvlib based on weather data from the 
    National Solar Radiation Database (NSRDB) for a given location (gid).

    TODO: only return 'poa_global', its the only one we seem to use
    
    Parameters
    ----------
    nsrdb_fp : str
        The file path to the NSRDB file.
    gid : int
        The geographical location ID in the NSRDB file.
    tilt : float, optional
        The tilt angle of the PV panels in degrees, if None, the latitude of the 
        location is used.
    azimuth : float, optional
        The azimuth angle of the PV panels in degrees, 180 by default - facing south.
    sky_model : str, optional
        The pvlib sky model to use, 'isotropic' by default.
    
    Returns
    -------
    poa : pandas.DataFrame
         Contains keys/columns 'poa_global', 'poa_direct', 'poa_diffuse', 
         'poa_sky_diffuse', 'poa_ground_diffuse'.
    """
    from pvlib.irradiance import get_total_irradiance
    
    with NSRDBX(nsrdb_fp, hsds=False) as f:
        meta = f.meta.loc[gid]
        dni = f.get_gid_ts('dni', gid)
        ghi = f.get_gid_ts('ghi', gid)
        dhi = f.get_gid_ts('dhi', gid)

    #TODO: change for handling HSAT tracking passed or requested
    if tilt is None:
        tilt = float(meta['latitude'])

    poa = get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        model=sky_model)

    return poa

def get_module_temperature(
    nsrdb_fp, 
    gid,
    poa,
    temp_model='sapm', 
    conf='open_rack_glass_polymer'):

    """
    Calculate module temperature based on weather data from the National Solar Radiation 
    Database (NSRDB) for a given location (gid).

    TODO: change input "poa" to accept only "poa_global" instead of the entire dataframe
    
    Parameters
    ----------
    nsrdb_file : str
        The file path to the NSRDB file.
    gid : int
        The geographical location ID in the NSRDB file.
    poa : pandas.DataFrame
         Contains keys/columns 'poa_global', 'poa_direct', 'poa_diffuse', 
         'poa_sky_diffuse', 'poa_ground_diffuse'.
    temp_model : str, optional
        The temperature model to use, 'sapm' from pvlib by default.
    conf : str, optional
        The configuration of the PV module architecture and mounting configuration.
        Options: 'open_rack_glass_polymer' (default), 'open_rack_glass_glass',
        'close_mount_glass_glass', 'insulated_back_glass_polymer'
    
    Returns
    -------
    module_temperature : pandas.DataFrame
        The module temperature in degrees Celsius at each time step.
    """

    from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS, sapm_module, sapm_cell
    
    with NSRDBX(nsrdb_fp, hsds=False) as f:
        air_temperature = f.get_gid_ts('air_temperature', gid)
        wind_speed = f.get_gid_ts('wind_speed', gid)

    parameters = TEMPERATURE_MODEL_PARAMETERS[temp_model][conf]
    module_temperature = sapm_module(
        poa_global=poa['poa_global'], 
        temp_air=air_temperature, 
        wind_speed=wind_speed,
        a=parameters['a'],
        b=parameters['b'])

    cell_temperature = sapm_cell(poa_global=poa['poa_global'],
                                 temp_air=air_temperature,
                                 wind_speed=wind_speed,
                                 **parameters)
    
    return {'temp_module':module_temperature,
            'temp_cell':cell_temperature}
