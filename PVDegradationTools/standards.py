"""
Collection of classes and functions for standard development.
"""

import numpy as np
import pandas as pd
import pvlib
from rex import NSRDBX
from rex import Outputs
from pathlib import Path
from random import random
from concurrent.futures import ProcessPoolExecutor, as_completed
#from gaps import ProjectPoints

from . import utilities

#TODO: move into 'spectral.py'
def get_solar_position(
    weather_df, 
    meta):

    """
    Calculate solar position using pvlib based on weather data from the 
    National Solar Radiation Database (NSRDB) for a given location (gid).
    
    Parameters
    ----------
    weather_df : pandas.DataFrame
        Weather data for given location.
    meta : pandas.Series
        Meta data of location.
    
    Returns
    -------
    solar_position : pandas.DataFrame
        Solar position like zenith and azimuth.
    """

    # location = pvlib.location.Location(
    #     latitude=meta['latitude'],
    #     longitude=meta['longitude'],
    #     altitude=meta['elevation'])
    
    #TODO: check if timeshift is necessary
    #times = weather_df.index
    #solar_position = location.get_solarposition(times)
    solar_position = pvlib.solarposition.get_solarposition(
        time = weather_df.index, 
        latitude = meta['latitude'], 
        longitude = meta['longitude'],
        altitude = meta['elevation'])

    return solar_position

#TODO: move into 'spectral.py'
def get_poa_irradiance(
    weather_df, 
    meta,
    solar_position,
    tilt=None, 
    azimuth=180, 
    sky_model='isotropic'):

    """
    Calculate plane-of-array (POA) irradiance using pvlib based on weather data from the 
    National Solar Radiation Database (NSRDB) for a given location (gid).
    
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

    #TODO: change for handling HSAT tracking passed or requested
    if tilt is None:
        tilt = float(meta['latitude'])

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=weather_df['dni'],
        ghi=weather_df['ghi'],
        dhi=weather_df['dhi'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        model=sky_model)

    return poa

#TODO: move into 'temperature.py'
def get_module_temperature(
    weather_df, 
    poa,
    temp_model='sapm', 
    conf='open_rack_glass_polymer',
    wind_speed_factor=1):

    """
    Calculate module temperature based on weather data from the National Solar Radiation 
    Database (NSRDB) for a given location (gid).
    
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
    parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[temp_model][conf]
    module_temperature = pvlib.temperature.sapm_module(
        poa_global=poa['poa_global'], 
        temp_air=weather_df['air_temperature'], 
        wind_speed=weather_df['wind_speed']*wind_speed_factor,
        a=parameters['a'],
        b=parameters['b'])

    return module_temperature

def get_eff_gap(T_0, T_inf, level=1, x_0=6.1):
    '''
    Calculate an ideal installation distance for roof top mounded PV systems.
    
    Parameters
    ----------
    level : int, optional
        Options 0, or 1. Level 1 or Level 0 module (in IEC TS 63216) define 
        the testing regime for the module; the boundaries are defined 
        internally, to use a level 0 module is the boundary is less than 
        70, and for Level 1 is less than 80. Above 80 Level 2 testing 
        regime is required.
    x0 : float, optional
        Thermal decay constant (cm), [Kempe, PVSC Proceedings 2023]


    Returns
    -------
    x : float
        Recommended installation distance in centimeter per IEC TS 63126.
        Effective gap "x" for the lower limit for Level 1 or Level 0 modules (IEC TS 63216)

    References
    ----------
    M. Kempe, et al. Close Roof Mounted System Temperature Estimation for Compliance 
    to IEC TS 63126, PVSC Proceedings 2023
    '''

    if level == 1:
        T98 = 70
    if level == 2:
        T98 = 80

    T98_0 = T_0.quantile(q=0.98, interpolation='linear')
    T98_inf = T_inf.quantile(q=0.98, interpolation='linear')

    x = -x_0 * np.log(1-(T98_0-T98)/(T98_0-T98_inf))

    return x, T98_0, T98_inf


def calc_standoff(
    weather_df,
    meta,
    tilt=None,
    azimuth=180,
    sky_model='isotropic',
    temp_model='sapm',
    module_type='glass_polymer', # self.module
    level=0,
    x_0=6.1,
    wind_speed_factor=1):

    """
    modeling chain example   #TODO: write docstring
    """

    if module_type == 'glass_polymer':
        conf_0 = 'insulated_back_glass_polymer'
        conf_inf = 'open_rack_glass_polymer'
    elif module_type == 'glass_glass':
        conf_0 = 'close_mount_glass_glass'
        conf_inf = 'open_rack_glass_glass'
  
    solar_position = get_solar_position(weather_df, meta)
    poa = get_poa_irradiance(weather_df, meta, solar_position, tilt, azimuth, sky_model)
    T_0 = get_module_temperature(weather_df, poa, temp_model, conf_0, wind_speed_factor)
    T_inf = get_module_temperature(weather_df, poa, temp_model, conf_inf, wind_speed_factor)
    x, T98_0, T98_inf = get_eff_gap(T_0, T_inf, level, x_0)

    return {'x':x, 'T98_0':T98_0, 'T98_inf':T98_inf}


def run_calc_standoff(
    project_points, 
    out_dir, 
    tag,
    weather_db,
    weather_satellite,
    weather_names,
    max_workers=None,
    tilt=None,
    azimuth=180,
    sky_model='isotropic',
    temp_model='sapm',
    module_type='glass_polymer',
    level=0,
    x_0=6.1,
    wind_speed_factor=1
):

    """
    parallelization utilizing gaps     #TODO: write docstring
    """

    #inputs
    weather_arg = {}
    weather_arg['satellite'] = weather_satellite
    weather_arg['names'] = weather_names
    weather_arg['NREL_HPC'] = True  #TODO: add argument or auto detect
    weather_arg['attributes'] = [
        'air_temperature', 
        'wind_speed', 
        'dhi', 'ghi', 
        'dni','relative_humidity'
        ]

    all_fields = ['x', 'T98_0', 'T98_inf']

    out_fp = Path(out_dir) / f"out_standoff{tag}.h5"
    shapes = {n : (len(project_points), ) for n in all_fields}
    attrs = {'x' : {'units': 'cm'},
             'T98_0' : {'units': 'Celsius'},
             'T98_inf' : {'units': 'Celsius'}}
    chunks = {n : None for n in all_fields}
    dtypes = {n : "float32" for n in all_fields}

    #TODO: is there a better way to add the meta data?
    nsrdb_fnames, hsds  = utilities.get_NSRDB_fnames(
        weather_arg['satellite'], 
        weather_arg['names'], 
        weather_arg['NREL_HPC'])
    
    with NSRDBX(nsrdb_fnames[0], hsds=hsds) as f:
        meta = f.meta[f.meta.index.isin(project_points.gids)]
 
    Outputs.init_h5(
        out_fp,
        all_fields,
        shapes,
        attrs,
        chunks,
        dtypes,
        meta=meta.reset_index()
    )

    future_to_point = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for point in project_points:
            gid = int(point.gid)
            weather_df, meta = utilities.get_weather(
                database = weather_db, 
                id = gid, 
                **weather_arg)
            future = executor.submit(
                calc_standoff,
                weather_df, 
                meta,
                tilt, 
                azimuth, 
                sky_model,
                temp_model, 
                module_type, 
                level, 
                x_0,
                wind_speed_factor
            )
            future_to_point[future] = gid

        with Outputs(out_fp, mode="a") as out:
            for future in as_completed(future_to_point):
                result = future.result()
                gid = future_to_point.pop(future)

                ind = project_points.index(gid)
                for dset, data in result.items():
                    out[dset,  ind] = np.array([data])

    return out_fp.as_posix()
