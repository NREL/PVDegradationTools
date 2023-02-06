"""Collection of classes and functions for standard development.
"""

import numpy as np
import pandas as pd
import pvlib
from rex import NSRDBX


def module_temperature(nsrdb_file, gid, 
                       temp_model='sapm', conf='open_rack_glass_polymer', 
                       tilt=None, azimuth=180, sky_model='isotropic'):

    """
    Calculate module temperature based on weather data from the National Solar Radiation 
    Database (NSRDB) for a given location (gid).
    
    Parameters
    ----------
    nsrdb_file : str
        The file path to the NSRDB file.
    gid : int
        The geographical location ID in the NSRDB file.
    temp_model : str, optional
        The temperature model to use, 'sapm' from pvlib by default.
    conf : str, optional
        The configuration of the PV module architecture and mounting configuration.
        Options: 'open_rack_glass_polymer' (default), 'open_rack_glass_glass',
        'close_mount_glass_glass', 'insulated_back_glass_polymer'
    tilt : float, optional
        The tilt angle of the PV panels in degrees, if None, the latitude of the 
        location is used.
    azimuth : float, optional
        The azimuth angle of the PV panels in degrees, 180 by default - facing south.
    sky_model : str, optional
        The pvlib sky model to use, 'isotropic' by default.
    
    Returns
    -------
    pandas.DataFrame
        The module temperature in degrees Celsius at each time step.
    """

    with NSRDBX(nsrdb_file, hsds=False) as f:
        meta = f.meta.loc[gid]
        times = f.time_index

    location = pvlib.location.Location(latitude=meta['latitude'],
                                       longitude=meta['longitude'],
                                       altitude=meta['elevation'])
    
    #TODO: Is timeshift necessary? NSRDB/TMY timestamps are XX:30
    solar_position = location.get_solarposition(times)

    with NSRDBX(nsrdb_file, hsds=False) as f:
        dni = f.get_gid_ts('dni', gid)
        ghi = f.get_gid_ts('ghi', gid)
        dhi = f.get_gid_ts('dhi', gid)
        air_temperature = f.get_gid_ts('air_temperature', gid)
        wind_speed = f.get_gid_ts('wind_speed', gid)

    #TODO: change for handling HSAT tracking passed or requested
    if tilt is None:
        tilt = float(meta['latitude'])

    df_poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        model=sky_model)

    #TODO: Ask Silvana why she used sapm_cell?
    parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[temp_model][conf]
    module_temperature = pvlib.temperature.sapm_module(
        poa_global=df_poa['poa_global'], 
        temp_air=air_temperature, 
        wind_speed=wind_speed,
        a=parameters['a'],
        b=parameters['b'])


    # cell_temperature = pvlib.temperature.sapm_cell(
    #     poa_global=df_poa['poa_global'], 
    #     temp_air=air_temperature, 
    #     wind_speed=wind_speed,
    #     **parameters)

    # pd.testing.assert_series_equal(module_temperature, cell_temperature)
    # #TODO: Why is cell and module temp the same?

    return module_temperature


def ideal_installation_distance(T_0, T_inf, level=0, x_0=6.1):
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
    tilt : float, optional
        Tilt of the PV array. If None, uses latitude. 
    azimuth : float, optional
        Azimuth of the PV array. The default is 180, facing south
    skymodel : str
        Tells PVlib which sky model to use. Default 'isotropic'.

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

    if level == 0:
        T98 = 70
    if level == 1:
        T98 = 80

    Tq_0 = T_0.quantile(q=0.98, interpolation='linear')
    Tq_inf = T_inf.quantile(q=0.98, interpolation='linear')

    x = -x_0 * np.log(1-(Tq_0-T98)/(Tq_0-Tq_inf))

    return x


def test_pipeline(nsrdb_file, gid):

    """modeling chain example"""

    T_0 = module_temperature(nsrdb_file, gid, 
                       temp_model='sapm', conf='insulated_back_glass_polymer', 
                       tilt=None, azimuth=180, sky_model='isotropic')

    T_inf = module_temperature(nsrdb_file, gid, 
                       temp_model='sapm', conf='open_rack_glass_polymer', 
                       tilt=None, azimuth=180, sky_model='isotropic')

    x = ideal_installation_distance(T_0, T_inf, level=0, x_0=6.1)

    return x

