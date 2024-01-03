"""
Collection of classes and functions for standard development.
"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
import pvlib
from rex import NSRDBX
from rex import Outputs
from pathlib import Path
from random import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# from gaps import ProjectPoints

from . import temperature
from . import spectral
from . import utilities
from . import weather

def eff_gap_parameters(
    weather_df=None,
    meta=None,
    weather_kwarg=None,
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    wind_speed_factor=1.7,):

    """
    Calculate and set up data necessary to calculate the effective standoff distance for rooftop mounded PV system 
    according to IEC TS 63126. The the temperature is modeled for T_0 and T_inf and the corresponding test
    module temperature must be provided in the weather data.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    tilt : float, 
        Tilt angle of PV system relative to horizontal. 
    azimuth : float, optional
        Azimuth angle of PV system relative to north.
    sky_model : str, optional
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    temp_model : str, optional
        Options: 'sapm'.  'pvsyst' and 'faiman' will be added later.
    conf_0 : str, optional
        Default: 'insulated_back_glass_polymer'
    conf_inf : str, optional
        Default: 'open_rack_glass_polymer'
    wind_speed_factor : float, optional
        Wind speed correction factor to account for different wind speed measurement heights
        between weather database (e.g. NSRDB) and the tempeature model (e.g. SAPM)
        The NSRD provides calculations at 2m (i.e module height) but SAPM uses a 10m height.
        It is recommended that a power-law relationship between height and wind speed of 0.33 be used.
        This results in a wind_speed_factor of 1.7. It is acknowledged that this can vary significantly. 
    
    Returns
    -------
    T_0 : float
        An array of temperature values for a module with an insulated back or an 
        alternatively desired small or zero standoff, [°C]. Used as the basis for the 
        maximum achievable temperature.
    T_inf : float
        An array of temperature values for a module that is rack mounted, [°C].
    T_measured : float
        An array of values for the test module in the system, [°C] interest.

    """

    parameters = ["temp_air", "wind_speed", "dhi", "ghi", "dni", "temp_measured"]

    if isinstance(weather_df, dd.DataFrame):
        weather_df = weather_df[parameters].compute()
        weather_df.set_index("time", inplace=True)
    elif isinstance(weather_df, pd.DataFrame):
        weather_df = weather_df[parameters]
    elif weather_df is None:
        weather_df, meta = weather.get(**weather_kwarg)

    solar_position = spectral.solar_position(weather_df, meta)
    poa = spectral.poa_irradiance(
        weather_df,
        meta,
        sol_position=solar_position,
        tilt=meta.tilt,
        azimuth=meta.azimuth,
        sky_model=sky_model, )
    T_0 = temperature.module(
        weather_df, meta, poa, temp_model, conf_0, wind_speed_factor )
    T_inf = temperature.module(
        weather_df, meta, poa, temp_model, conf_inf, wind_speed_factor )
    T_measured = weather_df.Module_Temperature
    T_ambient = weather_df.temperature

    return T_0, T_inf, T_measured, T_ambient, poa


def eff_gap(T_0, T_inf, T_measured, T_ambient, poa, x_0=6.5, poa_min=100, t_amb_min=0):
    """
    Calculate the effective standoff distance for rooftop mounded PV system 
    according to IEC TS 63126. The 98th percentile calculations for T_0 and T_inf are
    also calculated.

    Parameters
    ----------
    T_0 : float
        An array of temperature values for a module with an insulated back or an 
        alternatively desired small or zero standoff. Used as the basis for the 
        maximum achievable temperature, [°C].
    T_inf : float
        An array of temperature values for a module that is rack mounted, [°C].
    T_measured : float
        An array of values for the measured module temperature, [°C].
    T_ambient : float
        An array of values for the ambient temperature, [°C].
        poa : float
        An array of values for the plane of array irradiance, [W/m²]
    x_0 : float, optional
        Thermal decay constant [cm], [Kempe, PVSC Proceedings 2023].
        According to edition 2 of IEC TS 63126 a value of 6.5 cm is recommended.

    Returns
    -------
    x_eff : float
        Effective module standoff distance. While not the actual physical standoff, 
        this can be thought of as a heat transfer coefficient that produces results
        that are similar to the modeled singl module temperature with that gap.

    References
    ----------
    M. Kempe, et al. Close Roof Mounted System Temperature Estimation for Compliance
    to IEC TS 63126, PVSC Proceedings 2023
    """

    n = 0
    summ = 0
    for i in range(0, len(t_0)):
        if T_ambient > t_amb_min:
            if poa > poa_min:
                n = n + 1
                summ = summ + (T_0[i] - T_module[i]) / (T_0[i] - T_inf[i])

    try:
        x_eff = -x_0 * np.log(1 - summ/n)
    except RuntimeWarning as e:
        x_eff = np.nan # results if the specified T₉₈ is cooler than an open_rack temperature 
    if x<0: 
        x_eff=0 

    return x_eff


def standoff(
    weather_df=None,
    meta=None,
    weather_kwarg=None,
    tilt=None,
    azimuth=180,
    sky_model="isotropic",
    temp_model="sapm",
    #module_type="glass_polymer",  # This is removed forcing one to specify alternatives if desired
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    #level=1, # I am removing this because it is unnecessarily complicated
    T98=70, # [°C]
    x_0=6.5, # [cm]
    wind_speed_factor=1.7,
):
    """
    Calculate a minimum standoff distance for roof mounded PV systems.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    tilt : float, optional
        Tilt angle of PV system relative to horizontal.
    azimuth : float, optional
        Azimuth angle of PV system relative to north.
    sky_model : str, optional
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    temp_model : str, optional
        Options: 'sapm'.  'pvsyst' and 'faiman' will be added later.
    conf_0 : str, optional
        Default: 'insulated_back_glass_polymer'
    conf_inf : str, optional
        Default: 'open_rack_glass_polymer'
    x0 : float, optional
        Thermal decay constant (cm), [Kempe, PVSC Proceedings 2023]
    wind_speed_factor : float, optional
        Wind speed correction factor to account for different wind speed measurement heights
        between weather database (e.g. NSRDB) and the tempeature model (e.g. SAPM)
        The NSRD provides calculations at 2m (i.e module height) but SAPM uses a 10m height.
        It is recommended that a power-law relationship between height and wind speed of 0.33 be used.
        This results in a wind_speed_factor of 1.7. It is acknowledged that this can vary significantly. 

    Returns
    -------
    x : float [cm]
        Minimum installation distance in centimeter per IEC TS 63126 when the default settings are used.
        Effective gap "x" for the lower limit for Level 1 or Level 0 modules (IEC TS 63216)
    T98_0 : float [°C]
        This is the 98th percential temperature of a theoretical module with no standoff.
     T98_inf : float [°C]
        This is the 98th percential temperature of a theoretical rack mounted module.
        
    References
    ----------
    M. Kempe, et al. Close Roof Mounted System Temperature Estimation for Compliance
    to IEC TS 63126, PVSC Proceedings 2023
    """

    parameters = ["temp_air", "wind_speed", "dhi", "ghi", "dni"]

    if isinstance(weather_df, dd.DataFrame):
        weather_df = weather_df[parameters].compute()
        weather_df.set_index("time", inplace=True)
    elif isinstance(weather_df, pd.DataFrame):
        weather_df = weather_df[parameters]
    elif weather_df is None:
        weather_df, meta = weather.get(**weather_kwarg)

    solar_position = spectral.solar_position(weather_df, meta)
    poa = spectral.poa_irradiance(
        weather_df,
        meta,
        sol_position=solar_position,
        tilt=tilt,
        azimuth=azimuth,
        sky_model=sky_model, )
    T_0 = temperature.module(
        weather_df, meta, poa, temp_model, conf_0, wind_speed_factor )
    T98_0 = T_0.quantile(q=0.98, interpolation="linear")
    T_inf = temperature.module(
        weather_df, meta, poa, temp_model, conf_inf, wind_speed_factor )
    T98_inf = T_inf.quantile(q=0.98, interpolation="linear")

    try:
        x = -x_0 * np.log(1 - (T98_0 - T98) / (T98_0 - T98_inf))
    except RuntimeWarning as e:
        x = np.nan # results if the specified T₉₈ is cooler than an open_rack temperature 
    if x<0: 
        x=0 

    res = {"x": x, "T98_0": T98_0, "T98_inf": T98_inf}
    df_res = pd.DataFrame.from_dict(res, orient="index").T

    return df_res

def T98_estimate(
    weather_df=None,
    meta=None,
    weather_kwarg=None,
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    wind_speed_factor=1.7,
    x_eff=None,
    x_0=6.5):

    """
    Calculate and set up data necessary to calculate the effective standoff distance for rooftop mounded PV system 
    according to IEC TS 63126. The the temperature is modeled for T_0 and T_inf and the corresponding test
    module temperature must be provided in the weather data.

    Parameters
    ----------
    x_eff : float
        This is the effective module standoff distance according to the model.
    x_0 : float, optional
        Thermal decay constant [cm],
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    tilt : float, 
        Tilt angle of PV system relative to horizontal. 
    azimuth : float, optional
        Azimuth angle of PV system relative to north.
    sky_model : str, optional
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    temp_model : str, optional
        Options: 'sapm'.  'pvsyst' and 'faiman' will be added later.
    conf_0 : str, optional
        Default: 'insulated_back_glass_polymer'
    conf_inf : str, optional
        Default: 'open_rack_glass_polymer'
    wind_speed_factor : float, optional
        Wind speed correction factor to account for different wind speed measurement heights
        between weather database (e.g. NSRDB) and the tempeature model (e.g. SAPM)
        The NSRD provides calculations at 2m (i.e module height) but SAPM uses a 10m height.
        It is recommended that a power-law relationship between height and wind speed of 0.33 be used.
        This results in a wind_speed_factor of 1.7. It is acknowledged that this can vary significantly. 
    
    Returns
    -------
    T98: float
        This is the 98th percential temperature for the module at the given tilt, azimuth, and x_eff.

    """

    parameters = ["temp_air", "wind_speed", "dhi", "ghi", "dni", "temp_measured"]

    if isinstance(weather_df, dd.DataFrame):
        weather_df = weather_df[parameters].compute()
        weather_df.set_index("time", inplace=True)
    elif isinstance(weather_df, pd.DataFrame):
        weather_df = weather_df[parameters]
    elif weather_df is None:
        weather_df, meta = weather.get(**weather_kwarg)

    solar_position = spectral.solar_position(weather_df, meta)
    poa = spectral.poa_irradiance(
        weather_df,
        meta,
        sol_position=solar_position,
        tilt=meta.tilt,
        azimuth=meta.azimuth,
        sky_model=sky_model, )
    T_0 = temperature.module(
        weather_df, meta, poa, temp_model, conf_0, wind_speed_factor )
    T98_0 = T_0.quantile(q=0.98, interpolation="linear")
    T_inf = temperature.module(
        weather_df, meta, poa, temp_model, conf_inf, wind_speed_factor )
    T98_inf = T_inf.quantile(q=0.98, interpolation="linear")

    T98 = T_0 - (T_0-T_inf)*(1-np.exp(-x_eff/x_0))

    return T98

def standoff_tilt_azimuth_scan(
    weather_df=None,
    meta=None,
    weather_kwarg=None,
    tilt_count=18,
    azimuth_count=72,
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    T98=70, # [°C]
    x_0=6.5, # [cm]
    wind_speed_factor=1.7):

    """
    Calculate a minimum standoff distance for roof mounded PV systems as a function of tilt and azimuth.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    tilt_count : integer
        Step in degrees of change in tilt angle of PV system between calculations.
    azimuth_count : integer
        Step in degrees of change in Azimuth angle of PV system relative to north.
    sky_model : str, optional
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    temp_model : str, optional
        Options: 'sapm'.  'pvsyst' and 'faiman' will be added later.
    conf_0 : str, optional
        Default: 'insulated_back_glass_polymer'
    conf_inf : str, optional
        Default: 'open_rack_glass_polymer'
    x_0 : float, optional
        Thermal decay constant [cm], [Kempe, PVSC Proceedings 2023]
    wind_speed_factor : float, optional
        Wind speed correction factor to account for different wind speed measurement heights
        between weather database (e.g. NSRDB) and the tempeature model (e.g. SAPM)
        The NSRD provides calculations at 2m (i.e module height) but SAPM uses a 10m height.
        It is recommended that a power-law relationship between height and wind speed of 0.33 be used.
        This results in a wind_speed_factor of 1.7. It is acknowledged that this can vary significantly. 
    Returns
        standoff_series : 2-D array with each row consiting of tilt, azimuth, then standoff
    """
    
    standoff_series=np.array([(azimuth_count+1)*(tilt_count+1)][3])
    for x in range (0, azimuth_count+1):
        for y in range (0,tilt_count+1):
            standoff_series[x+y][0]=azimuth_count*180/azimuth_count
            standoff_series[x+y][1]=tilt_count*90/azimuth_count
            standoff_series[x+y][2]=standards.standoff(
                weather_df=weather_df, 
                meta=meta,
                T98=T98,
                tilt=y*90/tilt_count,
                azimuth=x*180/azimuth_count,
                sky_model=sky_model,
                temp_model=temp_model,
                conf_0=conf_0,
                conf_inf=conf_inf,
                x_0=x_0,
                wind_speed_factor=wind_speed_factor)
            
    return standoff_series

# def run_calc_standoff(
#     project_points,
#     out_dir,
#     tag,
#     #weather_db,
#     #weather_satellite,
#     #weather_names,
#     max_workers=None,
#     tilt=None,
#     azimuth=180,
#     sky_model='isotropic',
#     temp_model='sapm',
#     module_type='glass_polymer',
#     level=1,
#     x_0=6.1,
#     wind_speed_factor=1
# ):

#     """
#     parallelization utilizing gaps     #TODO: write docstring
#     """

#     #inputs
#     weather_arg = {}
#     #weather_arg['satellite'] = weather_satellite
#     #weather_arg['names'] = weather_names
#     weather_arg['NREL_HPC'] = True  #TODO: add argument or auto detect
#     weather_arg['attributes'] = [
#         'air_temperature',
#         'wind_speed',
#         'dhi',
#         'ghi',
#         'dni',
#         'relative_humidity'
#         ]

#     all_fields = ['x', 'T98_0', 'T98_inf']

#     out_fp = Path(out_dir) / f"out_standoff{tag}.h5"
#     shapes = {n : (len(project_points), ) for n in all_fields}
#     attrs = {'x' : {'units': 'cm'},
#              'T98_0' : {'units': 'Celsius'},
#              'T98_inf' : {'units': 'Celsius'}}
#     chunks = {n : None for n in all_fields}
#     dtypes = {n : "float32" for n in all_fields}

#     # #TODO: is there a better way to add the meta data?
#     # nsrdb_fnames, hsds  = weather.get_NSRDB_fnames(
#     #     weather_arg['satellite'],
#     #     weather_arg['names'],
#     #     weather_arg['NREL_HPC'])

#     # with NSRDBX(nsrdb_fnames[0], hsds=hsds) as f:
#     #     meta = f.meta[f.meta.index.isin(project_points.gids)]

#     Outputs.init_h5(
#         out_fp,
#         all_fields,
#         shapes,
#         attrs,
#         chunks,
#         dtypes,
#         #meta=meta.reset_index()
#         meta=project_points.df
#     )

#     future_to_point = {}
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         for idx, point in project_points.df.iterrows():
#             database = point.weather_db
#             gid = idx #int(point.gid)
#             df_weather_kwargs = point.drop('weather_db', inplace=False).filter(like='weather_')
#             df_weather_kwargs.index = df_weather_kwargs.index.map(
#                 lambda arg: arg.lstrip('weather_'))
#             weather_kwarg = weather_arg | df_weather_kwargs.to_dict()

#             weather_df, meta = weather.load(
#                 database = database,
#                 id = gid,
#                 #satellite = point.satellite,  #TODO: check input
#                 **weather_kwarg)
#             future = executor.submit(
#                 calc_standoff,
#                 weather_df,
#                 meta,
#                 tilt,
#                 azimuth,
#                 sky_model,
#                 temp_model,
#                 module_type,
#                 level,
#                 x_0,
#                 wind_speed_factor
#             )
#             future_to_point[future] = gid

#         with Outputs(out_fp, mode="a") as out:
#             for future in as_completed(future_to_point):
#                 result = future.result()
#                 gid = future_to_point.pop(future)

#                 #ind = project_points.index(gid)
#                 for dset, data in result.items():
#                     out[dset,  idx] = np.array([data])

#     return out_fp.as_posix()
