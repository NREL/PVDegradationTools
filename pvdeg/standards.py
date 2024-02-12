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

from pvdeg import temperature, spectral, utilities, weather


def eff_gap_parameters(
    weather_df=None,
    meta=None,
    module_temp=None,
    weather_kwarg=None,
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    tilt=None,
    azimuth=None,
    wind_factor=0.33,
):
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
    measured_df : pd.DataFrame
        Measured module temperature data.
    sky_model : str, optional
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    temp_model : str, optional
        Options: 'sapm'.  'pvsyst' and 'faiman' will be added later.
        Performs the calculation for the cell temperature.
    conf_0 : str, optional        Model for the high temperature module on the exponential decay curve.
        Default: 'insulated_back_glass_polymer'
    conf_inf : str, optional
        Model for the lowest temperature module on the exponential decay curve.
        Default: 'open_rack_glass_polymer'
    tilt : float, optional
        Tilt angle of PV system relative to horizontal. [°]
    azimuth : float, optional
        Azimuth angle of PV system relative to north. [°]
    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement heights
        between weather database (e.g. NSRDB) and the tempeature model (e.g. SAPM)
        The NSRDB provides calculations at 2 m (i.e module height) but SAPM uses a 10 m height.
        It is recommended that a power-law relationship between height and wind speed of 0.33
        be used*. This results in a wind speed that is 1.7 times higher. It is acknowledged that
        this can vary significantly.


    References
    ----------
    R. Rabbani, M. Zeeshan, "Exploring the suitability of MERRA-2 reanalysis data for wind energy
    estimation, analysis of wind characteristics and energy potential assessment for selected
    sites in Pakistan", Renewable Energy 154 (2020) 1240-1251.


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
    poa : float
        An array of values for the plane of array irradiance, [W/m²]

    """

    parameters = ["temp_air", "wind_speed", "dhi", "ghi", "dni"]

    if isinstance(weather_df, dd.DataFrame):
        weather_df = weather_df[parameters].compute()
        weather_df.set_index("time", inplace=True)
    elif isinstance(weather_df, pd.DataFrame):
        weather_df = weather_df[parameters]
    elif weather_df is None:
        weather_df, meta = weather.get(**weather_kwarg)

    if tilt == None:
        tilt = meta["latitude"]

    if azimuth == None:  # Sets the default orientation to equator facing.
        if float(meta["latitude"]) < 0:
            azimuth = 0
        else:
            azimuth = 180
    if "wind_height" not in meta.keys():
        wind_factor = 1

    solar_position = spectral.solar_position(weather_df, meta)
    poa = spectral.poa_irradiance(
        weather_df,
        meta,
        sol_position=solar_position,
        tilt=tilt,
        azimuth=azimuth,
        sky_model=sky_model,
    )
    T_0 = temperature.cell(
        weather_df=weather_df,
        meta=meta,
        poa=poa,
        temp_model=temp_model,
        conf=conf_0,
        wind_factor=wind_factor,
    )
    T_inf = temperature.cell(
        weather_df=weather_df,
        meta=meta,
        poa=poa,
        temp_model=temp_model,
        conf=conf_inf,
        wind_factor=wind_factor,
    )
    T_measured = module_temp
    T_ambient = weather_df["temp_air"]

    return T_0, T_inf, T_measured, T_ambient, poa


def eff_gap(T_0, T_inf, T_measured, T_ambient, poa, x_0=6.5, poa_min=100, t_amb_min=0):
    """
    Calculate the effective standoff distance for rooftop mounded PV system
    according to IEC TS 63126. The 98ᵗʰ percentile calculations for T_0 and T_inf are
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
    for i in range(0, len(T_0)):
        if T_ambient.iloc[i] > t_amb_min:
            if poa.poa_global.iloc[i] > poa_min:
                n = n + 1
                summ = summ + (T_0.iloc[i] - T_measured.iloc[i]) / (
                    T_0.iloc[i] - T_inf.iloc[i]
                )

    try:
        x_eff = -x_0 * np.log(1 - summ / n)
    except RuntimeWarning as e:
        x_eff = (
            np.nan
        )  # results if the specified T₉₈ is cooler than an open_rack temperature
    if x_eff < 0:
        x_eff = 0

    return x_eff


def standoff(
    weather_df=None,
    meta=None,
    weather_kwarg=None,
    tilt=0,
    azimuth=None,
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    T98=70,  # [°C]
    x_0=6.5,  # [cm]
    wind_factor=0.33,
):
    """
    Calculate a minimum standoff distance for roof mounded PV systems.
    Will default to horizontal tilt. If the azimuth is not provided, it
    will use equator facing.
    You can use customized temperature models for the building integrated
    and the rack mounted configuration, but it will still assume an
    exponential decay.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    tilt : float, optional
        Tilt angle of PV system relative to horizontal. [°]
    azimuth : float, optional
        Azimuth angle of PV system relative to north. [°]
    sky_model : str, optional
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    temp_model : str, optional
        Options: 'sapm'.  'pvsyst' and 'faiman' will be added later.
        Performs the calculations for the cell temperature.
    conf_0 : str, optional
        Model for the high temperature module on the exponential decay curve.
        Default: 'insulated_back_glass_polymer'
    conf_inf : str, optional
        Model for the lowest temperature module on the exponential decay curve.
        Default: 'open_rack_glass_polymer'
    x_0 : float, optional
        Thermal decay constant (cm), [Kempe, PVSC Proceedings 2023]
    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement heights
        between weather database (e.g. NSRDB) and the tempeature model (e.g. SAPM)
        The NSRDB provides calculations at 2 m (i.e module height) but SAPM uses a 10 m height.
        It is recommended that a power-law relationship between height and wind speed of 0.33
        be used*. This results in a wind speed that is 1.7 times higher. It is acknowledged that
        this can vary significantly.

    R. Rabbani, M. Zeeshan, "Exploring the suitability of MERRA-2 reanalysis data for wind energy
        estimation, analysis of wind characteristics and energy potential assessment for selected
        sites in Pakistan", Renewable Energy 154 (2020) 1240-1251.

    Returns
    -------
    x : float [cm]
        Minimum installation distance in centimeter per IEC TS 63126 when the default settings are used.
        Effective gap "x" for the lower limit for Level 1 or Level 0 modules (IEC TS 63216)
    T98_0 : float [°C]
        This is the 98ᵗʰ percential temperature of a theoretical module with no standoff.
    T98_inf : float [°C]
        This is the 98ᵗʰ percential temperature of a theoretical rack mounted module.
    T98 : float [°C]
        This is the 98ᵗʰ percential temperature that was calculated to.

    References
    ----------
    M. Kempe, et al. Close Roof Mounted System Temperature Estimation for Compliance
    to IEC TS 63126, PVSC Proceedings 2023
    """

    if azimuth == None:  # Sets the default orientation to equator facing.
        if float(meta["latitude"]) < 0:
            azimuth = 0
        else:
            azimuth = 180
    if "wind_height" not in meta.keys():
        wind_factor = 1
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
        weather_df=weather_df,
        meta=meta,
        sol_position=solar_position,
        tilt=tilt,
        azimuth=azimuth,
        sky_model=sky_model,
    )
    T_0 = temperature.cell(
        weather_df=weather_df,
        meta=meta,
        poa=poa,
        temp_model=temp_model,
        conf=conf_0,
        wind_factor=wind_factor,
    )
    T98_0 = T_0.quantile(q=0.98, interpolation="linear")
    T_inf = temperature.cell(
        weather_df=weather_df,
        meta=meta,
        poa=poa,
        temp_model=temp_model,
        conf=conf_inf,
        wind_factor=wind_factor,
    )
    T98_inf = T_inf.quantile(q=0.98, interpolation="linear")

    try:
        x = -x_0 * np.log(1 - (T98_0 - T98) / (T98_0 - T98_inf))
    except RuntimeWarning as e:
        x = (
            np.nan
        )  # results if the specified T₉₈ is cooler than an open_rack temperature
    if x < 0:
        x = 0

    res = {"x": x, "T98_0": T98_0, "T98_inf": T98_inf, "T98": T98}
    df_res = pd.DataFrame.from_dict(res, orient="index").T

    return df_res


def interpret_standoff(
    standoff_1=pd.DataFrame.from_dict({"T98": None}, orient="index").T,
    standoff_2=pd.DataFrame.from_dict({"T98": None}, orient="index").T,
):
    """
    This is a set of statments designed to provide a printable output to interpret the results of standoff calculations.
    At a minimum, data for Standoff_1 must be included.

    Parameters
    ----------
    Standoff_1 and Standoff_2 : df
    This is the dataframe output from the standoff calculation method for calculations at 70°C and 80°C, respectively.
        x : float [°C]
            Minimum installation distance in centimeter per IEC TS 63126 when the default settings are used.
            Effective gap "x" for the lower limit for Level 1 or Level 0 modules (IEC TS 63216)
        T98_0 : float [°C]
            This is the 98ᵗʰ percential temperature of a theoretical module with no standoff.
        T98_inf : float [°C]
            This is the 98ᵗʰ percential temperature of a theoretical rack mounted module.

    Returns
    -------
    Output: str
        This is an interpretation of the accepatble effective standoff values suitable for presentation.
    """

    if (standoff_1.T98[0] == 80 and standoff_2.T98[0] == 70) or standoff_2.T98[0] == 70:
        standoff_1, standoff_2 = standoff_2, standoff_1

    if standoff_1.T98[0] == 70 and standoff_2.T98[0] == 80:
        Output = (
            "The estimated temperature of an insulated-back module is "
            + "%.1f" % standoff_1.T98_0[0]
            + "°C. \n"
        )
        Output = (
            Output
            + "The estimated temperature of an open-rack module is "
            + "%.1f" % standoff_1.T98_inf[0]
            + "°C. \n"
        )
        Output = (
            Output
            + "Level 0 certification is valid for a standoff greather than "
            + "%.1f" % standoff_1.x[0]
            + " cm. \n"
        )
        if standoff_1.x[0] > 0:
            if standoff_2.x[0] > 0:
                Output = (
                    Output
                    + "Level 1 certification is required for a standoff between than "
                    + "%.1f" % standoff_1.x[0]
                    + " cm, and "
                    + "%.1f" % standoff_2.x[0]
                    + " cm. \n"
                )
                Output = (
                    Output
                    + "Level 2 certification is required for a standoff less than "
                    + "%.1f" % standoff_2.x[0]
                    + " cm."
                )
            else:
                Output = (
                    Output
                    + "Level 1 certification is required for a standoff less than "
                    + "%.1f" % standoff_1.x[0]
                    + " cm. \n"
                )
                Output = (
                    Output
                    + "Level 2 certification is never required for this temperature profile."
                )
    elif standoff_1.T98[0] == 70:
        Output = (
            "The estimated temperature of an insulated-back module is "
            + "%.1f" % standoff_1.T98_0[0]
            + "°C. \n"
        )
        Output = (
            Output
            + "The estimated temperature of an open-rack module is "
            + "%.1f" % standoff_1.T98_inf[0]
            + "°C. \n"
        )
        Output = (
            Output
            + "The minimum standoff for Level 0 certification and T₉₈<70°C is "
            + "%.1f" % standoff_1.x[0]
            + " cm."
        )
    else:
        Output = "Incorrect data for IEC TS 63126 Level determination."

    return Output


def T98_estimate(
    weather_df=None,
    meta=None,
    weather_kwarg=None,
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    wind_factor=0.33,
    tilt=None,
    azimuth=None,
    x_eff=None,
    x_0=6.5,
):
    """
    Estimate the 98ᵗʰ percential temperature for the module at the given tilt, azimuth, and x_eff.
    If any of these factors are supplied, it default to latitide tilt, equatorial facing, and
    open rack mounted, respectively.

    Parameters
    ----------
    x_eff : float
        This is the effective module standoff distance according to the model. [cm]
    x_0 : float, optional
        Thermal decay constant. [cm]
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    tilt : float,
        Tilt angle of PV system relative to horizontal. [°]
    azimuth : float, optional
        Azimuth angle of PV system relative to north. [°]
    sky_model : str, optional
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    temp_model : str, optional
        Options: 'sapm'.  'pvsyst' and 'faiman' will be added later.
        Performs the calculations for the cell temperature.
    conf_0 : str, optional
        Model for the high temperature module on the exponential decay curve.
        Default: 'insulated_back_glass_polymer'
    conf_inf : str, optional
        Model for the lowest temperature module on the exponential decay curve.
        Default: 'open_rack_glass_polymer'
    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement heights
        between weather database (e.g. NSRDB) and the tempeature model (e.g. SAPM)
        The NSRDB provides calculations at 2 m (i.e module height) but SAPM uses a 10 m height.
        It is recommended that a power-law relationship between height and wind speed of 0.33
        be used*. This results in a wind speed that is 1.7 times higher. It is acknowledged that
        this can vary significantly.

    R. Rabbani, M. Zeeshan, "Exploring the suitability of MERRA-2 reanalysis data for wind energy
        estimation, analysis of wind characteristics and energy potential assessment for selected
        sites in Pakistan", Renewable Energy 154 (2020) 1240-1251.

    Returns
    -------
    T98: float
        This is the 98ᵗʰ percential temperature for the module at the given tilt, azimuth, and x_eff.

    """

    if tilt == None:
        tilt = meta["latitude"]

    if azimuth == None:  # Sets the default orientation to equator facing.
        if float(meta["latitude"]) < 0:
            azimuth = 0
        else:
            azimuth = 180
    if "wind_height" not in meta.keys():
        wind_factor = 1
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
        weather_df=weather_df,
        meta=meta,
        sol_position=solar_position,
        tilt=tilt,
        azimuth=azimuth,
        sky_model=sky_model,
    )

    T_inf = temperature.cell(
        weather_df,
        meta,
        poa,
        temp_model,
        conf_inf,
        wind_factor,
    )
    T98_inf = T_inf.quantile(q=0.98, interpolation="linear")

    if x_eff == None:
        return T98_inf
    else:
        T_0 = temperature.cell(
            weather_df,
            meta,
            poa,
            temp_model,
            conf_0,
            wind_factor,
        )
        T98_0 = T_0.quantile(q=0.98, interpolation="linear")
        T98 = T98_0 - (T98_0 - T98_inf) * (1 - np.exp(-x_eff / x_0))
        return T98


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
