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
    weather_kwarg : dict
        other variables needed to access a particular weather dataset.
    sky_model : str, optional
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    temp_model : str, optional
        Options: 'sapm'.  'pvsyst' and 'faiman' and others from PVlib.
        Performs the calculation for the cell temperature.
    conf_0 : str, optional
        Model for the high temperature module on the exponential decay curve.
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

    return T_0, T_inf, poa


def eff_gap(T_0, T_inf, T_measured, T_ambient, poa, x_0=6.5, poa_min=400, t_amb_min=0):
    """
    Calculate the effective standoff distance for rooftop mounded PV system
    according to IEC TS 63126. The 98ᵗʰ percentile calculations for T_0 and T_inf are
    also calculated.

    Parameters
    ----------
    T_0 : pd.series
        An array of temperature values for a module with an insulated back or an
        alternatively desired small or zero standoff. Used as the basis for the
        maximum achievable temperature, [°C].
    T_inf : pd.series
        An array of temperature values for a module that is rack mounted, [°C].
    T_measured : pd.series
        An array of values for the measured module temperature, [°C].
    T_ambient : pd.series
        An array of values for the ambient temperature, [°C].
    poa : pd.series
        An array of values for the plane of array irradiance, [W/m²]
    x_0 : float, optional
        Thermal decay constant [cm], [Kempe, PVSC Proceedings 2023].
        According to edition 2 of IEC TS 63126 a value of 6.5 cm is recommended.
    poa_min : float, optional
        Minimum iradiance 
    t_ambient_min : floa, optional
        Minimum am

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
            if poa.iloc[i] > poa_min:
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
    tilt=None,
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
    weather_kwarg : dict
        other variables needed to access a particular weather dataset.
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
        x = np.nan
        # results if the specified T₉₈ is cooler than an open_rack temperature
    if x < 0:
        x = 0

    res = {"x": x, "T98_0": T98_0, "T98_inf": T98_inf}
    df_res = pd.DataFrame.from_dict(res, orient="index").T

    return df_res


def interpret_standoff(standoff_1=None, standoff_2=None):
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

    if standoff_1 is not None:
        x70 = standoff_1["x"].iloc[0]
        T98_0 = standoff_1["T98_0"].iloc[0]
        T98_inf = standoff_1["T98_inf"].iloc[0]
        if standoff_2 is not None:
            x80 = standoff_2["x"].iloc[0]
        else:
            try:
                x80 = -(-x70 / (np.log(1 - (T98_0 - 70) / (T98_0 - T98_inf)))) * np.log(
                    1 - (T98_0 - 80) / (T98_0 - T98_inf)
                )
            except RuntimeWarning as e:
                x80 = None
    else:
        x70 = None

    if x70 == None:
        Output = "Insufficient data for IEC TS 63126 Level determination."
    else:
        if T98_0 is not None:
            Output = (
                "The estimated T₉₈ of an insulated-back module is "
                + "%.1f" % T98_0
                + "°C. \n"
            )
        if T98_inf is not None:
            Output = (
                Output
                + "The estimated T₉₈ of an open-rack module is "
                + "%.1f" % T98_inf
                + "°C. \n"
            )
        if x80 == None:
            Output = (
                Output
                + "The minimum standoff for Level 0 certification and T₉₈<70°C is "
                + "%.1f" % x70
                + " cm."
            )
        else:
            Output = (
                Output
                + "Level 0 certification is valid for a standoff greather than "
                + "%.1f" % x70
                + " cm. \n"
            )
            if x70 > 0:
                if x80 > 0:
                    Output = (
                        Output
                        + "Level 1 certification is required for a standoff between than "
                        + "%.1f" % x70
                        + " cm, and "
                        + "%.1f" % x80
                        + " cm. \n"
                    )
                    Output = (
                        Output
                        + "Level 2 certification is required for a standoff less than "
                        + "%.1f" % x80
                        + " cm."
                    )
                else:
                    Output = (
                        Output
                        + "Level 1 certification is required for a standoff less than "
                        + "%.1f" % x70
                        + " cm. \n"
                    )
                    Output = (
                        Output
                        + "Level 2 certification is never required for this temperature profile."
                    )

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
    weather_kwarg : dict
        other variables needed to access a particular weather dataset.
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
        weather_df=weather_df,
        meta=meta,
        poa=poa,
        temp_model=temp_model,
        conf_inf=conf_inf,
        wind_factor=wind_factor,
    )
    T98_inf = T_inf.quantile(q=0.98, interpolation="linear")

    if x_eff == None:
        return T98_inf
    else:
        T_0 = temperature.cell(
            weather_df=weather_df,
            meta=meta,
            poa=poa,
            temp_model=temp_model,
            conf_0=conf_0,
            wind_factor=wind_factor,
        )
        T98_0 = T_0.quantile(q=0.98, interpolation="linear")
        T98 = T98_0 - (T98_0 - T98_inf) * (1 - np.exp(-x_eff / x_0))
        return T98


def standoff_x(
    weather_df,
    meta,
    tilt,
    azimuth,
    sky_model,
    temp_model=None,
    conf_0=None,
    conf_inf=None,
    T98=None,
    x_0=None,
    wind_factor=None,
):
    """
    Calculate a minimum standoff distance for roof mounded PV systems.
    Will default to horizontal tilt and return only that value. It just passes
    through the calling function and returns a single value.

    Parameters
    ----------
    See Standoff() documentation

    Returns
    -------
    x : float [cm]
        Minimum installation distance in centimeter per IEC TS 63126 when the default settings are used.
        Effective gap "x" for the lower limit for Level 1 or Level 0 modules (IEC TS 63216)
    """

    temp_df = standoff(
        weather_df=weather_df,
        meta=meta,
        tilt=tilt,
        azimuth=azimuth,
        sky_model=sky_model,
        temp_model=temp_model,
        conf_0=conf_0,
        conf_inf=conf_inf,
        T98=T98,
        x_0=x_0,
        wind_factor=wind_factor,
    ).x[0]

    return temp_df


def vertical_POA(
    weather_df,
    meta,
    jsonfolder='/projects/pvsoiling/pvdeg/analysis/northern_lat/jsons',
    samjsonname='vertical',
    weather_kwarg=None,
):
    """
    Run a SAM

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    weather_kwarg : dict
        other variables needed to access a particular weather dataset.
    jsonfolder : string
        Location and base name for the json files

    Returns
    -------
    annual_gh : float [Wh/m2/y]
        Annual GHI
    annual_energy : float [kWh]
        Annual AC energy
    lcoa_nom : float [cents/kWh]
        LCOE Levelized cost of energy nominal

    """

    import PySAM
    import PySAM.Pvsamv1 as PV
    import PySAM.Grid as Grid
    import PySAM.Utilityrate5 as UtilityRate
    import PySAM.Cashloan as Cashloan
    import json
    import os
    import sys
    
    parameters = ["temp_air", "wind_speed", "dhi", "ghi", "dni"]
    print("weather_df KEYs", weather_df.keys())
    print("meta KEYs", meta.keys())

    
    if isinstance(weather_df, dd.DataFrame):
        weather_df = weather_df[parameters].compute()
        weather_df.set_index("time", inplace=True)
    elif isinstance(weather_df, pd.DataFrame):
        weather_df = weather_df[parameters]
    elif weather_df is None:
        weather_df, meta = weather.get(**weather_kwarg)


    file_names = ["pvsamv1", "grid", "utilityrate5", "cashloan"]
    pv4 = PV.new()  # also tried PVWattsSingleOwner
    grid4 = Grid.from_existing(pv4)
    ur4 = UtilityRate.from_existing(pv4)
    so4 = Cashloan.from_existing(grid4, 'FlatPlatePVCommercial')

    # LOAD Values
    for count, module in enumerate([pv4, grid4, ur4, so4]):
        filetitle= samjsonname + '_' + file_names[count] + ".json"
        with open(os.path.join(jsonfolder,filetitle), 'r') as file:
            data = json.load(file)
            for k, v in data.items():
                if k == 'number_inputs':
                    continue
                try:
                    if sys.version.split(' ')[0] == '3.11.7': 
                        # Check needed for python 3.10.7 and perhaps other releases above 3.10.4.
                        # This prevents the failure "UnicodeDecodeError: 'utf-8' codec can't decode byte... 
                        # This bug will be fixed on a newer version of pysam (currently not working on 5.1.0)
                        if 'adjust_' in k:  # This check is needed for Python 3.10.7 and some others. Not needed for 3.7.4
                            print(k)
                            k = k.split('adjust_')[1]
                    module.value(k, v)
                except AttributeError:
                    # there is an error is setting the value for ppa_escalation
                    print(module, k, v)

    pv4.unassign('solar_resource_file')
                
    if meta.get('tz') == None: 
        meta['tz'] = '+0'

    if "albedo" not in weather_df.columns:
        weather_df['albedo'] = 0.2
    
    data = {'dn':list(weather_df.dni),
           'df':list(weather_df.dhi),
            'gh':list(weather_df.ghi),
           'tdry':list(weather_df.temp_air),
           'wspd':list(weather_df.wind_speed),
           'lat':meta['latitude'],
           'lon':meta['longitude'],
           'tz':meta['tz'],
           'elev':meta['altitude'],
           'year':list(weather_df.index.year),
           'month':list(weather_df.index.month),
           'day':list(weather_df.index.day),
           'hour':list(weather_df.index.hour),
           'minute':list(weather_df.index.minute),
           'alb':list(weather_df.albedo)}

    pv4.value('solar_resource_data', data)

    pv4.execute()
    grid4.execute()
    ur4.execute()
    so4.execute()
    
    # SAVE RESULTS|
    results = pv4.Outputs.export()
    economicresults = so4.Outputs.export()
    
    annual_gh = results['annual_gh']
    annual_energy = results['annual_ac_gross']
    lcoe = economicresults['lcoe_nom']
    
    res = {"annual_gh": x, "annual_energy": annual_energy, "lcoe_nom": lcoe_nom}
    df_res = pd.DataFrame.from_dict(res, orient="index").T

    return df_res
