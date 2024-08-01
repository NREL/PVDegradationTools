"""Collection of classes and functions for humidity calculations."""

import numpy as np
import pandas as pd
from numba import jit, njit, vectorize, guvectorize, float64
from typing import Union

from . import temperature
from . import spectral


def _ambient(weather_df):
    """
    Calculate ambient relative humidity from dry bulb air temperature and dew point

    references:
    Alduchov, O. A., and R. E. Eskridge, 1996: Improved Magnus' form approximation of saturation
    vapor pressure. J. Appl. Meteor., 35, 601–609.
    August, E. F., 1828: Ueber die Berechnung der Expansivkraft des Wasserdunstes. Ann. Phys. Chem.,
    13, 122–137.
    Magnus, G., 1844: Versuche über die Spannkräfte des Wasserdampfs. Ann. Phys. Chem., 61, 225–247.

    Parameters:
    -----------
    weather_df : pd.DataFrame
        Datetime-indexed weather dataframe which contains (at minimum) Ambient temperature
        ('temp_air') and dew point ('temp_dew') in units [C]

    Returns:
    --------
    weather_df : pd.DataFrame
        identical datetime-indexed dataframe with addional column 'relative_humidity' containing
        ambient relative humidity [%]
    """
    temp_air = weather_df["temp_air"]
    # "Dew Point" fallback handles key-name bug in pvlib < v0.10.3.
    dew_point = weather_df.get("dew_point")

    num = np.exp(17.625 * dew_point / (243.04 + dew_point))
    den = np.exp(17.625 * temp_air / (243.04 + temp_air))
    rh_ambient = 100 * num / den

    weather_df["relative_humidity"] = rh_ambient

    return weather_df


# TODO: When is dew_yield used?
@jit(nopython=True, error_model="python")
def dew_yield(elevation, dew_point, dry_bulb, wind_speed, n):
    """
    Estimates the dew yield in [mm/day].  Calculation taken from:
    Beysens, "Estimating dew yield worldwide from a few meteo data", Atmospheric Research 167
    (2016) 146-155

    Parameters
    -----------
    elevation : int
        Site elevation [km]
    dew_point : float
        Dewpoint temperature in Celsius [°C]
    dry_bulb : float
        Air temperature "dry bulb temperature" [°C]
    wind_speed : float
        Air or windspeed measure [m/s]
    n : float
        Total sky cover(okta)
        This is a quasi emperical scale from 0 to 8 used in meterology which corresponds to
        0-sky completely clear, to 8-sky completely cloudy. Does not account for cloud type
        or thickness.

    Returns
    -------
    dew_yield : float
        Amount of dew yield in [mm/day]

    """
    wind_speed_cut_off = 4.4
    dew_yield = (1 / 12) * (
        0.37
        * (
            1
            + (0.204323 * elevation)
            - (0.0238893 * elevation**2)
            - (18.0132 - (1.04963 * elevation**2) + (0.21891 * elevation**2))
            * (10 ** (-3) * dew_point)
        )
        * ((((dew_point + 273.15) / 285) ** 4) * (1 - (n / 8)))
        + (0.06 * (dew_point - dry_bulb))
        * (1 + 100 * (1 - np.exp(-((wind_speed / wind_speed_cut_off) ** 20))))
    )

    return dew_yield


def psat(temp, average=True):
    """
    Function calculated the water saturation temperature or dew point for a given water vapor
    pressure. Water vapor pressure model created from an emperical fit of ln(Psat) vs
    temperature using a 6th order polynomial fit. The fit produced R^2=0.999813.
    Calculation created by Michael Kempe, unpublished data.

    Parameters:
    -----------
    temp : series, float
        The air temperature (dry bulb) as a time-indexed series [C]
    average : boolean, default = True
        If true, return both psat serires and average psat (used for certain calcs)
    Returns:
    --------
    psat : array, float
        Saturation point
    avg_psat : float, optional
        mean saturation point for the series given
    """

    psat = np.exp(
        (3.2575315268e-13 * temp**6)
        - (1.5680734584e-10 * temp**5)
        + (2.2213041913e-08 * temp**4)
        + (2.3720766595e-7 * temp**3)
        - (4.0316963015e-04 * temp**2)
        + (7.9836323361e-02 * temp)
        - (5.6983551678e-1)
    )
    if average:
        return psat, psat.mean()
    else:
        return psat


def surface_outside(rh_ambient, temp_ambient, temp_module):
    """
    Function calculates the Relative Humidity of a Solar Panel Surface at module temperature

    Parameters
    ----------
    rh_ambient : float
        The ambient outdoor environmnet relative humidity [%].
    temp_ambient : float
        The ambient outdoor environmnet temperature [°C]
    temp_module : float
        The surface temperature of the solar panel module [°C]

    Returns
    --------
    rh_Surface : float
        The relative humidity of the surface of a solar module as a fraction or percent depending on input.

    """
    rh_Surface = rh_ambient * (psat(temp_ambient)[0] / psat(temp_module)[0])

    return rh_Surface

    ###########
    # Front Encapsulant RH
    ###########


def _diffusivity_numerator(
    rh_ambient, temp_ambient, temp_module, So=1.81390702, Eas=16.729, Ead=38.14
):
    """
    Calculation is used in determining a weighted average Relative Humidity of the outside surface of a module.
    This funciton is used exclusively in the function _diffusivity_weighted_water and could be combined.

    The function returns values needed for the numerator of the Diffusivity weighted water
    content equation. This function will return a pandas series prior to summation of the
    numerator

    Parameters
    ----------
    rh_ambient : pandas series (float)
        The ambient outdoor environmnet relative humidity in [%]
        EXAMPLE: "50 = 50% NOT .5 = 50%"
    temp_ambient : pandas series (float)
        The ambient outdoor environmnet temperature [C]
    temp_module : pandas series (float)
        The surface temperature of the solar panel module [C]
    So : float
        Float, Encapsulant solubility prefactor in [g/cm3]
        So = 1.81390702(g/cm3) is the suggested value for EVA.
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729(kJ/mol) is the suggested value for EVA.
    Ead : float
        Encapsulant diffusivity activation energy in [kJ/mol]
        Ead = 38.14(kJ/mol) is the suggested value for EVA.

    Returns
    -------
    diff_numerator : pandas series (float)
        Nnumerator of the Sdw equation prior to summation

    """

    # Get the relative humidity of the surface
    rh_surface = surface_outside(rh_ambient, temp_ambient, temp_module)

    # Generate a series of the numerator values "prior to summation"
    diff_numerator = (
        So
        * np.exp(-(Eas / (0.00831446261815324 * (temp_module + 273.15))))
        * rh_surface
        * np.exp(-(Ead / (0.00831446261815324 * (temp_module + 273.15))))
    )

    return diff_numerator


def _diffusivity_denominator(temp_module, Ead=38.14):
    """
    Calculation is used in determining a weighted average Relative Humidity of the outside surface of a module.
    This funciton is used exclusively in the function _diffusivity_weighted_water and could be combined.

    The function returns values needed for the denominator of the Diffusivity
    weighted water content equation(diffuse_water). This function will return a pandas
    series prior to summation of the denominator

    Parameters
    ----------
    Ead : float
        Encapsulant diffusivity activation energy in [kJ/mol]
        38.14(kJ/mol) is the suggested value for EVA.
    temp_module : pandas series (float)
        The surface temperature in Celsius of the solar panel module

    Returns
    -------
    diff_denominator : pandas series (float)
        Denominator of the diffuse_water equation prior to summation

    """

    diff_denominator = np.exp(-(Ead / (0.00831446261815324 * (temp_module + 273.15))))
    return diff_denominator


def _diffusivity_weighted_water(
    rh_ambient, temp_ambient, temp_module, So=1.81390702, Eas=16.729, Ead=38.14
):
    """
    Calculation is used in determining a weighted average water content at the surface of a module.
    It is used as a constant water content that is equivalent to the time varying one with respect to moisture ingress.

    The function calculates the Diffusivity weighted water content.

    Parameters
    ----------
    rh_ambient : pandas series (float)
        The ambient outdoor environmnet relative humidity in (%)
        EXAMPLE: "50 = 50% NOT .5 = 50%"
    temp_ambient : pandas series (float)
        The ambient outdoor environmnet temperature in Celsius
    temp_module : pandas series (float)
        The surface temperature in Celsius of the solar panel module
    So : float
        Float, Encapsulant solubility prefactor in [g/cm3]
        So = 1.81390702(g/cm3) is the suggested value for EVA.
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729(kJ/mol) is the suggested value for EVA.
    Ead : float
        Encapsulant diffusivity activation energy in [kJ/mol]
        Ead = 38.14(kJ/mol) is the suggested value for EVA.

    Returns
    ------
    diffuse_water : float
        Diffusivity weighted water content

    """

    numerator = _diffusivity_numerator(
        rh_ambient, temp_ambient, temp_module, So, Eas, Ead
    )
    # get the summation of the numerator
    numerator = numerator.sum(axis=0, skipna=True)

    denominator = _diffusivity_denominator(temp_module, Ead)
    # get the summation of the denominator
    denominator = denominator.sum(axis=0, skipna=True)

    diffuse_water = (numerator / denominator) / 100

    return diffuse_water


def front_encap(rh_ambient, temp_ambient, temp_module, So=1.81390702, Eas=16.729):
    """
    Function returns a diffusivity weighted average Relative Humidity of the module surface.

    Parameters
    ----------
    rh_ambient : series (float)
        ambient Relative Humidity [%]
    temp_ambient : series (float)
        ambient outdoor temperature [°C]
    temp_module : pandas series (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    So : float
        Encapsulant solubility prefactor in [g/cm3]
        So = 1.81390702(g/cm3) is the suggested value for EVA.
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729(kJ/mol) is the suggested value for EVA.


    Return
    ------
    RHfront_series : pandas series (float)
        Relative Humidity of Frontside Solar module Encapsulant [%]

    """
    diffuse_water = _diffusivity_weighted_water(
        rh_ambient=rh_ambient, temp_ambient=temp_ambient, temp_module=temp_module
    )

    RHfront_series = (
        diffuse_water
        / (So * np.exp(-(Eas / (0.00831446261815324 * (temp_module + 273.15)))))
    ) * 100

    return RHfront_series

    ###########
    # Back Encapsulant Relative Humidity
    ###########


def _csat(temp_module, So=1.81390702, Eas=16.729):
    """
    Calculation is used in determining Relative Humidity of Backside Solar
    Module Encapsulant, and returns saturation of Water Concentration [g/cm³]

    Parameters
    -----------
    temp_module : pandas series (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    So : float
        Encapsulant solubility prefactor in [g/cm3]
        So = 1.81390702(g/cm3) is the suggested value for EVA.
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729(kJ/mol) is the suggested value for EVA.

    Returns
    -------
    Csat : pandas series (float)
        Saturation of Water Concentration [g/cm³]

    """

    # Saturation of water concentration
    Csat = So * np.exp(-(Eas / (0.00831446261815324 * (273.15 + temp_module))))

    return Csat


def _ceq(Csat, rh_SurfaceOutside):
    """
    Calculation is used in determining Relative Humidity of Backside Solar
    Module Encapsulant, and returns Equilibration water concentration (g/cm³)

    Parameters
    ------------
    Csat : pandas series (float)
        Saturation of Water Concentration (g/cm³)
    rh_SurfaceOutside : pandas series (float)
        The relative humidity of the surface of a solar module (%)

    Returns
    --------
    Ceq : pandas series (float)
        Equilibration water concentration (g/cm³)

    """

    Ceq = Csat * (rh_SurfaceOutside / 100)

    return Ceq


# @jit(nopython=True)
@njit
def Ce_numba(
    start,
    temp_module,
    rh_surface,
    WVTRo=7970633554,
    EaWVTR=55.0255,
    So=1.81390702,
    l=0.5,
    Eas=16.729,
):
    """
    Calculation is used in determining Relative Humidity of Backside Solar
    Module Encapsulant. This function returns a numpy array of the Concentration of water in the
    encapsulant at every time step

    Numba was used to isolate recursion requiring a for loop
    Numba Functions compile and run in machine code but can not use pandas (Very fast).

    Parameters
    -----------
    start : float
        Initial value of the Concentration of water in the encapsulant
        currently takes the first value produced from
        the _ceq(Saturation of Water Concentration) as a point
        of acceptable equilibrium
    temp_module : pandas series (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    rh_Surface : list (float)
        The relative humidity of the surface of a solar module [%]
        EXAMPLE: "50 = 50% NOT .5 = 50%"
    WVTRo : float
        Water Vapor Transfer Rate prefactor [g/m2/day].
        The suggested value for EVA is WVTRo = 7970633554(g/m2/day).
    EaWVTR : float
        Water Vapor Transfer Rate activation energy [kJ/mol] .
        It is suggested to use 0.15[mm] thick PET as a default
        for the backsheet and set EaWVTR=55.0255[kJ/mol]
    So : float
        Encapsulant solubility prefactor in [g/cm3]
        So = 1.81390702(g/cm3) is the suggested value for EVA.
    l : float
        Thickness of the backside encapsulant [mm].
        The suggested value for encapsulat is EVA l=0.5(mm)
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729[kJ/mol] is the suggested value for EVA.

    Returns
    --------
    Ce_list : numpy array
        Concentration of water in the encapsulant at every time step

    """

    dataPoints = len(temp_module)
    Ce_list = np.zeros(dataPoints)

    for i in range(0, len(rh_surface)):
        if i == 0:
            # Ce = Initial start of concentration of water
            Ce = start
        else:
            Ce = Ce_list[i - 1]

        Ce = Ce + (
            (
                WVTRo
                / 100
                / 100
                / 24
                * np.exp(
                    -((EaWVTR) / (0.00831446261815324 * (temp_module[i] + 273.15)))
                )
            )
            / (
                So
                * l
                / 10
                * np.exp(-((Eas) / (0.00831446261815324 * (temp_module[i] + 273.15))))
            )
            * (
                rh_surface[i]
                / 100
                * So
                * np.exp(-((Eas) / (0.00831446261815324 * (temp_module[i] + 273.15))))
                - Ce
            )
        )

        Ce_list[i] = Ce

    return Ce_list


def back_encap(
    rh_ambient,
    temp_ambient,
    temp_module,
    WVTRo=7970633554,
    EaWVTR=55.0255,
    So=1.81390702,
    l=0.5,
    Eas=16.729,
):
    """
    rh_back_encap()

    Function to calculate the Relative Humidity of Backside Solar Module Encapsulant
    and return a pandas series for each time step

    Parameters
    -----------
    rh_ambient : pandas series (float)
        The ambient outdoor environmnet relative humidity in [%]
        EXAMPLE: "50 = 50% NOT .5 = 50%"
    temp_ambient : pandas series (float)
        The ambient outdoor environmnet temperature in Celsius
    temp_module : list (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    WVTRo : float
        Water Vapor Transfer Rate prefactor [g/m2/day].
        The suggested value for EVA is WVTRo = 7970633554[g/m2/day].
    EaWVTR : float
        Water Vapor Transfer Rate activation energy [kJ/mol] .
        It is suggested to use 0.15[mm] thick PET as a default
        for the backsheet and set EaWVTR=55.0255[kJ/mol]
    So : float
        Encapsulant solubility prefactor in [g/cm3]
        So = 1.81390702[g/cm3] is the suggested value for EVA.
    l : float
        Thickness of the backside encapsulant [mm].
        The suggested value for encapsulat is EVA l=0.5[mm]
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729[kJ/mol] is the suggested value for EVA.

    Returns
    --------
    RHback_series : pandas series (float)
        Relative Humidity of Backside Solar Module Encapsulant [%]

    """

    rh_surface = surface_outside(
        rh_ambient=rh_ambient, temp_ambient=temp_ambient, temp_module=temp_module
    )

    Csat = _csat(temp_module=temp_module, So=So, Eas=Eas)
    Ceq = _ceq(Csat=Csat, rh_SurfaceOutside=rh_surface)

    start = Ceq.iloc[0]

    # Need to convert these series to numpy arrays for numba function
    temp_module_numba = temp_module.to_numpy()
    rh_surface_numba = rh_surface.to_numpy()
    Ce_nparray = Ce_numba(
        start=start,
        temp_module=temp_module_numba,
        rh_surface=rh_surface_numba,
        WVTRo=WVTRo,
        EaWVTR=EaWVTR,
        So=So,
        l=l,
        Eas=Eas,
    )

    # RHback_series = 100 * (Ce_nparray / (So * np.exp(-( (Eas) /
    #                   (0.00831446261815324 * (temp_module + 273.15))  )) ))
    RHback_series = 100 * (Ce_nparray / Csat)

    return RHback_series


def backsheet_from_encap(rh_back_encap, rh_surface_outside):
    """
    Function to calculate the Relative Humidity of solar module backsheet as timeseries.
    Requires the RH of the backside encapsulant and the outside surface of the module.

    Parameters
    ----------
    rh_back_encap : pandas series (float)
        Relative Humidity of Frontside Solar module Encapsulant. *See rh_back_encap()
    rh_surface_outside : pandas series (float)
        The relative humidity of the surface of a solar module. *See rh_surface_outside()

    Returns
    --------
    RHbacksheet_series : pandas series (float)
        Relative Humidity of Backside Backsheet of a Solar Module [%]
    """

    RHbacksheet_series = (rh_back_encap + rh_surface_outside) / 2

    return RHbacksheet_series


def backsheet(
    rh_ambient,
    temp_ambient,
    temp_module,
    WVTRo=7970633554,
    EaWVTR=55.0255,
    So=1.81390702,
    l=0.5,
    Eas=16.729,
):
    """Function to calculate the Relative Humidity of solar module backsheet as timeseries.

    Parameters
    ----------
    rh_ambient : pandas series (float)
        The ambient outdoor environmnet relative humidity in (%)
        EXAMPLE: "50 = 50% NOT .5 = 50%"
    temp_ambient : pandas series (float)
        The ambient outdoor environmnet temperature in Celsius
    temp_module : list (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    WVTRo : float
        Water Vapor Transfer Rate prefactor [g/m2/day].
        The suggested value for EVA is WVTRo = 7970633554[g/m2/day].
    EaWVTR : float
        Water Vapor Transfer Rate activation energy [kJ/mol] .
        It is suggested to use 0.15[mm] thick PET as a default
        for the backsheet and set EaWVTR=55.0255[kJ/mol]
    So : float
        Encapsulant solubility prefactor in [g/cm3]
        So = 1.81390702[g/cm3] is the suggested value for EVA.
    l : float
        Thickness of the backside encapsulant [mm].
        The suggested value for encapsulat is EVA l=0.5[mm]
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729[kJ/mol] is the suggested value for EVA.

    Returns
    --------
    rh_backsheet : float series or array
        relative humidity of the PV backsheet as a time-series [%]
    """

    RHback_series = back_encap(
        rh_ambient=rh_ambient,
        temp_ambient=temp_ambient,
        temp_module=temp_module,
        WVTRo=WVTRo,
        EaWVTR=EaWVTR,
        So=So,
        l=l,
        Eas=Eas,
    )
    surface = surface_outside(
        rh_ambient=rh_ambient, temp_ambient=temp_ambient, temp_module=temp_module
    )
    backsheet = (RHback_series + surface) / 2
    return backsheet


def module(
    weather_df,
    meta,
    tilt=None,
    azimuth=180,
    sky_model="isotropic",
    temp_model="sapm",
    conf="open_rack_glass_glass",
    WVTRo=7970633554,
    EaWVTR=55.0255,
    So=1.81390702,
    l=0.5,
    Eas=16.729,
    wind_factor=0.33,
):
    """Calculate the Relative Humidity of solar module backsheet from timeseries data.

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
        Options: 'sapm', 'pvsyst', 'faiman', 'sandia'.
    mount_type : str, optional
        Options: 'insulated_back_glass_polymer',
                 'open_rack_glass_polymer'
                 'close_mount_glass_glass',
                 'open_rack_glass_glass'
    WVTRo : float
        Water Vapor Transfer Rate prefactor (g/m2/day).
        The suggested value for EVA is WVTRo = 7970633554(g/m2/day).
    EaWVTR : float
        Water Vapor Transfer Rate activation energy (kJ/mol) .
        It is suggested to use 0.15(mm) thick PET as a default
        for the backsheet and set EaWVTR=55.0255(kJ/mol)
    So : float
        Encapsulant solubility prefactor in [g/cm3]
        So = 1.81390702(g/cm3) is the suggested value for EVA.
    l : float
        Thickness of the backside encapsulant (mm).
        The suggested value for encapsulat is EVA l=0.5(mm)
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729(kJ/mol) is the suggested value for EVA.
    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement heights
        between weather database (e.g. NSRDB) and the tempeature model (e.g. SAPM)
        The NSRDB provides calculations at 2 m (i.e module height) but SAPM uses a 10 m height.
        It is recommended that a power-law relationship between height and wind speed of 0.33
        be used. This results in a wind speed that is 1.7 times higher. It is acknowledged that
        this can vary significantly.

    Returns
    --------
    rh_backsheet : float series or array
        relative humidity of the PV backsheet as a time-series
    """

    # solar_position = spectral.solar_position(weather_df, meta)
    # poa = spectral.poa_irradiance(weather_df, meta, solar_position, tilt, azimuth, sky_model)
    # temp_module = temperature.module(weather_df, poa, temp_model, mount_type, wind_factor)

    poa = spectral.poa_irradiance(
        weather_df=weather_df,
        meta=meta,
        tilt=tilt,
        azimuth=azimuth,
        sky_model=sky_model,
    )

    temp_module = temperature.module(
        weather_df,
        meta,
        poa=poa,
        temp_model=temp_model,
        conf=conf,
        wind_factor=wind_factor,
    )

    rh_surface_outside = surface_outside(
        rh_ambient=weather_df["relative_humidity"],
        temp_ambient=weather_df["temp_air"],
        temp_module=temp_module,
    )

    rh_front_encap = front_encap(
        rh_ambient=weather_df["relative_humidity"],
        temp_ambient=weather_df["temp_air"],
        temp_module=temp_module,
        So=So,
        Eas=Eas,
    )

    rh_back_encap = back_encap(
        rh_ambient=weather_df["relative_humidity"],
        temp_ambient=weather_df["temp_air"],
        temp_module=temp_module,
        WVTRo=WVTRo,
        EaWVTR=EaWVTR,
        So=So,
        l=l,
        Eas=Eas,
    )

    rh_backsheet = backsheet_from_encap(
        rh_back_encap=rh_back_encap, rh_surface_outside=rh_surface_outside
    )

    data = {
        "RH_surface_outside": rh_surface_outside,
        "RH_front_encap": rh_front_encap,
        "RH_back_encap": rh_back_encap,
        "RH_backsheet": rh_backsheet,
    }
    results = pd.DataFrame(data=data)
    return results


@vectorize(['float64(float64, float64)'], nopython=True)
def water_vapor_pressure(
    temp: float,
    rh: float,
) -> float:
    """
    Calculate water vapor pressure at a temp and relative humidity pairing.

    Parameters:
    -----------
    temp: numeric
        temperature [C]
    rh: numeric
        percent relative humidity [%]

    Returns:
    -------
    water_vapor_pressure: numeric
        pressure of water vapor [kPa]
    """

    # where do these constants come from?
    water_vapor_pressure = (
        np.exp(
            -0.580109
            + 0.078001 * temp
            - 0.0003782525 * temp**2
            + 0.000001420179 * temp**3
            - 0.000000002447042 * temp**4
        )
        * rh
        / 100
    )

    return water_vapor_pressure


# need to modify in place
# this is badly behaving, returning zeros
def equilibrium_eva_water(
    sample_temp: np.ndarray,
    rh_at_sample_temp: np.ndarray,
    eva_solubility: float,
    solubility_prefactor: float,
):
    """
    Calculate Equlibirum EVA H20 content [g/cm^3]. Modify `result` array in place.

    Parameters:
    -----------
    sample_temp:
        temperature of the sample [C]
    rh_at_sample_temp:
        relative humidity at the sample temperature [%, unitless]
    eva_solubility:
        activation energy for solubility in EVA [eV]
    solubility_prefactor:
        amount of substance already present [g/cm^3]
        >>> should this just say water present at t=0
    """

    return (
        (
            solubility_prefactor
            * np.exp(-eva_solubility / 0.0000861733241 / (273.15 + sample_temp))
        )
        * rh_at_sample_temp
        / 100
    )


@njit
def _stable_min_steps(
    delta_t: float,
    backsheet_moisture: float,
    pet_prefactor: float,
    ea_pet: float,
    temperature: float,
    relative_humidity: float,
    equilibrium_eva_water: float,
    thickness_eva: float,  # [mm]
    thickness_pet: float,  # [mm]
) -> float:
    """
    Calculate the number of substeps required for a single step in the quasi steady state moisture eva back calculation to keep it stable.

    in the innner loop for the number of supsteps,
    we are saving and overwriting the substep values and only end up saving the last iteration
    using middle equation for spreadsheet
    dC/dt

    .. math::

        \frac{WVTR_{B,sat}}{C_{E,sat} L_{E}} \cdot \Delta t

    gives us a unitless number similar to a fourier number

    we want a value of < 0.25 to maintain stability within the solution
    """
    numerator = (
        (
            backsheet_moisture + pet_prefactor * np.exp((-ea_pet / temperature))
        )  # temperature needs to me in Kelvin
        * relative_humidity
        / 100
    )
    denominator = equilibrium_eva_water * thickness_eva * thickness_pet

    return numerator * delta_t / denominator


# qss calculation macro ported
# remove stable returns
def moisture_eva_back(
    eva_moisture_0: float,
    sample_temp: Union[pd.Series, np.ndarray],
    rh_at_sample_temp: Union[pd.Series, np.ndarray],
    equilibrium_eva_water: Union[pd.Series, np.ndarray],
    pet_permiability: float,
    pet_prefactor: float,
    thickness_eva: float,
    thickness_pet: float,
    n_steps: int,
) -> np.ndarray:
    if isinstance(sample_temp, pd.Series):
        sample_temp = sample_temp.to_numpy()
    if isinstance(rh_at_sample_temp, pd.Series):
        rh_at_sample_temp = rh_at_sample_temp.to_numpy()
    if isinstance(equilibrium_eva_water, pd.Series):
        equilibrium_eva_water = equilibrium_eva_water.to_numpy()

    sample_temp = np.add(sample_temp, 273.15)  # C -> K
    ea_pet = pet_permiability / 0.0000861733241  # boltzmann in eV/K

    moisture = np.empty_like(sample_temp)
    moisture[0] = eva_moisture_0

    stable = np.zeros_like(sample_temp)

    for i in range(1, sample_temp.shape[0]):
        # stable[i] = _stable_min_steps(
        #     delta_t=1, # 1 min,
        #     backsheet_moisture=moisture[i-1],
        #     temperature=sample_temp[i],
        #     relative_humidity=rh_at_sample_temp[i],
        #     equilibrium_eva_water=equilibrium_eva_water[i],
        #     ea_pet=ea_pet,
        #     pet_prefactor=pet_prefactor,
        #     thickness_eva=thickness_eva,
        #     thickness_pet=thickness_pet,
        # )

        moisture[i] = _calc_qss_substeps(
            moisture=moisture[i - 1],
            temperature=sample_temp[i],
            rh=rh_at_sample_temp[i],
            eq_eva_water=equilibrium_eva_water[i],
            thickness_eva=thickness_eva,
            thickness_pet=thickness_pet,
            pet_prefactor=pet_prefactor,
            ea_pet=ea_pet,
            n_steps=n_steps,
        )

    return moisture, stable


@njit
def _calc_qss_substeps(
    moisture,
    pet_prefactor,
    thickness_eva,
    thickness_pet,
    ea_pet,
    temperature,
    eq_eva_water,
    rh,
    n_steps,
) -> float:
    for _ in range(20):
        if 0 < moisture < 100:  # normal case
            moisture = (
                moisture
                + pet_prefactor
                / thickness_pet
                * np.exp(-ea_pet / temperature)
                / eq_eva_water
                * rh
                / 100
                / thickness_eva
                * (eq_eva_water - moisture)
                / 24000
                / n_steps
            )

        elif rh == 0 and eq_eva_water == 0:  # dry case
            moisture = (
                moisture
                + pet_prefactor
                / thickness_pet
                * np.exp(-ea_pet / temperature)
                / 100
                / thickness_eva
                * (eq_eva_water - moisture)
                / 24000
                / n_steps
            )

        else:  # superwet case rh >= 100
            moisture = (
                moisture
                + pet_prefactor
                / thickness_pet
                * np.exp(-ea_pet / temperature)
                / eq_eva_water
                / thickness_eva
                * (eq_eva_water - moisture)
                / 24000
                / n_steps
            )

    return moisture


@njit
def rh_internal_cell_backside(
    back_eva_moisture: Union[float, np.ndarray],
    equilibrium_eva_water: Union[float, np.ndarray],
    rh_at_sample_temp: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate relative humidity inside the module on the backside of the cells

    Parameters:
    -----------
    back_eva_moisture: Union[float, np.ndarray]
        back EVA moisture content [g/cm^3]
    equilibrium_eva_water: Union[float, np.ndarray]
        EVA equilibrium water content [g/cm^3]
    rh_at_sample_temp: Union[float, np.ndarray]
        relative humidity of sample in chamber [%, unitless]

    Returns:
    --------
    rh_internal_cell_backside: Union[float, np.ndarray]
        relative humidity inside the module on the backside of the cells [%, unitless]
    """

    return back_eva_moisture / equilibrium_eva_water * rh_at_sample_temp


# source of constants?
@njit
def chamber_dew_point_from_vapor_pressure(water_vap_pres: float) -> float:
    """
    Calculate chamber dew point from water vapor pressure.

    Parameters:
    -----------
    water_vap_pres: float
        chamber water vapor pressure [kPa]

    Returns:
    --------
    dew_point: float
        chamber dew point [C]
    """

    # where do these constants come from
    dew_point = (
        -0.000011449014849 * (np.log(water_vap_pres)) ** 6
        + 0.001637341324 * (np.log(water_vap_pres)) ** 5
        - 0.0077181540713 * (np.log(water_vap_pres)) ** 4
        + 0.045794594572 * (np.log(water_vap_pres)) ** 3
        + 1.1472781751 * (np.log(water_vap_pres)) ** 2
        + 13.892250408 * (np.log(water_vap_pres))
        + 7.1381806922
    )

    return dew_point


# where do these constants come from
@njit
def chamber_dew_point_from_t_rh(
    temp: float,
    rh: float,
) -> float:
    """
    Calculate chamber dew point from temperature and relative humidity

    Parameters:
    -----------
    temp: numeric
        temperature [C]
    rh: numeric
        percent relative humidity [%]

    Returns:
    --------
    dew_point: float
        chamber dew point [C]
    """

    water_vap_pressure = water_vapor_pressure(temp=temp, rh=rh)

    dew_point = chamber_dew_point_from_vapor_pressure(water_vap_pressure)

    return dew_point


# @njit
@vectorize(['float64(float64, float64, float64)'], nopython=True)
def rh_at_sample_temperature(
    temp_set: float,
    rh_set: float,
    sample_temp: float,
) -> float:
    """
    Calculate relative sample relative humidity using
    sample temperature and chamber set points

    Parameters:
    -----------
    temp_set: float
        temperature setpoint of chamber
        (not actual chamber air temp just an approximation)
    rh_set: float
        relative humidity setpoint of chamber
        (not actual chamber relative humidity)
    sample_temp: float
        temperature of the sample in the chamber

    Returns:
    --------
    rh: float
        relative humidity of sample in chamber (approx)
    """

    rh = (
        (
            np.exp(
                -0.580109
                + temp_set * 0.078001
                - 0.0003782525 * temp_set**2
                + 0.000001420179 * temp_set**3
                - 0.000000002447042 * temp_set**4
            )
            * rh_set
            / 100
        )
        / (
            np.exp(
                -0.580109
                + sample_temp * 0.078001
                - 0.0003782525 * sample_temp**2
                + 0.000001420179 * sample_temp**3
                - 0.000000002447042 * sample_temp**4
            )
        )
        * 100
    )

    return rh


def _calc_diff_substeps(
    water_new, water_old, n_steps, t, delta_t, dis, delta_dis, Fo
) -> None:  # inplace
    water_copy = water_old.copy()

    for _ in range(n_steps):
        for y in range(1, water_new.shape[0] - 1):
            # update the edge
            water_copy[0, :] = dis
            water_new[0, :] = dis + delta_dis  # one further, do we want this?

            water_new[-1, -1] = (
                Fo * (water_copy[-2, -1] * 4) + (1 - 4 * Fo) * water_copy[-1, -1]
            )  # inner node

            water_new[y, -1] = (
                Fo
                * (
                    water_copy[y + 1, -1]
                    + water_copy[y - 1, -1]
                    + 2 * water_copy[y, -2]
                )
                + (1 - 4 * Fo) * water_copy[y, -1]
            )  # internal edge

            for x in range(y + 1, water_copy.shape[1] - 1):
                neighbor_sum = (
                    water_copy[y - 1, x]
                    + water_copy[y + 1, x]
                    + water_copy[y, x - 1]
                    + water_copy[y, x + 1]
                )
                water_new[y, x] = (
                    Fo * neighbor_sum + (1 - 4 * Fo) * water_copy[y, x]
                )  # central nodes

            water_new[y, y] = (
                Fo * (2 * water_copy[y - 1, y] + 2 * water_copy[y, y + 1])
                + (1 - 4 * Fo) * water_copy[y, y]
            )  # diagonal

            dis += delta_dis
            t += delta_t  # do we need this one, temperature is not used anywhere?

            water_copy = water_new.copy()  # update copy so we can continue updating it


def module_front(
    time_index: pd.Index, # pandas index containing np.timedelta64
    backsheet_moisture: pd.Series, # g/m^3
    sample_temperature: pd.Series, # K
    p=0.1,  # [cm] perimiter area
    CW=15.6,  # [cm] cell dimensions
    nodes=20,  # number of nodes on each axis, square cell only
    eva_diffusivity_ea=0.395292897,  # eV
    Dif=2.31097881676966,  # cm^2/s diffusivity prefactor
    n_steps=20,  # number of substeps
) -> np.ndarray:
    """
    Calculate water intrusion into the front of the module using 2 dimensional finite difference method.

    Parameters:
    -----------
    time_index: pd.Index
        pandas instance with dtype, np.timedeltad64
    backsheet_moisture: pd.Series
        water content in the backsheet of a module [g/m^3]
    sample_temperature: pd.Series
        temperature of the module [K]
    p: float
        cell perimiter area [cm]
    CW: float
        cell edge dimension, only supports square modules [cm]
    nodes: int
        number of nodes to split each axis into for finite difference method analysis. higher is more accurate but slower. [unitless]
    eva_diffusivity_ea: float
        encapsulant diffusion activation energy [eV]
    Dif: float
        prefactor encapsulant diffusion [cm^2/s]
    n_steps: int
        number of stubsteps to calculate for numerical stability. 4-5 works for most cases but quick changes error can accumulate quickly so 20 is a good value for numerical safety.

    Returns:
    --------
    results: np.ndarray
        3d dimensional numpy array containing a 2 dimensional numpy matrix at each timestep corresponding to water intrusion. Shape (time_index.shape[0], nodes, nodes) [g/cm^3]
    """

    EaD = eva_diffusivity_ea / 0.0000861733241  # k in [eV/K]
    W = ((CW + 2 * p) / 2) / nodes  #

    # two options, we can have a 3d array that stores all timesteps
    results = np.zeros((len(time_index) + 1, nodes, nodes))
    results[0, 0, :] = backsheet_moisture.iloc[0]

    for i in range(
        len(time_index) - 1
    ):  # loop over each entry the the results
        Temperature = sample_temperature.iloc[i]
        DTemperature = (
            sample_temperature.iloc[i + 1]
            - sample_temperature.iloc[i]
        )
        Disolved = backsheet_moisture.iloc[i]
        DDisolved = (
            backsheet_moisture.iloc[i + 1]
            - backsheet_moisture.iloc[i]
        )

        time_step = (
            (
                time_index.values[i + 1]
                - time_index.values[i]
            )
            .astype("timedelta64[s]")
            .astype(int)
        )  # timestep in units of seconds
        Fo = Dif * np.exp(-EaD / (273.15 + Temperature)) * time_step / (W * W)

        _calc_diff_substeps(
            water_new=results[i + 1, :, :],
            water_old=results[i, :, :],
            n_steps=n_steps,
            t=Temperature,
            delta_t=DTemperature,
            dis=Disolved,
            delta_dis=DDisolved,
            Fo=Fo,
        )

    return results



# def run_module(
#     project_points,
#     out_dir,
#     tag,
#     weather_db,
#     weather_satellite,
#     weather_names,
#     max_workers=None,
#     tilt=None,
#     azimuth=180,
#     sky_model='isotropic',
#     temp_model='sapm',
#     mount_type='open_rack_glass_glass',
#     WVTRo=7970633554,
#     EaWVTR=55.0255,
#     So=1.81390702,
#     l=0.5,
#     Eas=16.729,
#     wind_factor=1
# ):

#     """Run the relative humidity calculation for a set of project points."""

#     #inputs
#     weather_arg = {}
#     weather_arg['satellite'] = weather_satellite
#     weather_arg['names'] = weather_names
#     weather_arg['NREL_HPC'] = True  #TODO: add argument or auto detect
#     weather_arg['attributes'] = [
#         'temp_air',
#         'wind_speed',
#         'dhi', 'ghi',
#         'dni','relative_humidity'
#         ]

#     #TODO: is there a better way to add the meta data?
#     nsrdb_fnames, hsds  = weather.get_NSRDB_fnames(
#         weather_arg['satellite'],
#         weather_arg['names'],
#         weather_arg['NREL_HPC'])

#     with NSRDBX(nsrdb_fnames[0], hsds=hsds) as f:
#         meta = f.meta[f.meta.index.isin(project_points.gids)]
#         ti = f.time_index

#     all_fields = ['RH_surface_outside',
#                 'RH_front_encap',
#                 'RH_back_encap',
#                 'RH_backsheet']

#     out_fp = Path(out_dir) / f"out_rel_humidity{tag}.h5"
#     shapes = {n : (len(ti), len(project_points)) for n in all_fields}
#     attrs = {n : {'units': '%'} for n in all_fields}
#     chunks = {n : None for n in all_fields}
#     dtypes = {n : "float32" for n in all_fields}

#     Outputs.init_h5(
#         out_fp,
#         all_fields,
#         shapes,
#         attrs,
#         chunks,
#         dtypes,
#         meta=meta.reset_index(),
#         time_index=ti
#     )

#     future_to_point = {}
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         for point in project_points:
#             gid = int(point.gid)
#             weather_df, meta = weather.load(
#                 database = weather_db,
#                 id = gid,
#                 **weather_arg)
#             future = executor.submit(
#                 module,
#                 weather_df,
#                 meta,
#                 tilt,
#                 azimuth,
#                 sky_model,
#                 temp_model,
#                 mount_type,
#                 WVTRo,
#                 EaWVTR,
#                 So,
#                 l,
#                 Eas,
#                 wind_factor
#             )
#             future_to_point[future] = gid

#         with Outputs(out_fp, mode="a") as out:
#             for future in as_completed(future_to_point):
#                 result = future.result()
#                 gid = future_to_point.pop(future)

#                 ind = project_points.index(gid)
#                 for dset, data in result.items():
#                     out[dset, :, ind] = data.values

#     return out_fp.as_posix()
