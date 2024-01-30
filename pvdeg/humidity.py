"""Collection of classes and functions for humidity calculations.
"""

import numpy as np
import pandas as pd
import pvlib
from numba import jit
from rex import NSRDBX
from rex import Outputs
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import temperature
from . import spectral
from . import weather


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
    dew_point = weather_df.get("temp_dew", weather_df.get("Dew Point"))

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


@jit(nopython=True)
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

    start = Ceq[0]

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
    mount_type="open_rack_glass_glass",
    WVTRo=7970633554,
    EaWVTR=55.0255,
    So=1.81390702,
    l=0.5,
    Eas=16.729,
    wind_speed_factor=1,
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
    wind_speed_factor : float, optional
        Wind speed correction factor to account for different wind speed measurement heights
        between weather database (e.g. NSRDB) and the tempeature model (e.g. SAPM)

    Returns
    --------
    rh_backsheet : float series or array
        relative humidity of the PV backsheet as a time-series
    """

    # solar_position = spectral.solar_position(weather_df, meta)
    # poa = spectral.poa_irradiance(weather_df, meta, solar_position, tilt, azimuth, sky_model)
    # temp_module = temperature.module(weather_df, poa, temp_model, mount_type, wind_speed_factor)

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
        conf=mount_type,
        wind_speed_factor=wind_speed_factor,
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
#     wind_speed_factor=1
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
#                 wind_speed_factor
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
