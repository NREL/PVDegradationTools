"""Collection of functions for degradation calculations.
"""

import numpy as np
import pandas as pd
from numba import jit
from rex import NSRDBX
from rex import Outputs
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import temperature
from . import spectral
from . import weather

# TODO: Clean up all those functions and add gaps functionality


def _deg_rate_env(poa_global, temp, temp_chamber, p, Tf):
    """
    Helper function. Find the rate of degradation kenetics using the Fischer model.
    Degradation kentics model interpolated 50 coatings with respect to
    color shift, cracking, gloss loss, fluorescense loss,
    retroreflectance loss, adhesive transfer, and shrinkage.

    (ADD IEEE reference)

    Parameters
    ------------
    poa_global : float
        (Global) Plan of Array irradiance [W/m²]
    temp : float
        Solar module temperature [°C]
    temp_chamber : float
        Reference temperature [°C] "Chamber Temperature"
    p : float
        Fit parameter
    Tf : float
        Multiplier for the increase in degradation
                                        for every 10[°C] temperature increase

    Returns
    --------
    degradationrate : float
        rate of Degradation (NEED TO ADD METRIC)

    """
    return poa_global ** (p) * Tf ** ((temp - temp_chamber) / 10)


def _deg_rate_chamber(I_chamber, p):
    """
    Helper function. Find the rate of degradation kenetics of a simulated chamber. Mike Kempe's
    calculation of the rate of degradation inside a accelerated degradation chamber.

    (ADD IEEE reference)

    Parameters
    ----------
    I_chamber : float
        Irradiance of Controlled Condition W/m²
    p : float
        Fit parameter

    Returns
    --------
    chamberdegradationrate : float
        Degradation rate of chamber
    """
    chamberdegradationrate = I_chamber ** (p)

    return chamberdegradationrate


def _acceleration_factor(numerator, denominator):
    """
    Helper Function. Find the acceleration factor

    (ADD IEEE reference)

    Parameters
    ----------
    numerator : float
        Typically the numerator is the chamber settings
    denominator : float
        Typically the TMY data summation

    Returns
    -------
    chamberAccelerationFactor : float
        Acceleration Factor of chamber (NEED TO ADD METRIC)
    """

    chamberAccelerationFactor = numerator / denominator

    return chamberAccelerationFactor


def vantHoff_deg(
    weather_df, meta, I_chamber, temp_chamber, poa=None, temp=None, p=0.5, Tf=1.41
):
    """
    Van't Hoff Irradiance Degradation

    Parameters
    -----------
    weather_df : pd.dataframe
        Dataframe containing at least dni, dhi, ghi, temperature, wind_speed
    meta : dict
        Location meta-data containing at least latitude, longitude, altitude
    I_chamber : float
        Irradiance of Controlled Condition [W/m²]
    temp_chamber : float
        Reference temperature [°C] "Chamber Temperature"
    poa : series or data frame, optional
        dataframe containing 'poa_global', Global Plane of Array Irradiance [W/m²]
    temp : pandas series, optional
        Solar module temperature or Cell temperature [°C]. If no cell temperature is given, it will
        be generated using the default paramters of pvdeg.temperature.cell
    p : float
        fit parameter
    Tf : float
        Multiplier for the increase in degradation for every 10[°C] temperature increase

    Returns
    -------
    accelerationFactor : float or series
        Degradation acceleration factor

    """

    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta)

    if isinstance(poa, pd.DataFrame):
        poa_global = poa["poa_global"]

    if temp is None:
        temp = temperature.cell(weather_df=weather_df, meta=meta, poa=poa)

    rateOfDegEnv = _deg_rate_env(
        poa_global=poa_global, temp=temp, temp_chamber=temp_chamber, p=p, Tf=Tf
    )
    # sumOfDegEnv = rateOfDegEnv.sum(axis = 0, skipna = True)
    avgOfDegEnv = rateOfDegEnv.mean()

    rateOfDegChamber = _deg_rate_chamber(I_chamber, p)

    accelerationFactor = _acceleration_factor(rateOfDegChamber, avgOfDegEnv)

    return accelerationFactor


def _to_eq_vantHoff(temp, Tf=1.41):
    """
    Function to obtain the Vant Hoff temperature equivalent [°C]

    Parameters
    ----------
    Tf : float
        Multiplier for the increase in degradation for every 10[°C] temperature increase. Default value of 1.41.
    temp : pandas series
        Solar module surface or Cell temperature [°C]

    Returns
    -------
    Toeq : float
        Vant Hoff temperature equivalent [°C]

    """

    toSum = Tf ** (temp / 10)
    summation = toSum.sum(axis=0, skipna=True)

    Toeq = (10 / np.log(Tf)) * np.log(summation / len(temp))

    return Toeq


def IwaVantHoff(weather_df, meta, poa=None, temp=None, Teq=None, p=0.5, Tf=1.41):
    """
    IWa : Environment Characterization [W/m²]
    For one year of degredation the controlled environmnet lamp settings will
    need to be set to IWa.

    Parameters
    -----------
    weather_df : pd.dataframe
        Dataframe containing at least dni, dhi, ghi, temperature, wind_speed
    meta : dict
        Location meta-data containing at least latitude, longitude, altitude
    poa : float series or dataframe
        Series or dataframe containing 'poa_global', Global Plane of Array Irradiance W/m²
    temp : float series
        Solar module temperature or Cell temperature [°C]
    Teq : series
        VantHoff equivalent temperature [°C]
    p : float
        Fit parameter
    Tf : float
        Multiplier for the increase in degradation for every 10[°C] temperature increase

    Returns
    --------
    Iwa : float
        Environment Characterization [W/m²[]

    """
    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta)

    if temp is None:
        temp = temperature.cell(weather_df, meta, poa)

    if Teq is None:
        Teq = _to_eq_vantHoff(temp, Tf)

    if isinstance(poa, pd.DataFrame):
        poa_global = poa["poa_global"]
    else:
        poa_global = poa

    toSum = (poa_global**p) * (Tf ** ((temp - Teq) / 10))
    summation = toSum.sum(axis=0, skipna=True)

    Iwa = (summation / len(poa_global)) ** (1 / p)

    return Iwa


def _arrhenius_denominator(poa_global, rh_outdoor, temp, Ea, p, n):
    """
    Helper function. Calculates the rate of degredation of the Environmnet

    Parameters
    ----------
    poa_global : float series
        (Global) Plan of Array irradiance [W/m²]
    p : float
        Fit parameter
    rh_outdoor : pandas series
        Relative Humidity of material of interest. Acceptable relative
        humiditys can be calculated from these functions: rh_backsheet(),
        rh_back_encap(); rh_front_encap();  rh_surface_outside()
    n : float
        Fit parameter for relative humidity
    temp : pandas series
        Solar module temperature or Cell temperature [°C]
    Ea : float
        Degredation Activation Energy [kJ/mol]

    Returns
    -------
    environmentDegradationRate : pandas series
        Degradation rate of environment
    """

    environmentDegradationRate = (
        poa_global ** (p)
        * rh_outdoor ** (n)
        * np.exp(-(Ea / (0.00831446261815324 * (temp + 273.15))))
    )

    return environmentDegradationRate


def _arrhenius_numerator(I_chamber, rh_chamber, temp_chamber, Ea, p, n):
    """
    Helper function. Find the rate of degradation of a simulated chamber.

    Parameters
    ----------
    I_chamber : float
        Irradiance of Controlled Condition [W/m²]
    Rhchamber : float
        Relative Humidity of Controlled Condition [%]
        EXAMPLE: "50 = 50% NOT .5 = 50%"
    temp_chamber : float
        Reference temperature [°C] "Chamber Temperature"
    Ea : float
        Degredation Activation Energy [kJ/mol]
    p : float
        Fit parameter
    n : float
        Fit parameter for relative humidity

    Returns
    --------
    arrheniusNumerator : float
        Degradation rate of the chamber
    """

    arrheniusNumerator = (
        I_chamber ** (p)
        * rh_chamber ** (n)
        * np.exp(-(Ea / (0.00831446261815324 * (temp_chamber + 273.15))))
    )
    return arrheniusNumerator


def arrhenius_deg(
    weather_df,
    meta,
    rh_outdoor,
    I_chamber,
    rh_chamber,
    Ea,
    temp_chamber,
    poa=None,
    temp=None,
    p=0.5,
    n=1,
):
    """
    Calculate the Acceleration Factor between the rate of degredation of a
    modeled environmnet versus a modeled controlled environmnet. Example: "If the AF=25 then 1 year
    of Controlled Environment exposure is equal to 25 years in the field"

    Parameters
    ----------
    weather_df : pd.dataframe
        Dataframe containing at least dni, dhi, ghi, temperature, wind_speed
    meta : dict
        Location meta-data containing at least latitude, longitude, altitude
    rh_outdoor : float series
        Relative Humidity of material of interest
        Acceptable relative humiditys can be calculated
        from these functions: rh_backsheet(), rh_back_encap(), rh_front_encap(),
        rh_surface_outside()
    I_chamber : float
        Irradiance of Controlled Condition [W/m²]
    rh_chamber : float
        Relative Humidity of Controlled Condition [%].
        EXAMPLE: "50 = 50% NOT .5 = 50%"
    temp_chamber : float
        Reference temperature [°C] "Chamber Temperature"
    Ea : float
        Degredation Activation Energy [kJ/mol]
        if Ea=0 is used there will be not dependence on temperature and degradation will proceed according to the amount of light and humidity.
    poa : pd.dataframe, optional
        Global Plane of Array Irradiance [W/m²]
    temp : pd.series, optional
        Solar module temperature or Cell temperature [°C]. If no cell temperature is given, it will
        be generated using the default parameters from pvdeg.temperature.cell
    p : float
        Fit parameter
        When p=0 the dependence on light will be ignored and degradation will happen both day an night. As a caution or a feature, a very small value of p (e.g. p=0.0001) will provide very little degradation dependence on irradiance, but degradation will only be accounted for during daylight. i.e. averages will be computed over half of the time only.
    n : float
        Fit parameter for relative humidity
        When n=0 the degradation rate will not be dependent on humidity.

    Returns
    --------
    accelerationFactor : pandas series
        Degradation acceleration factor

    """

    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta)

    if temp is None:
        temp = temperature.cell(weather_df, meta, poa)

    if isinstance(poa, pd.DataFrame):
        poa_global = poa["poa_global"]
    else:
        poa_global = poa

    arrheniusDenominator = _arrhenius_denominator(
        poa_global=poa_global, rh_outdoor=rh_outdoor, temp=temp, Ea=Ea, p=p, n=n
    )

    AvgOfDenominator = arrheniusDenominator.mean()

    arrheniusNumerator = _arrhenius_numerator(
        I_chamber=I_chamber,
        rh_chamber=rh_chamber,
        temp_chamber=temp_chamber,
        Ea=Ea,
        p=p,
        n=n,
    )

    accelerationFactor = _acceleration_factor(arrheniusNumerator, AvgOfDenominator)

    return accelerationFactor


def _T_eq_arrhenius(temp, Ea):
    """
    Get the Temperature equivalent required for the settings of the controlled environment
    Calculation is used in determining Arrhenius Environmental Characterization

    Parameters
    -----------
    temp : pandas series
        Solar module temperature or Cell temperature [°C]
    Ea : float
        Degredation Activation Energy [kJ/mol]

    Returns
    -------
    Teq : float
        Temperature equivalent (Celsius) required
        for the settings of the controlled environment

    """

    summationFrame = np.exp(-(Ea / (0.00831446261815324 * (temp + 273.15))))
    sumForTeq = summationFrame.sum(axis=0, skipna=True)
    Teq = -((Ea) / (0.00831446261815324 * np.log(sumForTeq / len(temp))))
    # Convert to celsius
    Teq = Teq - 273.15

    return Teq


def _RH_wa_arrhenius(rh_outdoor, temp, Ea, Teq=None, n=1):
    """
    NOTE

    Get the Relative Humidity Weighted Average.
    Calculation is used in determining Arrhenius Environmental Characterization

    Parameters
    -----------
    rh_outdoor : pandas series
        Relative Humidity of material of interest. Acceptable relative
        humiditys can be calculated from the below functions:
        rh_backsheet(), rh_back_encap(), rh_front_encap(), rh_surface_outside()
    temp : pandas series
        solar module temperature or Cell temperature [°C]
    Ea : float
        Degredation Activation Energy [kJ/mol]
    Teq : series
        Equivalent Arrhenius temperature [°C]
    n : float
        Fit parameter for relative humidity

    Returns
    --------
    RHwa : float
        Relative Humidity Weighted Average [%]

    """

    if Teq is None:
        Teq = _T_eq_arrhenius(temp, Ea)

    summationFrame = (rh_outdoor**n) * np.exp(
        -(Ea / (0.00831446261815324 * (temp + 273.15)))
    )
    sumForRHwa = summationFrame.sum(axis=0, skipna=True)
    RHwa = (
        sumForRHwa
        / (len(summationFrame) * np.exp(-(Ea / (0.00831446261815324 * (Teq + 273.15)))))
    ) ** (1 / n)

    return RHwa


# TODO:   CHECK
# STANDARDIZE
def IwaArrhenius(
    weather_df,
    meta,
    rh_outdoor,
    Ea,
    poa=None,
    temp=None,
    RHwa=None,
    Teq=None,
    p=0.5,
    n=1,
):
    """
    Function to calculate IWa, the Environment Characterization [W/m²].
    For one year of degredation the controlled environmnet lamp settings will
    need to be set at IWa.

    Parameters
    ----------
    weather_df : pd.dataframe
        Dataframe containing at least dni, dhi, ghi, temperature, wind_speed
    meta : dict
        Location meta-data containing at least latitude, longitude, altitude
    rh_outdoor : pd.series
        Relative Humidity of material of interest
        Acceptable relative humiditys include: rh_backsheet(), rh_back_encap(), rh_front_encap(),
        rh_surface_outside()
    Ea : float
        Degradation Activation Energy [kJ/mol]
    poa : pd.dataframe, optional
        must contain 'poa_global', Global Plan of Array irradiance [W/m²]
    temp : pd.series, optional
        Solar module temperature or Cell temperature [°C]
    RHwa : float, optional
        Relative Humidity Weighted Average [%]
    Teq : float, optional
        Temperature equivalent (Celsius) required
        for the settings of the controlled environment
    p : float
        Fit parameter
    n : float
        Fit parameter for relative humidity

    Returns
    --------
    Iwa : float
        Environment Characterization [W/m²]
    """
    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta)

    if temp is None:
        temp = temperature.cell(weather_df, meta, poa)

    if Teq is None:
        Teq = _T_eq_arrhenius(temp, Ea)

    if RHwa is None:
        RHwa = _RH_wa_arrhenius(rh_outdoor, temp, Ea)

    if isinstance(poa, pd.DataFrame):
        poa_global = poa["poa_global"]
    else:
        poa_global = poa

    numerator = (
        poa_global ** (p)
        * rh_outdoor ** (n)
        * np.exp(-(Ea / (0.00831446261815324 * (temp + 273.15))))
    )
    sumOfNumerator = numerator.sum(axis=0, skipna=True)

    denominator = (
        (len(numerator))
        * ((RHwa) ** n)
        * (np.exp(-(Ea / (0.00831446261815324 * (Teq + 273.15)))))
    )

    IWa = (sumOfNumerator / denominator) ** (1 / p)

    return IWa


############
# Misc. Functions for Energy Calcs
############


def _rh_Above85(rh):
    """
    Helper function. Determines if the relative humidity is above 85%.

    Parameters
    ----------
    rh : float
        Relative Humidity %

    Returns
    --------
    rhabove85 : boolean
        True if the relative humidity is above 85% or False if the relative
        humidity is below 85%

    """

    if rh > 85:
        rhabove85 = True

    else:
        rhabove85 = False

    return rhabove85


def _hoursRH_Above85(df):
    """
    Helper Function. Count the number of hours relative humidity is above 85%.

    Parameters
    ----------
    df : dataframe
        DataFrame, dataframe containing Relative Humidity %

    Returns
    -------
    numhoursabove85 : int
        Number of hours relative humidity is above 85%

    """
    booleanDf = df.apply(lambda x: _rh_Above85(x))
    numhoursabove85 = booleanDf.sum()

    return numhoursabove85


def _whToGJ(wh):
    """
    NOTE: unused, remove?

    Helper Function to convert Wh/m² to GJ/m²

    Parameters
    -----------
    wh : float
        Input Value in Wh/m²

    Returns
    -------
    gj : float
        Value in GJ/m²

    """

    gj = 0.0000036 * wh

    return gj


def _gJtoMJ(gJ):
    """
    NOTE: unused, remove?

    Helper Function to convert GJ/m² to MJ/y

    Parameters
    -----------
    gJ : float
        Value in GJ/m^-2

    Returns
    -------
    MJ : float
        Value in MJ/m^-2

    """
    MJ = gJ * 1000

    return MJ


def degradation(
    spectra, rh_module, temp_module, wavelengths, Ea=40.0, n=1.0, p=0.5, C2=0.07, C=1.0
):
    """
    Compute degredation as double integral of Arrhenius (Activation
    Energy, RH, Temperature) and spectral (wavelength, irradiance)
    functions over wavelength and time.

    Parameters
    ----------
    spectra : pd.Series type=Float
        front or rear irradiance at each wavelength in "wavelengths" [W/m^2 nm]
    rh_module : pd.Series type=Float
        module RH, time indexed [%]
    temp_module : pd.Series type=Float
        module temperature, time indexed [C]
    wavelengths : int-array
        integer array (or list) of wavelengths tested w/ uniform delta
        in nanometers [nm]
    Ea : float
        Arrhenius activation energy. The default is 40. [kJ/mol]
    n : float
        Fit paramter for RH sensitivity. The default is 1.
    p : float
        Fit parameter for irradiance sensitivity. Typically
        0.6 +- 0.22
    C2 : float
        Fit parameter for sensitivity to wavelength exponential.
        Typically 0.07
    C : float
        Fit parameter for the Degradation equaiton
        Typically 1.0

    Returns
    -------
    degradation : float
        Total degredation factor over time and wavelength.

    """
    # --- TO DO ---
    # unpack input-dataframe
    # spectra = df['spectra']
    # temp_module = df['temp_module']
    # rh_module = df['rh_module']

    # Constants
    R = 0.0083145  # Gas Constant in [kJ/mol*K]

    wav_bin = list(np.diff(wavelengths))
    wav_bin.append(wav_bin[-1])  # Adding a bin for the last wavelength

    # Integral over Wavelength
    try:
        irr = pd.DataFrame(spectra.tolist(), index=spectra.index)
        irr.columns = wavelengths
    except:
        # TODO: Fix this except it works on some cases, veto it by cases
        print("Removing brackets from spectral irradiance data")
        # irr = data['spectra'].str.strip('[]').str.split(',', expand=True).astype(float)
        irr = spectra.str.strip("[]").str.split(",", expand=True).astype(float)
        irr.columns = wavelengths

    sensitivitywavelengths = np.exp(-C2 * wavelengths)
    irr = irr * sensitivitywavelengths
    irr *= np.array(wav_bin)
    irr = irr**p
    data = pd.DataFrame(index=spectra.index)
    data["G_integral"] = irr.sum(axis=1)

    EApR = -Ea / R
    C4 = np.exp(EApR / temp_module)

    RHn = rh_module**n
    data["Arr_integrand"] = C4 * RHn

    data["dD"] = data["G_integral"] * data["Arr_integrand"]

    degradation = C * data["dD"].sum(axis=0)

    return degradation
