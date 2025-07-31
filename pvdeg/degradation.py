"""Collection of functions for degradation calculations."""

import numpy as np
import pandas as pd
from numba import njit
from typing import Union
from pvdeg import humidity

from . import (
    temperature,
    spectral,
    decorators,
)

# TODO: Clean up all those functions and add gaps functionality

def arrhenius(
    weather_df=None,
    temperature=None,
    RH=None,
    irradiance=None,
    Ro=None,
    Ea=None,
    p=None,
    n=None,
    C2=None,
    parameters=None
    ):

    """
    Calculate the degradation rate using an Arrhenius function with power law functions for
    the humidity dependence and irradiance dependence. 
    D=R_0 ∫[RH(t)]^n·e^[-E_a/RT(t) ] {∫[e^(-C_2∙ λ)∙G(λ,t)]^p dλ}dt

    Parameters
    ----------
    weather_df : pd.dataframe
        Dataframe containing a the temperature, humidity, and irradiance data as an alternate source. 
        If used, it will default to the module surface temperature, the module surface humidity,
        and the POA global irradiance.
    temperature: pd.DataFrame,
        This is the temperature data that will be used for the Arrhenius degradation calculation. 
        If it isn't specified, the module surface temperature will be used as supplied in weather_df.
        if Ea=0 then temperature data will not be necessary.
    RH: pd.DataFrame,
        This is the relative humidity data that will be used for the Arrhenius degradation calculation. 
        If it isn't specified, the module surface relative humidity will be used as determined using weather_df.
        The module temperature is not in weather_df by default and must therfore be added prior to this function call
        If the humidity power law factor n is not provided, n will be assumed to be equal to zero and humidity
        data is not needed.
    irradiance: pd.DataFrame,
        This is the irradiance data that will be used for the Arrhenius degradation calculation. 
        If it isn't specified, the module POA irradiance will be used as supplied in weather_df.
        If the irradiance power law factor (Schwarchild coefficient) p is not provided, p will be assumed 
        to be equal to zero and irradiance data is not needed. 

        If C2 is provided, then the wavelength spectral intensity data must be provided. Here the header 
        is a list with the first element starting with the word "spectra", and the rest of the elements 
        being the wavelength irradiance intensity points. Then each element in the dataframe is a list
        of light intensity values at the corresponding wavelength in units of W/m²/nm, or similar.  
    Ro : float
        This is the degradation rate prefactor with units determined by the user [e.g. %/h/%RH/(1000 W/m²)]
        if not provided direction or through parameters, a value of 1 [%/h/(%^(1/n)/(w^(1/p)))] will be used.
    Ea : float
        Degredation Activation Energy [kJ/mol]
        if Ea=0 is used there will be not dependence on temperature and degradation will proceed according 
        to the amount of light and humidity.
    p : float
        Schwartchild coefficient for power law dependence on irradiance.
        When p=0 the dependence on light will be ignored and degradation will happen both day an night. 
        As a caution or a feature, a very small value of p (e.g. p=0.0001) will provide very little degradation 
        dependence on irradiance, but degradation will only be accounted for during daylight. i.e. averages will 
        be computed over half of the time only.
    n : float
        Parameter for relative humidity power law dependence on degradation.
        When n=0 the degradation rate will not be dependent on humidity.
    C2 : float
        Parameter for describing the spectral response of the module using a power law relationship of irradiance
        to photon wavelength.
    parameters : json
        This is a json file that contains the parameters for the Arrhenius degradation calculation.
        If Ea, n or p are not provided for the calculation, they will be taken from this json file.
    
    Returns
    --------
    degradation : float
        Total degradation with units as determined by Ro
    """

    if Ro==None:
        if parameters is not None:
            if "R_0.value" in parameters:
                Ro = parameters["R_0.value"]
            else:
                Ro = 1
        else:
            Ro = 1
    if Ea==None:
        if parameters is not None:
            if "Ea.value" in parameters:
                Ea = parameters["Ea.value"]
            else:
                Ea=0
        else:
            Ea=0
    if n==None:
        if parameters is not None:
            if "n.value" in parameters:
                n = parameters["n.value"]
            else:
                n=0
        else:
            n=0
    if p==None:
        if parameters is not None:
            if "p.value" in parameters:
                p = parameters["p.value"]
            else:
                p=0
        else:
            p=0
    if temperature is None:
        temperature = weather_df["temp"]
    if RH==None and "relative_humidity" in weather_df and "temp_air" in weather_df and "temp_module" in weather_df:
        RH = humidity.surface_outside(weather_df["relative_humidity"], weather_df["temp_air"], weather_df["temp_module"])

    if C2==None:
        if parameters is not None:
            if "C_2.value" in parameters:
                C2 = parameters["C_2.value"]
            else:
                C2 = 0
        else:
            C2 = 0  
    if irradiance is None:
        if C2 !=0 or p !=0:
            if weather_df is not None:
                for col in weather_df.columns:
                    if "SPECTRA" in (col[:7]).upper():
                        irradiance = weather_df[col].copy
                        irradiance.columns = [col]
                        break  
                if "poa_global" in weather_df:
                    irradiance = weather_df["poa_global"]
                    print("Using poa_global from weather_df for irradiance.")   

    if C2 !=0:
        wavelengths = [float(i) for i in irradiance.columns[0].split("[")[1].split("]")[0].split(",")]
        wavelengths = np.array(wavelengths)
        bin_widths = (np.append(wavelengths,[0,0])- np.append([0,0],wavelengths))/2
        bin_widths = bin_widths[1:]
        bin_widths = bin_widths[:-1]
        #assumes the first and last bin widths are the width of that between the next or previous bin, respectively.
        bin_widths[0] = bin_widths[1]
        bin_widths[-1] = bin_widths[-2]
        
        if p==0:
            if Ea!=0:
                if n==0:
                    degradation = Ro * np.exp(-(Ea / (0.00831446261815324 * (temperature + 273.15))))
                else: 
                    degradation = Ro * np.exp(-(Ea / (0.00831446261815324 * (temperature + 273.15)))) * (RH**n)
            else:   
                if n==0:
                    degradation = Ro
                else: 
                    degradation = Ro * (RH**n)
        else:
            degradation = bin_widths * ((np.exp(-C2*wavelengths)*irradiance)**p) 
            if Ea!=0:
                if n==0:
                    degradation = degradation * Ro * np.exp(-(Ea / (0.00831446261815324 * (temperature + 273.15))))  
                else:
                    degradation = degradation * Ro * np.exp(-(Ea / (0.00831446261815324 * (temperature + 273.15)))) * (RH**n)    
            else:   
                if n==0:
                    degradation = degradation * Ro 
                else:
                    degradation = degradation * Ro * (RH**n) 
    elif Ea!=0:
        if n==0 and p==0:
            degradation = Ro * np.exp(-(Ea / (0.00831446261815324 * (temperature + 273.15))))
        elif n==0 and p!=0:
            degradation = Ro * np.exp(-(Ea / (0.00831446261815324 * (temperature + 273.15)))) * (irradiance**p) 
        elif n!=0 and p==0: 
            degradation = Ro * np.exp(-(Ea / (0.00831446261815324 * (temperature + 273.15)))) * (RH**n)
        else:
            degradation = Ro * np.exp(-(Ea / (0.00831446261815324 * (temperature + 273.15)))) * (RH**n) * (irradiance**p)   
    else:   
        if n==0 and p==0:
            degradation = Ro
        elif n==0 and p!=0:
            degradation = Ro * (irradiance**p) 
        elif n!=0 and p==0: 
            degradation = Ro * (RH**n)
        else:
            degradation = Ro * (RH**n) * (irradiance**p)

    return degradation.sum(axis=0, skipna=True)



def _deg_rate_env(poa_global, temp, temp_chamber, p, Tf):
    """Find degradation rate, helper function.

    Find the rate of degradation kenetics using the Fischer model.
    Degradation kentics model interpolated 50 coatings with respect to color shift,
    cracking, gloss loss, fluorescense loss, retroreflectance loss, adhesive transfer,
    and shrinkage.

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
    -------/
    degradationrate : float
        rate of Degradation (NEED TO ADD METRIC)
    """
    # poa_global ** (p) * Tf ** ((temp - temp_chamber) / 10)
    return np.multiply(
        np.power(poa_global, p),
        np.power(Tf, np.divide(np.subtract(temp, temp_chamber), 10)),
    )


def _deg_rate_chamber(I_chamber, p):
    """Calculate simulated chamber degredation rate, helper function.

    Find the rate of degradation kenetics of a simulated chamber.
    Mike Kempe's calculation of the rate of degradation inside a accelerated degradation
    chamber.

    (ADD IEEE reference)

    Parameters
    ----------
    I_chamber : float
        Irradiance of Controlled Condition W/m²
    p : float
        Fit parameter

    Returns
    -------
    chamberdegradationrate : float
        Degradation rate of chamber
    """
    # chamberdegradationrate = I_chamber ** (p)
    chamberdegradationrate = np.power(I_chamber, p)

    return chamberdegradationrate


def _acceleration_factor(numerator, denominator):
    """Calculate the acceleration factor, helper Function.

    Find the acceleration factor.

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
    chamberAccelerationFactor = np.divide(numerator, denominator)
    # chamberAccelerationFactor = numerator / denominator

    return chamberAccelerationFactor


def vantHoff_deg(
    weather_df,
    meta,
    I_chamber,
    temp_chamber,
    poa=None,
    temp=None,
    p=0.5,
    Tf=1.41,
    temp_model="sapm",
    conf="open_rack_glass_polymer",
    wind_factor=0.33,
    irradiance_kwarg={},
    model_kwarg={},
):
    """Van't Hoff Irradiance Degradation.

    Parameters
    ----------
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
        Solar module temperature or Cell temperature [°C]. If no cell temperature is
        given, it will be generated using the default parameters of
        pvdeg.temperature.cell
    p : float
        fit parameter
    Tf : float
        Multiplier for the increase in degradation for every 10[°C] temperature increase
    temp_model : (str, optional)
        Specify which temperature model from pvlib to use. Current options:
    conf : (str)
        The configuration of the PV module architecture and mounting
        configuration. Currently only used for 'sapm' and 'pvsys'.
        With different options for each.

        'sapm' options: ``open_rack_glass_polymer`` (default),
        ``open_rack_glass_glass``, ``close_mount_glass_glass``,
        ``insulated_back_glass_polymer``

        'pvsys' options: ``freestanding``, ``insulated``

    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement
        heights between weather database (e.g. NSRDB) and the tempeature model
        (e.g. SAPM)
        The NSRDB provides calculations at 2 m (i.e module height) but SAPM uses a 10m
        height.
        It is recommended that a power-law relationship between height and wind speed
        of 0.33 be used*. This results in a wind speed that is 1.7 times higher. It is
        acknowledged that this can vary significantly.
    irradiance_kwarg : (dict, optional)
        keyword argument dictionary used for the poa irradiance caluation.
        options: ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``.
        See ``pvdeg.spectral.poa_irradiance``.
    model_kwarg : (dict, optional)
        keyword argument dictionary used for the pvlib temperature model calculation.
        See https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html # noqa
        or more.

    Returns
    -------
    accelerationFactor : float or series
        Degradation acceleration factor
    """
    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta, **irradiance_kwarg)

    if isinstance(poa, pd.DataFrame):
        poa_global = poa["poa_global"]

    if temp is None:
        # temp = temperature.cell(weather_df=weather_df, meta=meta, poa=poa)
        temp = temperature.temperature(
            cell_or_mod="cell",
            temp_model=temp_model,
            weather_df=weather_df,
            meta=meta,
            poa=poa,
            conf=conf,
            wind_factor=wind_factor,
            model_kwarg=model_kwarg,
        )

    rateOfDegEnv = _deg_rate_env(
        poa_global=poa_global, temp=temp, temp_chamber=temp_chamber, p=p, Tf=Tf
    )
    # sumOfDegEnv = rateOfDegEnv.sum(axis = 0, skipna = True)
    avgOfDegEnv = rateOfDegEnv.mean()

    rateOfDegChamber = _deg_rate_chamber(I_chamber, p)

    accelerationFactor = _acceleration_factor(rateOfDegChamber, avgOfDegEnv)

    return accelerationFactor


def _to_eq_vantHoff(temp, Tf=1.41):
    """Obtain the Vant Hoff temperature equivalent [°C].

    Parameters
    ----------
    Tf : float
        Multiplier for the increase in degradation for every 10[°C] temperature
        increase. Default value of 1.41.
    temp : pandas series
        Solar module surface or Cell temperature [°C]

    Returns
    -------
    Toeq : float
        Vant Hoff temperature equivalent [°C]
    """
    # toSum = Tf ** (temp / 10)
    toSum = np.power(Tf, np.divide(temp, 10))
    summation = toSum.sum(axis=0, skipna=True)

    Toeq = (10 / np.log(Tf)) * np.log(summation / len(temp))

    return Toeq


@decorators.geospatial_quick_shape("numeric", ["Iwa"])
def IwaVantHoff(
    weather_df,
    meta,
    poa=None,
    temp=None,
    Teq=None,
    p=0.5,
    Tf=1.41,
    temp_model="sapm",
    conf="open_rack_glass_polymer",
    wind_factor=0.33,
    model_kwarg={},
    irradiance_kwarg={},
):
    """
    Environment characterization function.

    IWa : Environment Characterization [W/m²].

    For one year of degredation the controlled environmnet lamp settings will
    need to be set to IWa.

    Parameters
    -----------
    weather_df : pd.dataframe
        Dataframe containing at least dni, dhi, ghi, temperature, wind_speed
    meta : dict
        Location meta-data containing at least latitude, longitude, altitude
    poa : float series or dataframe
        Series or dataframe containing 'poa_global', Global Plane of Array Irradiance
        [W/m²]
    temp : float series
        Solar module temperature or Cell temperature [°C]
    Teq : series
        VantHoff equivalent temperature [°C]
    p : float
        Fit parameter
    Tf : float
        Multiplier for the increase in degradation for every 10[°C] temperature increase
    temp_model : (str, optional)
        Specify which temperature model from pvlib to use. Current options:
    conf : (str)
        The configuration of the PV module architecture and mounting
        configuration. Currently only used for 'sapm' and 'pvsys'.
        With different options for each.

        'sapm' options: ``open_rack_glass_polymer`` (default),
        ``open_rack_glass_glass``, ``close_mount_glass_glass``,
        ``insulated_back_glass_polymer``

        'pvsys' options: ``freestanding``, ``insulated``

    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement
        heights between weather database (e.g. NSRDB) and the tempeature model
        (e.g. SAPM)
        The NSRDB provides calculations at 2 m (i.e module height) but SAPM uses a 10m
        height.
        It is recommended that a power-law relationship between height and wind speed of
        0.33 be used*. This results in a wind speed that is 1.7 times higher. It is
        acknowledged that this can vary significantly.
    irradiance_kwarg : (dict, optional)
        keyword argument dictionary used for the poa irradiance caluation.
        options: ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``.
        ee ``pvdeg.spectral.poa_irradiance``.
    model_kwarg : (dict, optional)
        keyword argument dictionary used for the pvlib temperature model calculation.
        See https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html # noqa
        for more.

    Returns
    --------
    Iwa : float
        Environment Characterization [W/m²]
    """
    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta, **irradiance_kwarg)

    if temp is None:
        # temp = temperature.cell(weather_df, meta, poa)
        temp = temperature.temperature(
            cell_or_mod="cell",
            temp_model=temp_model,
            weather_df=weather_df,
            meta=meta,
            poa=poa,
            conf=conf,
            wind_factor=wind_factor,
            model_kwarg=model_kwarg,
        )

    if Teq is None:
        Teq = _to_eq_vantHoff(temp, Tf)

    if isinstance(poa, pd.DataFrame):
        poa_global = poa["poa_global"]
    else:
        poa_global = poa

    # toSum = (poa_global**p) * (Tf ** ((temp - Teq) / 10))
    toSum = np.multiply(
        np.power(poa_global, p), np.power(Tf, np.divide(np.subtract(temp, Teq), 10))
    )

    summation = toSum.sum(axis=0, skipna=True)

    # Iwa = (summation / len(poa_global)) ** (1 / p)
    Iwa = np.power(np.divide(summation, len(poa_global)), np.divide(1, p))

    return Iwa


def _arrhenius_denominator(poa_global, rh_outdoor, temp, Ea, p, n):
    """Calculate environment degradation rate, helper function.

    Calculates the rate of degredation of the Environmnet.

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
    # environmentDegradationRate = (
    #     poa_global ** (p)
    #     * rh_outdoor ** (n)
    #     * np.exp(-(Ea / (0.00831446261815324 * (temp + 273.15))))
    # )

    environmentDegradationRate = np.multiply(
        np.multiply(np.power(poa_global, p), np.power(rh_outdoor, n)),
        np.exp(
            np.negative(
                np.divide(Ea, np.multiply(0.00831446261815324, np.add(temp, 273.15)))
            )
        ),
    )

    return environmentDegradationRate


def _arrhenius_numerator(I_chamber, rh_chamber, temp_chamber, Ea, p, n):
    """Calculate degradation rate, helper function.

    Find the rate of degradation of a simulated chamber.

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
    # arrheniusNumerator = (
    #     I_chamber ** (p)
    #     * rh_chamber ** (n)
    #     * np.exp(-(Ea / (0.00831446261815324 * (temp_chamber + 273.15))))
    # )

    arrheniusNumerator = np.multiply(
        np.multiply(np.power(I_chamber, p), np.power(rh_chamber, n)),
        np.exp(
            np.negative(
                np.divide(
                    Ea, np.multiply(0.00831446261815324, np.add(temp_chamber, 273.15))
                )
            )
        ),
    )

    return arrheniusNumerator


def arrhenius_deg(
    weather_df: pd.DataFrame,
    meta: dict,
    rh_outdoor,
    I_chamber,
    rh_chamber,
    Ea,
    temp_chamber,
    poa=None,
    temp=None,
    p=0.5,
    n=1,
    temp_model="sapm",
    conf="open_rack_glass_polymer",
    wind_factor=0.33,
    model_kwarg={},
    irradiance_kwarg={},
):
    """Calculate the Acceleration Factor between the rate of degredation of a modeled.

    environmnet versus a modeled controlled environmnet. Example: "If the AF=25 then 1
    year of Controlled Environment exposure is equal to 25 years in the field".

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
        if Ea=0 is used there will be not dependence on temperature and degradation will
        proceed according to the amount of light and humidity.
    poa : pd.dataframe, optional
        Global Plane of Array Irradiance [W/m²]
    temp : pd.series, optional
        Solar module temperature or Cell temperature [°C]. If no cell temperature is
        given, it will be generated using the default parameters from
        pvdeg.temperature.cell
    p : float
        Fit parameter
        When p=0 the dependence on light will be ignored and degradation will happen
        both day an night. As a caution or a feature, a very small value of p
        (e.g. p=0.0001) will provide very little degradation dependence on irradiance,
        but degradation will only be accounted for during daylight. i.e. averages will
        be computed over half of the time only.
    n : float
        Fit parameter for relative humidity
        When n=0 the degradation rate will not be dependent on humidity.
    temp_model : (str, optional)
        Specify which temperature model from pvlib to use. Current options:
    conf : (str)
        The configuration of the PV module architecture and mounting
        configuration. Currently only used for 'sapm' and 'pvsys'.
        With different options for each.

        'sapm' options: ``open_rack_glass_polymer`` (default),
        ``open_rack_glass_glass``, ``close_mount_glass_glass``,
        ``insulated_back_glass_polymer``

        'pvsys' options: ``freestanding``, ``insulated``

    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement
        heights between weather database (e.g. NSRDB) and the tempeature model
        (e.g. SAPM)
        The NSRDB provides calculations at 2 m (i.e module height) but SAPM uses a 10m
        height.
        It is recommended that a power-law relationship between height and wind speed of
        0.33 be used*. This results in a wind speed that is 1.7 times higher. It is
        acknowledged that this can vary significantly.
    irradiance_kwarg : (dict, optional)
        keyword argument dictionary used for the poa irradiance caluation.
        options: ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``. See
        ``pvdeg.spectral.poa_irradiance``.
    model_kwarg : (dict, optional)
        keyword argument dictionary used for the pvlib temperature model calculation.
        See https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html # noqa
        for more.

    Returns
    --------
    accelerationFactor : pandas series
        Degradation acceleration factor
    """
    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta, **irradiance_kwarg)

    if temp is None:
        # temp = temperature.cell(weather_df, meta, poa)
        temp = temperature.temperature(
            cell_or_mod="cell",
            temp_model=temp_model,
            weather_df=weather_df,
            meta=meta,
            poa=poa,
            conf=conf,
            wind_factor=wind_factor,
            model_kwarg=model_kwarg,
        )

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
    """Calculate temperature for Arrhenius Environmental Characterization.

    Calculate temperature equivalent required for the settings of the controlled
    environment Calculation is used in determining Arrhenius Environmental
    Characterization.

    Parameters
    ----------
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
    """NOTE.

    Get the Relative Humidity Weighted Average.
    Calculation is used in determining Arrhenius Environmental Characterization

    Parameters
    ----------
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
    -------
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
    weather_df: pd.DataFrame,
    meta: dict,
    rh_outdoor: pd.Series,
    Ea: float,
    poa: pd.DataFrame = None,
    temp: pd.Series = None,
    RHwa: float = None,
    Teq: float = None,
    p: float = 0.5,
    n: float = 1,
    temp_model="sapm",
    conf="open_rack_glass_polymer",
    wind_factor=0.33,
    model_kwarg={},
    irradiance_kwarg={},
) -> float:
    """Calculate IWa, the Environment Characterization [W/m²].

    For one year
    of degredation the controlled environmnet lamp settings will need to be set at IWa.

    Parameters
    ----------
    weather_df : pd.dataframe
        Dataframe containing at least dni, dhi, ghi, temperature, wind_speed
    meta : dict
        Location meta-data containing at least latitude, longitude, altitude
    rh_outdoor : pd.series
        Relative Humidity of material of interest
        Acceptable relative humiditys include: rh_backsheet(), rh_back_encap(),
        rh_front_encap(), rh_surface_outside()
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
    temp_model : (str, optional)
        Specify which temperature model from pvlib to use. Current options:
    conf : (str)
        The configuration of the PV module architecture and mounting
        configuration. Currently only used for 'sapm' and 'pvsys'.
        With different options for each.

        'sapm' options: ``open_rack_glass_polymer`` (default),
        ``open_rack_glass_glass``, ``close_mount_glass_glass``,
        ``insulated_back_glass_polymer``

        'pvsys' options: ``freestanding``, ``insulated``

    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement
        heights between weather database (e.g. NSRDB) and the tempeature model
        (e.g. SAPM). The NSRDB provides calculations at 2 m (i.e module height) but SAPM
        uses a 10m height. It is recommended that a power-law relationship between
        height and wind speed of 0.33 be used*. This results in a wind speed that is
        1.7 times higher. It is acknowledged that this can vary significantly.
    irradiance_kwarg : (dict, optional)
        keyword argument dictionary used for the poa irradiance caluation.
        options: ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``. See
        ``pvdeg.spectral.poa_irradiance``.
    model_kwarg : (dict, optional)
        keyword argument dictionary used for the pvlib temperature model calculation.
        See https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html # noqa
        for more.

    Returns
    --------
    Iwa : float
        Environment Characterization [W/m²]
    """
    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta, **irradiance_kwarg)

    if temp is None:
        # temp = temperature.cell(weather_df, meta, poa)
        temp = temperature.temperature(
            cell_or_mod="cell",
            temp_model=temp_model,
            weather_df=weather_df,
            meta=meta,
            poa=poa,
            conf=conf,
            wind_factor=wind_factor,
            model_kwarg=model_kwarg,
        )

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
    """Calculate whether RH>85%, helper function.

    Determines if the relative humidity is above 85%.

    Parameters
    ----------
    rh : float
        Relative Humidity %

    Returns
    -------
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
    """Count hours RH>85%, helper Function.

    Count the number of hours relative humidity is above 85%.

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
    NOTE: unused, remove(?).

    Helper Function to convert Wh/m² to GJ/m²

    Parameters
    ----------
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
    NOTE: unused, remove(?).

    Helper Function to convert GJ/m² to MJ/y

    Parameters
    ----------
    gJ : float
        Value in GJ/m^-2

    Returns
    -------
    MJ : float
        Value in MJ/m^-2
    """
    MJ = gJ * 1000

    return MJ


def degradation_spectral(
    spectra: pd.Series,
    rh: pd.Series,
    temp: pd.Series,
    wavelengths: Union[int, np.ndarray[float]],
    time: pd.Series,
    Ea: float = 0.0,
    n: float = 0.0,
    p: float = 0.6,
    C2: float = 0.07,
    R_0: float = 1.0,
) -> float:
    """Compute degredation as double integral of Arrhenius (Activation Energy, RH,.

    Temperature) and spectral (wavelength, irradiance) functions over wavelength and
    time.

    Parameters
    ----------
    spectra : pd.Series type=Float
        front or rear irradiance at each wavelength in "wavelengths" [W/m^2 nm]
    rh : pd.Series type=Float
        RH, time indexed [%]
    temp : pd.Series type=Float
        temperature, time indexed [°C]
    wavelengths : int-array
        integer array (or list) of wavelengths tested w/ uniform delta
        in nanometers [nm]
    time : time indicator in [h]
        if not included it will assume 1 h for each dataframe entry.
    Ea : float [kJ/mol]
        Arrhenius activation energy. The default is 0 ofr no dependence 
    n : float
        Power law fit paramter for RH sensitivity. The default is 0 for no dependence.
    p : float
        Power law fit parameter for irradiance sensitivity. Typically
        0.6 +- 0.22. Here it is applied separately for each wavelength bin.
    C2 : float
        Exponential fit parameter for sensitivity to wavelength.
        Typically 0.07 [1/nm]
    R_0 : float
        Prefactor for degradation. Units can vary, but would be something like [%/h]
        It will default to 1.0

    Returns
    -------
    degradation : float
        Total degredation over time and wavelength. Units are determined from R_0 and time.

    """
    # --- TO DO ---
    # unpack input-dataframe
    # spectra = df['spectra']
    # temp_module = df['temp_module']
    # rh_module = df['rh_module']

    # Constants
    R = 0.008314459848  # Gas Constant in [kJ/mol*K]

    wav_bin = list(np.diff(wavelengths))
    wav_bin.append(wav_bin[-1])  # Adding a bin for the last wavelength

    # Integral over Wavelength
    try:
        irr = pd.DataFrame(spectra.tolist(), index=spectra.index)
        irr.columns = wavelengths
    except:
        # TODO: Fix this except it works on some cases, veto it by cases
        print("Removing brackets from spectral irradiance data")
        # irr =
        # data['spectra'].str.strip('[]').str.split(',', expand=True).astype(float)
        irr = spectra.str.strip("[]").str.split(",", expand=True).astype(float)
        irr.columns = wavelengths

    sensitivitywavelengths = np.exp(-C2 * wavelengths)
    irr = irr * sensitivitywavelengths
    irr *= np.array(wav_bin)
    irr = irr**p
    data = pd.DataFrame(index=spectra.index)
    data["G_integral"] = irr.sum(axis=1)

    EApR = -Ea / R
    C4 = np.exp(EApR / temp)

    RHn = rh**n
    data["Arr_integrand"] = C4 * RHn

    data["dD"] = data["G_integral"] * data["Arr_integrand"]

    degradation = R_0 * data["dD"].sum(axis=0)

    return degradation


# change it to take pd.DataFrame? instead of np.ndarray
@njit
def vecArrhenius(
    poa_global: np.ndarray, module_temp: np.ndarray, ea: float, x: float, lnr0: float
) -> float:
    """Calculate degradation using :math:`R_D = R_0 * I^X * e^{\frac{-Ea}{kT}}`.

    Parameters
    ----------
    poa_global : numpy.ndarray
        Plane of array irradiance [W/m^2]

    module_temp : numpy.ndarray
        Cell temperature [C].

    ea : float
        Activation energy [kJ/mol]

    x : float
        Irradiance relation [unitless]

    lnR0 : float
        prefactor [ln(%/h)]

    Returns
    -------
    degredation : float
        Degradation Rate [%/h]
    """
    mask = poa_global >= 25
    poa_global = poa_global[mask]
    module_temp = module_temp[mask]

    ea_scaled = ea / 8.31446261815324e-03
    R0 = np.exp(lnr0)
    poa_global_scaled = poa_global / 1000

    degredation = 0
    for entry in range(
        len(poa_global_scaled)
    ):  # list comprehension not supported by numba
        degredation += (
            R0
            * np.exp(-ea_scaled / (273.15 + module_temp[entry]))
            * np.power(poa_global_scaled[entry], x)
        )

    return degredation / len(poa_global)
