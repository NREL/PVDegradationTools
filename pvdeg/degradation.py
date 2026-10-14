"""Collection of functions for degradation calculations."""

import warnings
import copy
import numpy as np
import pandas as pd
from typing import Union, Optional
from pvdeg import humidity
import pvlib

from . import temperature, spectral, decorators, utilities

R_GAS = 0.00831446261815324  # Gas Constant in [kJ/mol*K]
kB_eV = 8.617333e-5  # eV/K


def _extract_param(parameters, key, default=None):
    """Helper to extract parameter value from nested dict."""
    if parameters is not None and key in parameters:
        value = parameters[key].get("value", None)
        if value is not None:
            return value
    return default


def arrhenius(
    weather_df=None,
    temperature=None,
    RH=None,
    irradiance=None,
    elapsed_time=None,
    Ro=1,
    Ea=0,
    p=0,
    n=0,
    C2=0,
    parameters=None,
):
    """
    Calculate the degradation rate using an Arrhenius function with power law
    functions for humidity and irradiance dependence.

    D = R_0 ∫[RH(t)]^n·e^[-E_a/RT(t)] {∫[e^(-C_2∙λ)∙G(λ,t)]^p dλ}dt

    Parameters
    ----------
    weather_df : pd.DataFrame
        Dataframe containing temperature, humidity, and irradiance data.
        Defaults to module surface temperature, surface humidity, and POA global
        irradiance.
    temperature : pd.DataFrame
        Temperature data for Arrhenius degradation calculation. If not specified,
        uses module surface temperature from weather_df. If Ea=0, temperature is
        not needed.
    RH : pd.DataFrame
        Relative humidity data for Arrhenius degradation calculation. If not
        specified, uses module surface relative humidity from weather_df. If n=0,
        humidity is not needed.
    irradiance : pd.DataFrame
        Irradiance data for Arrhenius degradation calculation. If not specified,
        uses module POA irradiance from weather_df. If p=0, irradiance is not
        needed.
        If C2 is provided, wavelength spectral intensity data must be provided.
        The header should start with "spectra", followed by wavelength points.
        Each element is a list of intensity values at each wavelength [W/m²/nm].
    elapsed_time : pd.DataFrame
        If the time step for each interval is not constant, this can be used to
        provide a different elapsed time value for each element. If it is included
        in the weather_df, it must be under a column named "elapsed_time".
    Ro : float
        Degradation rate prefactor [e.g. %/h/%RH/(1000 W/m²)]. Defaults to 1 if
        not provided.
    Ea : float
        Degradation Activation Energy [kJ/mol]. If Ea=0, no temperature dependence and
        degradation will proceed according to the amount of light an humidity.
    p : float
        Power law coefficient for irradiance dependence. If p=0, ignores light.
        Small p (e.g. 0.0001) means little dependence of degradation on irradiance,
        but only daylight is considered.
    n : float
        Power law coefficient for humidity dependence. If n=0, ignores humidity.
    C2 : float
        Coefficient for spectral response dependence on wavelength.
    parameters : json
        Database containing parameters for Arrhenius calculation. If Ea, n, or p
        are not provided, values are taken from this json database.

    Returns
    -------
    degradation : float
        Total degradation with units as determined by Ro.
    """

    # override defaults with parameters if provided
    if parameters is not None:
        Ro = _extract_param(parameters, "R_0", Ro)
        Ea = _extract_param(parameters, "E_a", Ea)
        n = _extract_param(parameters, "n", n)
        p = _extract_param(parameters, "p", p)
        C2 = _extract_param(parameters, "C_2", C2)

    if temperature is None and Ea != 0:
        if weather_df is not None:
            if "temperature" in weather_df:
                temperature = weather_df["temperature"]
            elif "temp_module" in weather_df:
                temperature = weather_df["temp_module"]
                print("Using temp_module from weather_df for temperature.")
            else:
                raise ValueError("Temperature data must be provided if Ea is provided.")
    if n != 0 and RH is None:
        print(n)
        if "RH_surface_outside" in weather_df:
            RH = weather_df["RH_surface_outside"]
        elif (
            "relative_humidity" in weather_df
            and "temp_air" in weather_df
            and "temp_module" in weather_df
        ):

            RH = humidity.surface_relative(
                weather_df["relative_humidity"],
                weather_df["temp_air"],
                weather_df["temp_module"],
            )
        else:
            raise ValueError(
                "Relative Humidity data must be provided if n is provided."
            )
        if RH is not None:
            print("Using RH_surface_outside from weather_df for humidity.")

    if irradiance is None:
        if C2 != 0 or p != 0:
            if weather_df is not None:
                for col in weather_df.columns:
                    if "SPECTRA" in (col[:7]).upper():
                        irradiance = weather_df[col].copy()
                        irradiance = pd.DataFrame(irradiance)
                        break
                if irradiance is None:
                    if "poa_global" in weather_df:
                        irradiance = weather_df["poa_global"]
                        print("Using poa_global from weather_df for irradiance.")
                        if C2 != 0:
                            raise ValueError(
                                "Irradiance data not provided. Please provide "
                                "irradiance data in weather_df."
                            )
                            # In the future the spectra will be created using AM1.5.
                    else:
                        raise ValueError(
                            "Irradiance data not provided. Please provide it in "
                            "irradiance or weather_df."
                        )
            else:
                raise ValueError(
                    "Irradiance data must be provided when C2 or p are used."
                )
    if elapsed_time is None:
        if weather_df is not None:
            if "elapsed_time" in weather_df:
                elapsed_time = weather_df["elapsed_time"]
    if C2 != 0:
        wavelengths = [
            float(i)
            for i in irradiance.columns[0].split("[")[1].split("]")[0].split(",")
        ]
        wavelengths = np.array(wavelengths)
        bin_widths = (
            np.append(wavelengths, [0, 0]) - np.append([0, 0], wavelengths)
        ) / 2
        bin_widths = bin_widths[1:]
        bin_widths = bin_widths[:-1]
        # assumes the first and last bin widths are the width of that between the next
        # or previous bin, respectively.
        bin_widths[0] = bin_widths[1]
        bin_widths[-1] = bin_widths[-2]
        bin_widths = pd.Series(bin_widths)
        wavelengths = pd.Series(wavelengths)
        if isinstance(irradiance, pd.DataFrame):
            irradiance = irradiance.T.to_numpy().reshape(
                -1,
            )
            irradiance = pd.Series(irradiance)

        if p == 0:
            if Ea != 0:
                if n == 0:
                    degradation = Ro * np.exp(-(Ea / (R_GAS * (temperature + 273.15))))
                else:
                    degradation = (
                        Ro * np.exp(-(Ea / (R_GAS * (temperature + 273.15)))) * (RH**n)
                    )
            else:
                if n == 0:
                    degradation = (
                        Ro * weather_df.iloc[:, 0] / weather_df.iloc[:, 0]
                    )  # This makes sure it sums over the corect number of time
                    # intervals.
                else:
                    degradation = (
                        Ro * (RH**n) * weather_df.iloc[:, 0] / weather_df.iloc[:, 0]
                    )
        else:
            degradation = bin_widths * ((np.exp(-C2 * wavelengths) * irradiance) ** p)
            if Ea != 0:
                if n == 0:
                    degradation = (
                        degradation
                        * Ro
                        * np.exp(-(Ea / (R_GAS * (temperature + 273.15))))
                    )
                else:
                    degradation = (
                        degradation
                        * Ro
                        * np.exp(-(Ea / (R_GAS * (temperature + 273.15))))
                        * (RH**n)
                    )
            else:
                if n == 0:
                    degradation = degradation * Ro
                else:
                    degradation = degradation * Ro * (RH**n)
    elif Ea != 0:
        if n == 0 and p == 0:
            degradation = Ro * np.exp(-(Ea / (R_GAS * (temperature + 273.15))))
        elif n == 0 and p != 0:
            degradation = (
                Ro * np.exp(-(Ea / (R_GAS * (temperature + 273.15)))) * (irradiance**p)
            )
        elif n != 0 and p == 0:
            degradation = (
                Ro * np.exp(-(Ea / (R_GAS * (temperature + 273.15)))) * (RH**n)
            )
        else:
            degradation = (
                Ro
                * np.exp(-(Ea / (R_GAS * (temperature + 273.15))))
                * (RH**n)
                * (irradiance**p)
            )
    else:
        if n == 0 and p == 0:
            degradation = Ro * weather_df.iloc[:, 0] / weather_df.iloc[:, 0]
        elif n == 0 and p != 0:
            degradation = Ro * (irradiance**p)
        elif n != 0 and p == 0:
            degradation = Ro * (RH**n)
        else:
            degradation = Ro * (RH**n) * (irradiance**p)

    if elapsed_time is not None:
        if isinstance(elapsed_time, pd.DataFrame):
            elapsed_time = elapsed_time.T.to_numpy().reshape(
                -1,
            )
            elapsed_time = pd.Series(elapsed_time)
        degradation = degradation * elapsed_time

    return degradation.sum(axis=0, skipna=True)


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
    """
    Calculate Van't Hoff Irradiance Degradation acceleration factor.

    In this calculation, the rate of degradation kinetics is calculated using
    the Van't Hoff model.

    Parameters
    ----------
    weather_df : pd.DataFrame
        DataFrame containing at least dni, dhi, ghi, temperature, wind_speed
    meta : dict
        Location meta-data containing at least latitude, longitude, altitude
    I_chamber : float
        Irradiance of Controlled Condition [W/m²]
    temp_chamber : float
        Reference temperature [°C] ("Chamber Temperature")
    poa : pd.Series or pd.DataFrame, optional
        Series or DataFrame containing 'poa_global', Global Plane of Array Irradiance
        [W/m²]
    temp : pd.Series, optional
        Solar module temperature or Cell temperature [°C]. If not provided, it will
        be generated using the default parameters of pvdeg.temperature.cell
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
        heights between weather database (e.g. NSRDB) and the temperature model
        (e.g. SAPM)
        The NSRDB provides calculations at 2 m (i.e module height) but SAPM uses a 10 m
        height. It is recommended that a power-law relationship between height and wind
        speed of 0.33 be used*. This results in a wind speed that is 1.7 times higher.
        It is acknowledged that this can vary significantly.
    irradiance_kwarg : (dict, optional)
        keyword argument dictionary used for the poa irradiance calculation.
        options: ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``. See
        ``pvdeg.spectral.poa_irradiance``.
    model_kwarg : (dict, optional)
        keyword argument dictionary used for the pvlib temperature model calculation.
        See https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html  # noqa
        for more.


    Returns
    -------
    accelerationFactor : float or pd.Series
        Degradation acceleration factor
    """

    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta, **irradiance_kwarg)

    if isinstance(poa, pd.DataFrame):
        poa_global = poa["poa_global"]

    if temp is None:
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

    rateOfDegEnv = (poa_global**p) * (Tf ** ((temp - temp_chamber) / 10))
    avgOfDegEnv = rateOfDegEnv.mean()
    rateOfDegChamber = I_chamber**p
    accelerationFactor = rateOfDegChamber / avgOfDegEnv
    return accelerationFactor


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
    Calculate IWa: Environment Characterization [W/m²].
    For one year of degradation, the controlled environment lamp settings will need to
    be set to IWa.

    Parameters
    ----------
    weather_df : pd.DataFrame
        DataFrame containing at least dni, dhi, ghi, temperature, wind_speed
    meta : dict
        Location meta-data containing at least latitude, longitude, altitude
    poa : pd.Series or pd.DataFrame, optional
        Series or DataFrame containing 'poa_global', Global Plane of Array Irradiance
        [W/m²]
    temp : pd.Series, optional
        Solar module temperature or Cell temperature [°C]
    Teq : pd.Series, optional
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
        heights between weather database (e.g. NSRDB) and the temperature model
        (e.g. SAPM)
        The NSRDB provides calculations at 2 m (i.e module height) but SAPM uses a 10 m
        height. It is recommended that a power-law relationship between height and wind
        speed of 0.33 be used*. This results in a wind speed that is 1.7 times higher.
        It is acknowledged that this can vary significantly.
    irradiance_kwarg : (dict, optional)
        keyword argument dictionary used for the poa irradiance calculation.
        options: ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``. See
        ``pvdeg.spectral.poa_irradiance``.
    model_kwarg : (dict, optional)
        keyword argument dictionary used for the pvlib temperature model calculation.
        See https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html  # noqa
        for more.


    Returns
    -------
    Iwa : float
        Environment Characterization [W/m²]
    """
    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta, **irradiance_kwarg)

    if temp is None:
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
        toSum = Tf ** (temp / 10)
        summation = toSum.sum(axis=0, skipna=True)
        Teq = (10 / np.log(Tf)) * np.log(summation / len(temp))

    if isinstance(poa, pd.DataFrame):
        poa_global = poa["poa_global"]
    else:
        poa_global = poa

    toSum = (poa_global**p) * (Tf ** ((temp - Teq) / 10))

    summation = toSum.sum(axis=0, skipna=True)

    Iwa = (summation / len(poa_global)) ** (1 / p)

    return Iwa


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
    """
    Calculate the Acceleration Factor between the rate of degradation of a
    modeled environment versus a modeled controlled environment.
    Example: If AF=25, then 1 year of Controlled Environment exposure is equal to
    25 years in the field.

    Parameters
    ----------
    weather_df : pd.DataFrame
        DataFrame containing at least dni, dhi, ghi, temperature, wind_speed
    meta : dict
        Location meta-data containing at least latitude, longitude, altitude
    rh_outdoor : pd.Series
        Relative Humidity of material of interest.
        Acceptable relative humiditys can be calculated from these functions:
        - pvdeg.humidity.backsheet()
        - pvdeg.humidity.back_encapsulant()
        - pvdeg.humidity.front_encapsulant()
        - pvdeg.humidity.surface_relative()
    I_chamber : float
        Irradiance of Controlled Condition [W/m²]
    rh_chamber : float
        Relative Humidity of Controlled Condition [%].
        EXAMPLE: "50 = 50% NOT .5 = 50%"
    temp_chamber : float
        Reference temperature [°C] ("Chamber Temperature")
    Ea : float
        Degradation Activation Energy [kJ/mol]
        if Ea=0 is used there will be not dependence on temperature and degradation will
        proceed according to the amount of light and humidity.
    poa : pd.DataFrame, optional
        Global Plane of Array Irradiance [W/m²]
    temp : pd.Series, optional
        Solar module temperature or Cell temperature [°C]. If no cell temperature is
        given, it will be generated using the default parameters from
        pvdeg.temperature.cell
    p : float
        Fit parameter
        When p=0 the dependence on light will be ignored and degradation will happen
        both day and night. As a caution or a feature, a very small value of p
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
        heights between weather database (e.g. NSRDB) and the temperature model
        (e.g. SAPM)
        The NSRDB provides calculations at 2 m (i.e module height) but SAPM uses a 10 m
        height. It is recommended that a power-law relationship between height and wind
        speed of 0.33 be used*. This results in a wind speed that is 1.7 times higher.
        It is acknowledged that this can vary significantly.
    irradiance_kwarg : (dict, optional)
        keyword argument dictionary used for the poa irradiance calculation.
        options: ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``. See
        ``pvdeg.spectral.poa_irradiance``.
    model_kwarg : (dict, optional)
        keyword argument dictionary used for the pvlib temperature model calculation.
        See https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html  # noqa
        for more.

    Returns
    -------
    accelerationFactor : float or pd.Series
        Degradation acceleration factor
    """

    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta, **irradiance_kwarg)

    if temp is None:
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

    # rate of degradation of the environment
    arrheniusDenominator = (
        (poa_global**p) * (rh_outdoor**n) * np.exp(-Ea / (R_GAS * (temp + 273.15)))
    )

    AvgOfDenominator = arrheniusDenominator.mean()

    # rate of degradation of the simulated chamber
    arrheniusNumerator = (
        (I_chamber**p)
        * (rh_chamber**n)
        * np.exp(-Ea / (R_GAS * (temp_chamber + 273.15)))
    )

    accelerationFactor = arrheniusNumerator / AvgOfDenominator

    return accelerationFactor


def _T_eq_arrhenius(temp, Ea):
    """
    Get Temperature equivalent required for the settings of the controlled environment.
    Calculation is used in determining Arrhenius Environmental Characterization

    Parameters
    -----------
    temp : pandas series
        Solar module temperature or Cell temperature [°C]
    Ea : float
        Degradation Activation Energy [kJ/mol]

    Returns
    -------
    Teq : float
        Temperature equivalent (Celsius) required
        for the settings of the controlled environment

    """

    summationFrame = np.exp(-(Ea / (R_GAS * (temp + 273.15))))
    sumForTeq = summationFrame.sum(axis=0, skipna=True)
    Teq = -((Ea) / (R_GAS * np.log(sumForTeq / len(temp))))
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
        Relative Humidity of material of interest.
        Acceptable relative humiditys can be calculated from these functions:
        - pvdeg.humidity.backsheet()
        - pvdeg.humidity.back_encapsulant()
        - pvdeg.humidity.front_encapsulant()
        - pvdeg.humidity.surface_relative()
    temp : pandas series
        solar module temperature or Cell temperature [°C]
    Ea : float
        Degradation Activation Energy [kJ/mol]
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

    summationFrame = (rh_outdoor**n) * np.exp(-(Ea / (R_GAS * (temp + 273.15))))
    sumForRHwa = summationFrame.sum(axis=0, skipna=True)
    RHwa = (
        sumForRHwa / (len(summationFrame) * np.exp(-(Ea / (R_GAS * (Teq + 273.15)))))
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
    """
    Function to calculate IWa, the Environment Characterization [W/m²].
    For one year of degradation the controlled environment lamp settings will
    need to be set at IWa.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Dataframe containing at least dni, dhi, ghi, temperature, wind_speed
    meta : dict
        Location meta-data containing at least latitude, longitude, altitude
    rh_outdoor : pd.Series
        Relative Humidity of material of interest
        Acceptable relative humiditys can be calculated from these functions:
        - pvdeg.humidity.backsheet()
        - pvdeg.humidity.back_encapsulant()
        - pvdeg.humidity.front_encapsulant()
        - pvdeg.humidity.surface_relative()
    Ea : float
        Degradation Activation Energy [kJ/mol]
    poa : pd.DataFrame, optional
        must contain 'poa_global', Global Plane of Array irradiance [W/m²]
    temp : pd.Series, optional
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
        heights between weather database (e.g. NSRDB) and the temperature model
        (e.g. SAPM)
        The NSRDB provides calculations at 2 m (i.e module height) but SAPM uses a 10 m
        height. It is recommended that a power-law relationship between height and wind
        speed of 0.33 be used*. This results in a wind speed that is 1.7 times higher.
        It is acknowledged that this can vary significantly.
    irradiance_kwarg : (dict, optional)
        keyword argument dictionary used for the poa irradiance calculation.
        options: ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``. See
        ``pvdeg.spectral.poa_irradiance``.
    model_kwarg : (dict, optional)
        keyword argument dictionary used for the pvlib temperature model calculation.
        See https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html  # noqa
        for more.



    Returns
    --------
    Iwa : float
        Environment Characterization [W/m²]
    """
    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta, **irradiance_kwarg)

    if temp is None:
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
        * np.exp(-(Ea / (R_GAS * (temp + 273.15))))
    )
    sumOfNumerator = numerator.sum(axis=0, skipna=True)

    denominator = (
        (len(numerator)) * ((RHwa) ** n) * (np.exp(-(Ea / (R_GAS * (Teq + 273.15)))))
    )

    IWa = (sumOfNumerator / denominator) ** (1 / p)

    return IWa


def degradation_spectral(
    spectra: pd.Series,
    rh: pd.Series,
    temp: pd.Series,
    wavelengths: Union[int, np.ndarray],
    time: pd.Series,
    Ea: float = 0.0,
    n: float = 0.0,
    p: float = 0.6,
    C2: float = 0.07,
    R_0: float = 1.0,
) -> float:
    """
    Compute degradation as double integral of Arrhenius (Activation
    Energy, RH, Temperature) and spectral (wavelength, irradiance)
    functions over wavelength and time.

    Parameters
    ----------
    spectra : pd.Series type=Float
        front or rear irradiance at each wavelength in "wavelengths" [W/m² nm]
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
        Default 1.0

    Returns
    -------
    degradation : float
        Total degradation over time and wavelength. Units are determined from R_0 and
        time.


    """
    # --- TO DO ---
    # unpack input-dataframe
    # spectra = df['spectra']
    # temp_module = df['temp_module']
    # rh_module = df['rh_module']

    wav_bin = list(np.diff(wavelengths))
    wav_bin.append(wav_bin[-1])  # Adding a bin for the last wavelength

    try:
        irr = pd.DataFrame(spectra.tolist(), index=spectra.index)
        irr.columns = wavelengths
    except Exception:
        print("Removing brackets from spectral irradiance data")
        irr = spectra.str.strip("[]").str.split(",", expand=True).astype(float)
        irr.columns = wavelengths

    sensitivitywavelengths = np.exp(-C2 * np.array(wavelengths))
    irr = irr * sensitivitywavelengths
    irr *= np.array(wav_bin)
    irr = irr**p
    data = pd.DataFrame(index=spectra.index)
    data["G_integral"] = irr.sum(axis=1)

    EApR = -Ea / R_GAS
    C4 = np.exp(EApR / temp)

    RHn = rh**n
    data["Arr_integrand"] = C4 * RHn

    data["dD"] = data["G_integral"] * data["Arr_integrand"]

    degradation = R_0 * data["dD"].sum(axis=0)

    return degradation


def vecArrhenius(
    poa_global: np.ndarray, module_temp: np.ndarray, ea: float, x: float, lnr0: float
) -> float:
    """
    Calculates degradation using :math:`R_D = R_0 * I^X * e^{-Ea/(kT)}`
    Parameters
    ----------
    poa_global : numpy.ndarray
        Plane of array irradiance [W/m²]

    module_temp : numpy.ndarray
        Cell temperature [°C].

    ea : float
        Activation energy [kJ/mol]
    x : float
        Irradiance relation [unitless]
    lnr0 : float
        prefactor [ln(%/h)]
    Returns
    ----------
    degradation : float
        Degradation Rate [%/h]
    """
    mask = poa_global >= 25
    poa_global = poa_global[mask]
    module_temp = module_temp[mask]

    ea_scaled = ea / R_GAS
    R0 = np.exp(lnr0)
    poa_global_scaled = poa_global / 1000

    degradation = 0
    for entry in range(len(poa_global_scaled)):
        degradation += (
            R0
            * np.exp(-ea_scaled / (273.15 + module_temp[entry]))
            * np.power(poa_global_scaled[entry], x)
        )

    return degradation / len(poa_global)


def perovskite_degradation(
    weather_df: pd.DataFrame = None,
    meta: dict = None,
    component: str = "total",
    I_in: float = 1.59e21,
    P_O2: float = 21.2,
    P_H2O: pd.Series = None,
    parameters: dict = None,
) -> pd.Series:
    """Compute MAPbI3 perovskite degradation rate using the full kinetic model in [1].

    The net degradation rate is the sum of four terms:

    .. math::

        r = r_{WPO} + r_{DPO} + r_{hum} + r_{therm}

    where:

    .. math::

        r_{WPO} = k_{0,WPO} \\exp\\!\\left(\\frac{-E_{A,WPO}}{R T_K}\\right)
                  \\frac{P_{O_2} P_{H_2O} I_{in}^{0.7}}
                       {\\left[1 + K_{2W} P_{O_2}(1 + K_{3W} I_{in}^{0.7})\\right]^2}

        r_{DPO} = k_{0,DPO} \\exp\\!\\left(\\frac{-E_{A,DPO}}{R T_K}\\right)
                  \\frac{P_{O_2} I_{in}^{0.7}}
                       {1 + K_{2D} P_{O_2}(1 + K_{3D} I_{in}^{0.7})}

        r_{hum} = k_{0,hum} \\exp\\!\\left(\\frac{-E_A^{hum}}{R T_K}\\right)
                  P_{H_2O}\\, I_{in}^{0.7}

        r_{therm} = k_{0,therm} \\exp\\!\\left(\\frac{-E_A^{therm}}{R T_K}\\right)

    All rate constants and activation energies are taken directly from Table 3
    and SI §14 of [1], and are stored in ``DegradationDatabase.json`` entry D015.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data with a time index.  Required column: ``'temp_air'`` [°C].
        Column ``'relative_humidity'`` [%] is required for the WPO and r_hum
        terms; if absent a ValueError is raised.
    meta : dict
        Location metadata.  Not used directly; kept for pipeline compatibility.
    component : str, default ``"total"``
        Which component of the rate to return:

        - ``"total"``   — full model: :math:`r_{WPO} + r_{DPO} + r_{hum} + r_{therm}`
        - ``"WPO"``     — water-accelerated photooxidation term
        - ``"DPO"``     — dry photooxidation term
        - ``"r_hum"``   — humidity-induced minority pathway
        - ``"r_therm"`` — thermal minority pathway

    I_in : float, default ``1.59e21``
        Incident above-bandgap photon flux :math:`I_{in}` [photons m⁻² s⁻¹].
        The model uses :math:`n \\propto I_{in}^{0.7}` as the
        electron-activity proxy. Default value corresponds to 1 sun of AM1.5G
        above-bandgap photon flux.
    P_O2 : float, default ``21.2``
        Oxygen partial pressure :math:`P_{O_2}` [kPa].
        Default value corresponds to ambient air at sea level.
    P_H2O : pd.Series, optional
        Water vapour partial pressure [kPa], time-indexed.  If provided it is
        used directly and ``'relative_humidity'`` is not required in
        ``weather_df``.  Compute with ``pvdeg.humidity.water_vapor_pressure``
        for use as an upstream pipeline job.
    parameters : dict, optional
        Parameter overrides. Any key present here takes precedence over the hardcoded
        default. Entry matches D015 keys; retrieve the full D015 parameter set with
        ``pvdeg.utilities.get_kinetics("D015")``.

    Returns
    -------
    pd.Series
        Time-indexed degradation rate [mol m⁻² s⁻¹].

    Notes
    -----
    The minority pathways (:math:`r_{hum}`, :math:`r_{therm}`) are explicitly
    parameterised in SI §14 of [1].  The paper notes that these are rough
    approximations (they contribute ≪ 5 % of total degradation under typical
    outdoor conditions) and that further work is required for more complete modelling of
    those pathways.

    References
    ----------
    [1] Siegler et al. (2022) *J. Am. Chem. Soc.* 144 (12), 5552–5561.
        doi: 10.1021/jacs.2c00391
    """

    VALID_COMPONENTS = ("total", "WPO", "DPO", "r_hum", "r_therm")
    if component not in VALID_COMPONENTS:
        raise ValueError(
            f"component must be one of {VALID_COMPONENTS}, got '{component}'"
        )
    if weather_df is None:
        raise ValueError("weather_df is required")

    T_K = weather_df["temp_air"] + 273.15  # °C → K

    p = parameters or {}

    # WPO (water-accelerated photooxidation, humid-air column)
    k0_WPO = p.get("k_0,WPO", 3.16e-25)
    E_A_WPO = p.get("E_A,WPO", -8.6827)
    K_2W = p.get("K_2W", 4.40e-3)
    K_3W = p.get("K_3W", 4.32e-15)

    # DPO (dry photooxidation, dry-air column)
    k0_DPO = p.get("k_0,DPO", 5.45e-15)
    E_A_DPO = p.get("E_A,DPO", 59.82)
    K_2D = p.get("K_2D", 3.28e-3)
    K_3D = p.get("K_3D", 6.97e-15)

    # Humidity-induced degradation
    k0_hum = p.get("k_0,hum", 9.2e-22)
    E_A_hum = p.get("E_A^hum", 19.3)

    # Thermal decomposition
    k0_therm = p.get("k_0,therm", 4.1e-4)
    E_A_therm = p.get("E_A^therm", 43.42)

    # Electron-activity proxy: n ∝ I_in^0.7
    n = I_in**0.7

    # Water-vapour partial pressure P_H2O [kPa] = (RH/100) × P_sat
    if P_H2O is None:
        if "relative_humidity" in weather_df.columns:
            rh = weather_df["relative_humidity"]
            P_sat, _ = humidity.water_saturation_pressure(weather_df["temp_air"])
            P_H2O = (rh / 100.0) * P_sat
        else:
            raise ValueError(
                "'relative_humidity' missing in weather_df and P_H2O not given"
            )

    r_WPO = (
        k0_WPO
        * np.exp(-E_A_WPO / (R_GAS * T_K))
        * (P_O2 * P_H2O * n)
        / (1.0 + K_2W * P_O2 * (1.0 + K_3W * n)) ** 2
    )

    r_DPO = (
        k0_DPO
        * np.exp(-E_A_DPO / (R_GAS * T_K))
        * (P_O2 * n)
        / (1.0 + K_2D * P_O2 * (1.0 + K_3D * n))
    )

    r_hum = k0_hum * np.exp(-E_A_hum / (R_GAS * T_K)) * P_H2O * n

    r_therm = k0_therm * np.exp(-E_A_therm / (R_GAS * T_K))

    components = {
        "WPO": r_WPO,
        "DPO": r_DPO,
        "r_hum": r_hum,
        "r_therm": r_therm,
    }

    if component == "total":
        rate = r_WPO + r_DPO + r_hum + r_therm
    else:
        rate = components[component]

    rate.name = f"perovskite_degradation_{component}"
    return rate


def perovskite_degradation_factor(
    weather_df: pd.DataFrame,
    meta: dict = None,
    Ea_fast: float = 0.248,
    Ea_slow: float = 0.243,
    k0_fast: float = 0.56,
    k0_slow: float = 0.48,
    A1: float = 0.25,
    A2: float = 0.70,
    B: float = 0.05,
    gamma: float = 1.0,
    I_ref: float = 1200.0,
    NOCT: float = 48.0,
    poa=None,
    parameters: dict = None,
) -> pd.Series:
    """Compute the cumulative collection efficiency (CE) degradation factor using the
    Arrhenius + power-law model of Orooji et al. (2026), parameterised from Zhao et al.
    (2022).

    The model tracks how collection efficiency evolves over time due to
    combined temperature- and light-induced degradation [1].  At each hourly
    timestep *i*, a degradation factor is computed:

    .. math::

        k(T_i, I_i) = k_0 \\cdot \\exp\\!\\left(\\frac{-E_a}{k_B T_i}\\right)
                      \\cdot \\left(\\frac{I_i}{I_{ref}}\\right)^{\\gamma}

        DF(i) = A_1 \\exp(-k_{fast}(T_i, I_i) \\cdot 1\\text{h})
              + A_2 \\exp(-k_{slow}(T_i, I_i) \\cdot 1\\text{h}) + B

        DF_{total}(t) = \\prod_{i=1}^{t} DF(i)

    The cell temperature is estimated via the NOCT model:

    .. math::

        T_{cell} = T_{air} + \\frac{NOCT - 20}{800} \\cdot G_{POA}

    Degradation is zero at night (:math:`I_i = 0`, :math:`\\gamma > 0`).

    Default parameters reproduce the uncapped CsPbI3 device from [2] as used in
    [1].  They are stored in ``DegradationDatabase.json`` entry **D046** and can
    be retrieved with ``pvdeg.utilities.get_kinetics("D046")``.

    .. note::

        The default ``k0_fast`` / ``k0_slow`` values are calibrated to reproduce
        ``T90,Agg`` ≈ 1440 h under ISOS-L2 (85 °C, 1200 W m⁻², the Zhao
        reference intensity ``I_ref``) as reported by
        [1].  ``A1``, ``A2``, ``B`` are estimated from the biexponential fit
        shape in Fig. 3A of [2].  Exact fitted coefficients appear in the Zhao
        supplementary; re-fit these parameters to your own device data for
        quantitative predictions.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data with at least ``'temp_air'`` [°C].
    meta : dict, optional
        Location metadata.  Required when ``poa`` is not provided.
    Ea_fast : float, default 0.248
        Activation energy for the fast degradation process [eV].
    Ea_slow : float, default 0.243
        Activation energy for the slow degradation process [eV].
    k0_fast : float, default 0.56
        Arrhenius pre-exponential for the fast process [h⁻¹] evaluated at
        ``I_ref``.
    k0_slow : float, default 0.48
        Arrhenius pre-exponential for the slow process [h⁻¹] evaluated at
        ``I_ref``.
    A1 : float, default 0.25
        Amplitude of the fast exponential in the biexponential decay.
        Must satisfy ``A1 + A2 + B = 1``.
    A2 : float, default 0.70
        Amplitude of the slow exponential.
    B : float, default 0.05
        Residual stable fraction (long-term plateau).
    gamma : float, default 1.0
        Light-intensity exponent.  ``gamma=1`` → linear scaling with irradiance;
        ``gamma=0`` → temperature-only (no light dependence).
    I_ref : float, default 1200.0
        Reference irradiance [W m⁻²] at which ``k0`` values were measured.
    NOCT : float, default 48.0
        Nominal Operating Cell Temperature [°C].
    poa : pd.Series or pd.DataFrame, optional
        Plane-of-array global irradiance [W m⁻²].  If a DataFrame is passed,
        ``'poa_global'`` is used.  Computed from ``weather_df`` and ``meta``
        via :func:`pvdeg.spectral.poa_irradiance` when not provided.
    parameters : dict, optional
        Flat parameter dict (e.g. from ``pvdeg.utilities.get_kinetics("D046")``).
        Keys present here override the corresponding keyword arguments.

    Returns
    -------
    pd.Series
        Time-indexed cumulative degradation factor ``DF_total`` (dimensionless).
        Starts near 1.0 and decreases monotonically toward ``B`` as degradation
        accumulates.  Multiply by ``CE_0`` to obtain absolute CE at each hour.

    References
    ----------
    [1] Orooji et al. (2026) *EES Solar*. doi: 10.1039/d6el00021e
    [2] Zhao et al. (2022) *Science* 377, 307–310. doi: 10.1126/science.abn5679
    """
    if weather_df is None:
        raise ValueError("weather_df is required")

    # Override defaults from parameter dict (flat key-value from get_kinetics)
    if parameters is not None:
        Ea_fast = parameters.get("Ea_fast", Ea_fast)
        Ea_slow = parameters.get("Ea_slow", Ea_slow)
        k0_fast = parameters.get("k0_fast", k0_fast)
        k0_slow = parameters.get("k0_slow", k0_slow)
        A1 = parameters.get("A1", A1)
        A2 = parameters.get("A2", A2)
        B = parameters.get("B", B)
        gamma = parameters.get("gamma", gamma)
        I_ref = parameters.get("I_ref", I_ref)

    if abs(A1 + A2 + B - 1.0) > 1e-6:
        raise ValueError(
            f"A1 + A2 + B must equal 1.0 (got {A1} + {A2} + {B} = {A1 + A2 + B:.6f})"  # noqa
        )

    # POA irradiance
    if poa is None:
        poa_df = spectral.poa_irradiance(weather_df, meta)
        G = poa_df["poa_global"]
    elif isinstance(poa, pd.DataFrame):
        G = poa["poa_global"]
    else:
        G = poa

    G = G.clip(lower=0.0)

    # Cell temperature via NOCT model [K]
    T_cell_K = (weather_df["temp_air"] + (NOCT - 20.0) / 800.0 * G) + 273.15

    # Irradiance ratio (zero at night → no degradation when gamma > 0)
    I_ratio_gamma = (G / I_ref).clip(lower=0.0) ** gamma

    # Hourly degradation rates [h⁻¹]
    k_fast_h = k0_fast * np.exp(-Ea_fast / (kB_eV * T_cell_K)) * I_ratio_gamma
    k_slow_h = k0_slow * np.exp(-Ea_slow / (kB_eV * T_cell_K)) * I_ratio_gamma

    # Per-hour degradation factors (biexponential, Eq. 9 of [1])
    DF_hourly = A1 * np.exp(-k_fast_h) + A2 * np.exp(-k_slow_h) + B

    # Cumulative product → DF_total(t)
    DF_total = DF_hourly.cumprod()
    DF_total.name = "perovskite_degradation_factor"
    return DF_total


def degraded_power_ratio(
    weather_df: pd.DataFrame,
    meta: dict,
    ce_factor: pd.Series,
    I_sc_ref: float = 0.022,
    I_0_ref: float = 1e-13,
    R_s: float = 0.5,
    R_sh: float = 5000.0,
    n_diode: float = 1.5,
    alpha_isc: float = 5e-4,
    G_ref: float = 1000.0,
    T_ref: float = 25.0,
    poa=None,
    temp_cell: pd.Series = None,
    temp_model: str = "sapm",
    conf: str = "open_rack_glass_polymer",
    wind_factor: float = 0.33,
) -> dict:
    """Compute the aggregated power ratio PR_Agg and T90,Agg for a degrading
    single-junction perovskite solar cell.

    Degradation is applied to the photocurrent via the CE factor:

    .. math::

        I_L^{deg}(t) = I_{sc,ref} \\cdot \\frac{G(t)}{G_{ref}}
                       \\cdot [1 + \\alpha_{ISC}(T_{cell}(t) - T_{ref})]
                       \\cdot CE_{factor}(t)

    The maximum power point is computed at each hour using the pvlib single-diode
    model [1].  The aggregated power ratio is:

    .. math::

        PR_{Agg}(t) = \\frac{\\sum_{i=1}^{t} P_{deg}(i)}{\\sum_{i=1}^{t} P_{ref}(i)}

    ``T90,Agg`` is the first hour at which ``PR_Agg`` drops below 0.90.

    Default single-diode parameters correspond to a generic single-junction
    perovskite cell with ≈20 % PCE (1 cm² area).  Provide device-specific values
    for quantitative predictions.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data with at least ``'temp_air'`` [°C] and wind columns.
    meta : dict
        Location metadata (latitude, longitude, altitude, timezone).
    ce_factor : pd.Series
        Time-indexed CE degradation factor (0–1) from
        :func:`perovskite_degradation_factor`.
    I_sc_ref : float, default 0.022
        Short-circuit current at STC [A].  Default: 22 mA (1 cm² perovskite cell).
    I_0_ref : float, default 1e-13
        Dark saturation current [A].
    R_s : float, default 0.5
        Series resistance [Ω].
    R_sh : float, default 5000.0
        Shunt resistance [Ω].
    n_diode : float, default 1.5
        Diode ideality factor.
    alpha_isc : float, default 5e-4
        Temperature coefficient of I_sc [K⁻¹].
    G_ref : float, default 1000.0
        Reference irradiance [W m⁻²].
    T_ref : float, default 25.0
        Reference temperature [°C].
    poa : pd.Series or pd.DataFrame, optional
        Plane-of-array global irradiance [W m⁻²].  Computed when not provided.
    temp_cell : pd.Series, optional
        Pre-computed cell/module temperature [°C], time-indexed.  When provided
        the internal ``pvdeg.temperature.module()`` call is skipped.  Useful for
        controlled-environment simulations (e.g. ISOS-L2) or when ``meta`` does
        not contain ``'wind_height'``.
    temp_model : str, default ``"sapm"``
        pvlib temperature model for module temperature (ignored if ``temp_cell``
        is given).
    conf : str, default ``"open_rack_glass_polymer"``
        Module mounting configuration (ignored if ``temp_cell`` is given).
    wind_factor : float, default 0.33
        Wind-speed height correction exponent (ignored if ``temp_cell`` is given).

    Returns
    -------
    dict with keys:

    ``"PR_Agg"`` : pd.Series
        Cumulative aggregated power ratio (time-indexed).
    ``"T90_Agg_hours"`` : float or None
        Hours until PR_Agg first drops below 0.90, or ``None`` if not reached.
    ``"power_degraded"`` : pd.Series
        Hourly maximum power of the degraded cell [W].
    ``"power_reference"`` : pd.Series
        Hourly maximum power of the non-degraded reference cell [W].

    References
    ----------
    [1] pvlib: https://pvlib-python.readthedocs.io
    [2] Orooji et al. (2026) *EES Solar*. doi: 10.1039/d6el00021e
    """
    if weather_df is None:
        raise ValueError("weather_df is required")
    if ce_factor is None:
        raise ValueError("ce_factor is required")

    # Align ce_factor to weather_df.index — raises if any timestamps are missing
    if not ce_factor.index.equals(weather_df.index):
        ce_factor = ce_factor.reindex(weather_df.index)
        if ce_factor.isna().any():
            raise ValueError(
                "ce_factor is missing values for some timestamps in weather_df.index."
                "Ensure ce_factor covers the full time range of weather_df."
            )

    # POA irradiance
    if poa is None:
        poa_df = spectral.poa_irradiance(weather_df, meta)
    elif isinstance(poa, pd.Series):
        poa_df = pd.DataFrame({"poa_global": poa}, index=poa.index)
    else:
        poa_df = poa

    G = poa_df["poa_global"].clip(lower=0.0)

    # Module / cell temperature [°C]
    if temp_cell is not None:
        if not temp_cell.index.equals(weather_df.index):
            temp_cell = temp_cell.reindex(weather_df.index)
            if temp_cell.isna().any():
                raise ValueError(
                    "temp_cell is missing values for some timestamps in"
                    " weather_df.index. Ensure temp_cell covers the full time range of"
                    " weather_df."
                )
        T_cell_C = temp_cell.to_numpy()
    else:
        T_mod = temperature.module(
            weather_df,
            meta,
            poa=poa_df,
            temp_model=temp_model,
            conf=conf,
            wind_factor=wind_factor,
        )
        T_cell_C = T_mod.values
    T_cell_K = T_cell_C + 273.15

    # nNsVth: thermal voltage × ideality × Ns=1 [V]
    nNsVth = n_diode * kB_eV * T_cell_K

    # Photocurrent scaled by irradiance and temperature
    T_delta = T_cell_C - T_ref
    I_L_ref = np.clip(
        I_sc_ref * (G.values / G_ref) * (1.0 + alpha_isc * T_delta), 0.0, None
    )
    I_L_deg = np.clip(I_L_ref * ce_factor.to_numpy(), 0.0, None)

    # Maximum power via pvlib single-diode (vectorised).
    # Suppress the scipy RuntimeWarning that arises when G=0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd_ref = pvlib.pvsystem.singlediode(
            I_L_ref, I_0_ref, R_s, R_sh, nNsVth, method="lambertw"
        )
        sd_deg = pvlib.pvsystem.singlediode(
            I_L_deg, I_0_ref, R_s, R_sh, nNsVth, method="lambertw"
        )

    P_ref = pd.Series(
        np.asarray(sd_ref["p_mp"]), index=weather_df.index, name="power_reference"
    )
    P_deg = pd.Series(
        np.asarray(sd_deg["p_mp"]), index=weather_df.index, name="power_degraded"
    )

    # Cumulative aggregated power ratio
    cum_ref = P_ref.cumsum()
    cum_deg = P_deg.cumsum()
    PR_Agg = cum_deg / cum_ref.where(cum_ref > 0)
    PR_Agg.name = "PR_Agg"

    # T90,Agg: first timestep where PR_Agg < 0.90
    below_90 = PR_Agg.index[PR_Agg < 0.90]
    if len(below_90) > 0:
        try:
            T90_Agg_hours = (below_90[0] - PR_Agg.index[0]).total_seconds() / 3600.0
        except AttributeError:
            # Non-datetime index: fall back to positional count (assumes hourly)
            T90_Agg_hours = float(PR_Agg.index.get_loc(below_90[0]))
    else:
        T90_Agg_hours = None

    return {
        "PR_Agg": PR_Agg,
        "T90_Agg_hours": T90_Agg_hours,
        "power_degraded": P_deg,
        "power_reference": P_ref,
    }


def degraded_power_ratio_tandem(
    weather_df: pd.DataFrame,
    meta: dict,
    ce_factor: pd.Series,
    tandem_template,
    jsc_top_ref: float = 20.76,
    jsc_bot_ref: float = 20.76,
    alpha_isc_top: float = 5e-4,
    alpha_isc_bot: float = 5e-4,
    G_ref: float = 1000.0,
    T_ref: float = 25.0,
    poa=None,
    temp_cell: pd.Series = None,
    temp_model: str = "sapm",
    conf: str = "open_rack_glass_polymer",
    wind_factor: float = 0.33,
    jsc_top_ref_ma: Optional[float] = None,
    jsc_bot_ref_ma: Optional[float] = None,
) -> dict:
    """Compute PR_Agg and T90,Agg for a degrading perovskite/Si tandem device
    using PVCircuit.

    This function is the tandem counterpart to :func:`degraded_power_ratio`.
    The collection-efficiency degradation factor is applied to the top-cell
    photocurrent only, while the bottom-cell photocurrent remains undegraded by
    CE (both still scale with irradiance and temperature):

    .. math::

        J_{top,deg}(t) = J_{top,ref}(t) \\cdot CE_{factor}(t)

    Inputs ``jsc_*_ref`` are in mA/cm² and internally converted to A/cm²
    before assignment to PVCircuit junction ``Jext``.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data with a DateTimeIndex.
    meta : dict
        Location metadata used when ``poa`` or ``temp_cell`` are not provided.
    ce_factor : pd.Series
        Time-indexed CE degradation factor (0-1), typically from
        :func:`perovskite_degradation_factor`.
    tandem_template : pvcircuit.Multi2T
        PVCircuit tandem template device (copied internally for reference and
        degraded simulations).
    jsc_top_ref : float, default 20.76
        Top-cell reference short-circuit current density [mA/cm²] at
        ``G_ref`` and ``T_ref``.
    jsc_bot_ref : float, default 20.76
        Bottom-cell reference short-circuit current density [mA/cm²] at
        ``G_ref`` and ``T_ref``.
    alpha_isc_top : float, default 5e-4
        Top-cell temperature coefficient of Jsc [1/K].
    alpha_isc_bot : float, default 5e-4
        Bottom-cell temperature coefficient of Jsc [1/K].
    G_ref : float, default 1000.0
        Reference irradiance [W/m²].
    T_ref : float, default 25.0
        Reference cell temperature [°C].
    poa : pd.Series or pd.DataFrame, optional
        Plane-of-array irradiance [W/m²]. If omitted, computed with
        :func:`pvdeg.spectral.poa_irradiance`.
    temp_cell : pd.Series, optional
        Pre-computed cell temperature [°C]. If omitted, computed with
        :func:`pvdeg.temperature.module`.
    temp_model : str, default ``"sapm"``
        pvlib temperature model used when ``temp_cell`` is not provided.
    conf : str, default ``"open_rack_glass_polymer"``
        Module mounting configuration used when ``temp_cell`` is not provided.
    wind_factor : float, default 0.33
        Wind-speed correction exponent used when ``temp_cell`` is not provided.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"PR_Agg"``: cumulative aggregated power ratio [pd.Series]
        - ``"T90_Agg_hours"``: first hour where ``PR_Agg < 0.90`` [float|None]
        - ``"power_degraded"``: degraded tandem hourly MPP power [pd.Series]
        - ``"power_reference"``: reference tandem hourly MPP power [pd.Series]
    """
    if weather_df is None:
        raise ValueError("weather_df is required")
    if ce_factor is None:
        raise ValueError("ce_factor is required")
    if tandem_template is None:
        raise ValueError("tandem_template is required")

    # Backward-compatible aliases for older notebooks/scripts.
    if jsc_top_ref_ma is not None:
        warnings.warn(
            "'jsc_top_ref_ma' is deprecated; use 'jsc_top_ref' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        jsc_top_ref = jsc_top_ref_ma
    if jsc_bot_ref_ma is not None:
        warnings.warn(
            "'jsc_bot_ref_ma' is deprecated; use 'jsc_bot_ref' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        jsc_bot_ref = jsc_bot_ref_ma

    # Import here so non-tandem workflows do not require pvcircuit.
    try:
        import pvcircuit as pvc  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "degraded_power_ratio_tandem requires pvcircuit. "
            "Install pvcircuit to use tandem energy-yield modeling."
        ) from exc

    if not ce_factor.index.equals(weather_df.index):
        ce_factor = ce_factor.reindex(weather_df.index)
        if ce_factor.isna().any():
            raise ValueError(
                "ce_factor is missing values for some timestamps in weather_df.index."
                "Ensure ce_factor covers the full time range of weather_df."
            )

    # POA irradiance
    if poa is None:
        poa_df = spectral.poa_irradiance(weather_df, meta)
    elif isinstance(poa, pd.Series):
        poa_df = pd.DataFrame({"poa_global": poa}, index=poa.index)
    else:
        poa_df = poa
    G = poa_df["poa_global"].clip(lower=0.0)

    # Module / cell temperature [°C]
    if temp_cell is not None:
        if not temp_cell.index.equals(weather_df.index):
            temp_cell = temp_cell.reindex(weather_df.index)
            if temp_cell.isna().any():
                raise ValueError(
                    "temp_cell is missing values for some timestamps in"
                    " weather_df.index. Ensure temp_cell covers the full time range of"
                    " weather_df."
                )
        T_cell_C = temp_cell
    else:
        T_mod = temperature.module(
            weather_df,
            meta,
            poa=poa_df,
            temp_model=temp_model,
            conf=conf,
            wind_factor=wind_factor,
        )
        T_cell_C = T_mod

    # Use independent copies so degraded and reference states do not interfere.
    try:
        dev_ref = tandem_template.copy()
        dev_deg = tandem_template.copy()
    except AttributeError:
        dev_ref = copy.deepcopy(tandem_template)
        dev_deg = copy.deepcopy(tandem_template)

    p_ref = np.zeros(len(weather_df), dtype=float)
    p_deg = np.zeros(len(weather_df), dtype=float)

    for i, ts in enumerate(weather_df.index):
        g_scale = max(float(G.iloc[i]) / G_ref, 0.0)
        t_top = 1.0 + alpha_isc_top * (float(T_cell_C.iloc[i]) - T_ref)
        t_bot = 1.0 + alpha_isc_bot * (float(T_cell_C.iloc[i]) - T_ref)

        # mA/cm² -> A/cm² conversion for PVCircuit Junction.Jext
        j_top_ref_acm2 = max(jsc_top_ref * g_scale * t_top, 0.0) / 1000.0
        j_bot_ref_acm2 = max(jsc_bot_ref * g_scale * t_bot, 0.0) / 1000.0
        j_top_deg_acm2 = max(j_top_ref_acm2 * float(ce_factor.loc[ts]), 0.0)

        # Reference tandem (no CE degradation)
        dev_ref.j[0].set(Jext=j_top_ref_acm2, TC=float(T_cell_C.iloc[i]))
        dev_ref.j[1].set(Jext=j_bot_ref_acm2, TC=float(T_cell_C.iloc[i]))
        p_ref[i] = float(dev_ref.MPP()["Pmp"])

        # Degraded tandem (top-cell CE degradation only)
        dev_deg.j[0].set(Jext=j_top_deg_acm2, TC=float(T_cell_C.iloc[i]))
        dev_deg.j[1].set(Jext=j_bot_ref_acm2, TC=float(T_cell_C.iloc[i]))
        p_deg[i] = float(dev_deg.MPP()["Pmp"])

    P_ref = pd.Series(p_ref, index=weather_df.index, name="power_reference")
    P_deg = pd.Series(p_deg, index=weather_df.index, name="power_degraded")

    # Cumulative aggregated power ratio
    cum_ref = P_ref.cumsum()
    cum_deg = P_deg.cumsum()
    PR_Agg = cum_deg / cum_ref.where(cum_ref > 0)
    PR_Agg.name = "PR_Agg"

    # T90,Agg: first timestep where PR_Agg < 0.90
    below_90 = PR_Agg.index[PR_Agg < 0.90]
    if len(below_90) > 0:
        try:
            T90_Agg_hours = (below_90[0] - PR_Agg.index[0]).total_seconds() / 3600.0
        except AttributeError:
            T90_Agg_hours = float(PR_Agg.index.get_loc(below_90[0]))
    else:
        T90_Agg_hours = None

    return {
        "PR_Agg": PR_Agg,
        "T90_Agg_hours": T90_Agg_hours,
        "power_degraded": P_deg,
        "power_reference": P_ref,
    }


def acetic_acid_generation(
    temp_module: pd.Series,
    Ro: float = 0.00331,
    Ea_gen: float = 90.0,
    T_ref: float = 27.0,
    encapsulant: str = "AA002",
) -> pd.Series:
    """Calculate the acetic acid generation rate in EVA encapsulant.

    Uses an Arrhenius model for the hydrolysis source term of ethylene-vinyl
    acetate (EVA). The rate at reference temperature ``T_ref`` is scaled via:

    .. math::

        R(T) = R_0 \\cdot \\exp\\!\\left[
            \\frac{-E_a}{R}\\left(\\frac{1}{T} - \\frac{1}{T_{ref}}\\right)
        \\right]

    The baseline ``Ro`` is calibrated from Kempe et al. (2007) under damp-heat
    conditions (85°C/85% RH), so this implementation assumes that humidity basis
    implicitly and models temperature dependence directly.

    Validation: Gnocchi et al. (2018) [2]_.

    Parameters
    ----------
    temp_module : pd.Series
        Time-indexed module temperature [°C].
    Ro : float, default 0.00331
        Acetic acid source term at ``T_ref`` under the 85% RH calibration
        basis [ng/min/g].
    Ea_gen : float, default 90.0
        Activation energy for HAc generation [kJ/mol].
    T_ref : float, default 27.0
        Reference temperature for ``Ro`` [°C].
    encapsulant : str, default ``"AA002"``
        Key in ``AApermeation.json`` from which to load default parameters.
        Set to ``None`` to use explicitly provided values only.

    Returns
    -------
    generation_rate : pd.Series
        Time-indexed acetic acid generation rate [ng/min/g].

    References
    ----------
    .. [1] Kempe, M. D., et al. (2007). "Acetic acid production and glass
       transition concerns with ethylene-vinyl acetate used in photovoltaic
       devices." *Solar Energy Materials and Solar Cells* 91.4: 315-329.
    .. [2] Gnocchi, L., et al. (2018). "Measuring and modelling the generation
       of acetic acid in aged ethylene-vinyl acetate-based encapsulants used
       in solar modules." EU PVSEC.
    """

    if encapsulant is not None:
        params = utilities.read_material(
            pvdeg_file="AApermeation",
            key=encapsulant,
            parameters=["Ro", "Ea_gen"],
        )
        Ro = params.get("Ro", Ro)
        Ea_gen = params.get("Ea_gen", Ea_gen)

        full_params = utilities.read_material(
            pvdeg_file="AApermeation",
            key=encapsulant,
            parameters=["Ro"],
            values_only=False,
        )
        ro_entry = full_params.get("Ro", {})
        if isinstance(ro_entry, dict) and "ref_temp_C" in ro_entry:
            T_ref = ro_entry["ref_temp_C"]

    T_K = temp_module + 273.15
    T_ref_K = T_ref + 273.15

    generation_rate = Ro * np.exp((-Ea_gen / R_GAS) * (1.0 / T_K - 1.0 / T_ref_K))

    return pd.Series(generation_rate, index=temp_module.index, name="HAc_rate_ng_min_g")


def acetic_acid_cumulative(
    temp_module: pd.Series,
    Ro: float = 0.00331,
    Ea_gen: float = 90.0,
    T_ref: float = 27.0,
    encapsulant: str = "AA002",
) -> pd.Series:
    """Calculate the cumulative acetic acid produced in EVA over time.

    Integrates the hourly Arrhenius generation rate (assuming 1-hour time steps)
    to produce cumulative HAc concentration in [mg/g], matching the units and
    magnitude of experimental measurements from Gnocchi et al. (2018).

    Parameters
    ----------
    temp_module : pd.Series
        Time-indexed module temperature [°C].
    Ro : float, default 0.00331
        Acetic acid source term at ``T_ref`` [ng/min/g], calibrated at 85% RH.
    Ea_gen : float, default 90.0
        Activation energy for HAc generation [kJ/mol].
    T_ref : float, default 27.0
        Reference temperature for ``Ro`` [°C].
    encapsulant : str, default ``"AA002"``
        Key in ``AApermeation.json``.  Set to ``None`` to use explicit values.

    Returns
    -------
    cumulative_HAc : pd.Series
        Time-indexed cumulative acetic acid concentration [mg/g].
    """
    rate = acetic_acid_generation(
        temp_module, Ro=Ro, Ea_gen=Ea_gen, T_ref=T_ref, encapsulant=encapsulant
    )

    # rate is in [ng/min/g]; integrate over 60 min/h, convert ng → mg
    hourly_production_mg = rate * 60.0 / 1e6  # [mg/g per hour]
    cumulative_HAc = hourly_production_mg.cumsum()
    cumulative_HAc.name = "HAc_cumulative_mg_g"
    return cumulative_HAc
