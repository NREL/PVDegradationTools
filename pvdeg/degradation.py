"""Collection of functions for degradation calculations."""

import numpy as np
import pandas as pd
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
    elapsed_time=None,
    Ro=None,
    Ea=None,
    p=None,
    n=None,
    C2=None,
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

    if Ro is None:
        if parameters is not None:
            if "R_0.value" in parameters:
                Ro = parameters["R_0.value"]
            else:
                Ro = 1
        else:
            Ro = 1
    if Ea is None:
        if parameters is not None:
            if "Ea.value" in parameters:
                Ea = parameters["Ea.value"]
            else:
                Ea = 0
        else:
            Ea = 0
    if n is None:
        if parameters is not None:
            if "n.value" in parameters:
                n = parameters["n.value"]
            else:
                n = 0
        else:
            n = 0
    if p is None:
        if parameters is not None:
            if "p.value" in parameters:
                p = parameters["p.value"]
            else:
                p = 0
        else:
            p = 0
    if temperature is None:
        temperature = weather_df["temp"]
    if (
        RH is None
        and "relative_humidity" in weather_df
        and "temp_air" in weather_df
        and "temp_module" in weather_df
    ):
        RH = humidity.surface_relative(
            weather_df["relative_humidity"],
            weather_df["temp_air"],
            weather_df["temp_module"],
        )

    if C2 is None:
        if parameters is not None:
            if "C_2.value" in parameters:
                C2 = parameters["C_2.value"]
            else:
                C2 = 0
        else:
            C2 = 0
    if irradiance is None:
        if C2 != 0 or p != 0:
            if weather_df is not None:
                for col in weather_df.columns:
                    if "SPECTRA" in (col[:7]).upper():
                        irradiance = weather_df[col].copy()
                        irradiance = pd.DataFrame(irradiance)
                        break
                if "poa_global" in weather_df and irradiance is None:
                    if C2 == 0:
                        irradiance = weather_df["poa_global"]
                        print("Using poa_global from weather_df for irradiance.")
                    else:
                        raise ValueError(
                            "Irradiance data not provided. Please provide irradiance data in weather_df."  # noqa
                        )
                else:
                    if irradiance is None:
                        raise ValueError(
                            "POA data not provided. Please provide it in irradiance or weather_df."  # noqa
                        )
            else:
                raise ValueError(
                    "Irradiance data must be provided if C2 or p are provided."  # noqa
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
                    degradation = Ro * np.exp(
                        -(Ea / (0.00831446261815324 * (temperature + 273.15)))
                    )
                else:
                    degradation = (
                        Ro
                        * np.exp(-(Ea / (0.00831446261815324 * (temperature + 273.15))))
                        * (RH**n)
                    )
            else:
                if n == 0:
                    degradation = (
                        Ro * temperature / temperature
                    )  # This makes sure it sums over the corect number of time
                    # intervals.
                else:
                    degradation = Ro * (RH**n) * temperature / temperature
        else:
            degradation = bin_widths * ((np.exp(-C2 * wavelengths) * irradiance) ** p)
            if Ea != 0:
                if n == 0:
                    degradation = (
                        degradation
                        * Ro
                        * np.exp(-(Ea / (0.00831446261815324 * (temperature + 273.15))))
                    )
                else:
                    degradation = (
                        degradation
                        * Ro
                        * np.exp(-(Ea / (0.00831446261815324 * (temperature + 273.15))))
                        * (RH**n)
                    )
            else:
                if n == 0:
                    degradation = degradation * Ro
                else:
                    degradation = degradation * Ro * (RH**n)
    elif Ea != 0:
        if n == 0 and p == 0:
            degradation = Ro * np.exp(
                -(Ea / (0.00831446261815324 * (temperature + 273.15)))
            )
        elif n == 0 and p != 0:
            degradation = (
                Ro
                * np.exp(-(Ea / (0.00831446261815324 * (temperature + 273.15))))
                * (irradiance**p)
            )
        elif n != 0 and p == 0:
            degradation = (
                Ro
                * np.exp(-(Ea / (0.00831446261815324 * (temperature + 273.15))))
                * (RH**n)
            )
        else:
            degradation = (
                Ro
                * np.exp(-(Ea / (0.00831446261815324 * (temperature + 273.15))))
                * (RH**n)
                * (irradiance**p)
            )
    else:
        if n == 0 and p == 0:
            degradation = Ro * temperature / temperature
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
        (poa_global**p)
        * (rh_outdoor**n)
        * np.exp(-Ea / (0.00831446261815324 * (temp + 273.15)))
    )

    AvgOfDenominator = arrheniusDenominator.mean()

    # rate of degradation of the simulated chamber
    arrheniusNumerator = (
        (I_chamber**p)
        * (rh_chamber**n)
        * np.exp(-Ea / (0.00831446261815324 * (temp_chamber + 273.15)))
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
    """
    Compute degradation as double integral of Arrhenius (Activation
    Energy, RH, Temperature) and spectral (wavelength, irradiance)
    functions over wavelength and time.

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

    # Constants
    R = 0.008314459848  # Gas Constant in [kJ/mol*K]

    wav_bin = list(np.diff(wavelengths))
    wav_bin.append(wav_bin[-1])  # Adding a bin for the last wavelength

    # Integral over Wavelength
    try:
        irr = pd.DataFrame(spectra.tolist(), index=spectra.index)
        irr.columns = wavelengths
    except Exception:
        # TODO: Fix this except it works on some cases, veto it by cases
        print("Removing brackets from spectral irradiance data")
        # irr = data['spectra'].str.strip('[
        # ]').str.split(',', expand=True).astype(float)
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
def vecArrhenius(
    poa_global: np.ndarray, module_temp: np.ndarray, ea: float, x: float, lnr0: float
) -> float:
    """
    Calculates degradation using :math:`R_D = R_0 * I^X * e^{\\frac{-Ea}{kT}}`

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
    ----------
    degradation : float
        Degradation Rate [%/h]

    """

    mask = poa_global >= 25
    poa_global = poa_global[mask]
    module_temp = module_temp[mask]

    ea_scaled = ea / 8.31446261815324e-03
    R0 = np.exp(lnr0)
    poa_global_scaled = poa_global / 1000

    degradation = 0
    for entry in range(
        len(poa_global_scaled)
    ):  # list comprehension not supported by numba
        degradation += (
            R0
            * np.exp(-ea_scaled / (273.15 + module_temp[entry]))
            * np.power(poa_global_scaled[entry], x)
        )

    return degradation / len(poa_global)
