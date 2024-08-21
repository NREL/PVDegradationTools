import numpy as np
import pandas as pd
from scipy.constants import convert_temperature
from pvdeg.decorators import geospatial_quick_shape
import dask.dataframe as dd

from . import temperature
from . import spectral
from . import weather


def _avg_daily_temp_change(time_range, temperature):
    """
    Helper function. Get the average of a year for the daily maximum temperature change.

    For every 24hrs this function will find the delta between the maximum
    temperature and minimun temperature.  It will then take the deltas for
    every day of the year and return the average delta.

    Parameters
    ------------
    time_range : timestamp series
        Local time of specific site by the hour
        year-month-day hr:min:sec . (Example) 2002-01-01 01:00:00
    temperature : float series
        Photovoltaic module temperature(Celsius) for every hour of a year

    Returns
    -------
    avg_daily_temp_change : float
        Average Daily Temerature Change for 1-year (Celsius)
    avg_max_temperature : float
        Average of Daily Maximum Temperature for 1-year (Celsius)

    """

    if time_range.dtype == "object":
        time_range = pd.to_datetime(time_range)

    # Setup frame for vector processing
    timeAndTemp_df = pd.DataFrame(columns=["temperature"])
    timeAndTemp_df["temperature"] = temperature
    timeAndTemp_df.index = time_range
    timeAndTemp_df["month"] = timeAndTemp_df.index.month
    timeAndTemp_df["day"] = timeAndTemp_df.index.day

    # Group by month and day to determine the max and min temperature [°C] for each day
    dailyMaxTemp_series = timeAndTemp_df.groupby(["month", "day"])["temperature"].max()
    dailyMinTemp_series = timeAndTemp_df.groupby(["month", "day"])["temperature"].min()
    temperature_change = pd.DataFrame(
        {"Max": dailyMaxTemp_series, "Min": dailyMinTemp_series}
    )
    temperature_change["TempChange"] = (
        temperature_change["Max"] - temperature_change["Min"]
    )

    # Find the average temperature change for every day of one year [°C]
    avg_daily_temp_change = temperature_change["TempChange"].mean()
    # Find daily maximum temperature average
    avg_max_temperature = dailyMaxTemp_series.mean()

    return avg_daily_temp_change, avg_max_temperature


def _times_over_reversal_number(temperature, reversal_temp):
    """
    Helper function. Get the number of times a temperature increases or decreases over a
    specific temperature gradient.

    Parameters
    ------------
    temperature : float series
        Photovoltaic module temperature [C]
    reversal_temp : float
        Temperature threshold to cross above and below [C]

    Returns
    --------
    num_changes_temp_hist : int
        Number of times the temperature threshold is crossed

    """
    # Find the number of times the temperature crosses over 54.8(°C)

    temp_df = pd.DataFrame()
    temp_df["CellTemp"] = temperature
    temp_df["COMPARE"] = temperature
    temp_df["COMPARE"] = temp_df.COMPARE.shift(-1)

    # reversal_temp = 54.8

    temp_df["cross"] = (
        ((temp_df.CellTemp >= reversal_temp) & (temp_df.COMPARE < reversal_temp))
        | ((temp_df.COMPARE > reversal_temp) & (temp_df.CellTemp <= reversal_temp))
        | (temp_df.CellTemp == reversal_temp)
    )

    num_changes_temp_hist = temp_df.cross.sum()

    return num_changes_temp_hist


@geospatial_quick_shape(0, ["damage"])
def solder_fatigue(
    weather_df: pd.DataFrame,
    meta: dict,
    time_range: pd.Series = None,
    temp_cell: pd.Series = None,
    reversal_temp: float = 54.8,
    n: float = 1.9,
    b: float = 0.33,
    C1: float = 405.6,
    Q: float = 0.12,
    wind_factor: float = 0.33,
    temp_model="sapm",
    conf="open_rack_glass_polymer",
    model_kwarg={},
    irradiance_kwarg={},
) -> float:
    """
    Get the Thermomechanical Fatigue of flat plate photovoltaic module solder joints.
    Damage will be returned as the rate of solder fatigue for one year. Based on:

        Bosco, N., Silverman, T. and Kurtz, S. (2020). Climate specific thermomechanical
        fatigue of flat plate photovoltaic module solder joints. [online] Available
        at: https://www.sciencedirect.com/science/article/pii/S0026271416300609
        [Accessed 12 Feb. 2020].

    This function uses the default values for 60-min input intervals from Table 4 of the above
    paper. For other use cases, please refer to the paper for recommended values of C1 and
    the reversal temperature.

    Parameters
    ------------
    weather_df : pd.dataframe
        Must contain dni, dhi, ghi, temp_air, windspeed, and datetime index
    meta : dict
        site location meta-data
    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement heights
        between weather database (e.g. NSRDB) and the tempeature model (e.g. SAPM)
        The NSRDB provides calculations at 2 m (i.e module height) but SAPM uses a 10 m height.
        It is recommended that a power-law relationship between height and wind speed of 0.33
        be used. This results in a wind speed that is 1.7 times higher. It is acknowledged that
        this can vary significantly.
    time_range : timestamp series, optional
        Local time of specific site by the hour year-month-day hr:min:sec
        (Example) 2002-01-01 01:00:00
        If a time range is not give, function will use dt index from weather_df
    temp_cell : float series, optional
        Photovoltaic module temperature [C] for every hour of a year
    reversal_temp : float, optional
        Temperature threshold to cross above and below [C]
        See the paper for other use cases
    n : float
        fit parameter for daily max temperature amplitude
    b : float
        fit parameter for reversal temperature
    C1 : float
        scaling constant, see the paper for details on appropriate values
    Q : float
        activation energy [eV]
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
        Wind speed correction exponent to account for different wind speed measurement heights
        between weather database (e.g. NSRDB) and the tempeature model (e.g. SAPM)
        The NSRDB provides calculations at 2 m (i.e module height) but SAPM uses a 10 m height.
        It is recommended that a power-law relationship between height and wind speed of 0.33
        be used*. This results in a wind speed that is 1.7 times higher. It is acknowledged that
        this can vary significantly.
    irradiance_kwarg : (dict, optional)
        keyword argument dictionary used for the poa irradiance caluation.
        options: ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``. See ``pvdeg.spectral.poa_irradiance``.
    model_kwarg : (dict, optional)
        keyword argument dictionary used for the pvlib temperature model calculation.
        See https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html for more.

    Returns
    --------
    damage : float series
        Solder fatigue damage for a time interval depending on time_range [kPa]

    """

    # TODO this, and many other functions with temp_cell or temp_module would benefit from an
    # optional parameter "conf = 'open_rack_glass_glass' or equivalent"

    # TODO Make this function have more utility.
    # People want to run all the scenarios from the bosco paper.
    # Currently have everything hard coded for hourly calculation
    # i.e. 405.6, 1.9, .33, .12

    # Boltzmann Constant
    k = 0.00008617333262145

    # TODO detect sub-hourly time delta -> downsample to hourly

    if time_range is None:
        time_range = weather_df.index

    if temp_cell is None:
        # temp_cell = temperature.cell(
        #     weather_df=weather_df, meta=meta, wind_factor=wind_factor
        # )
        temp_cell = temperature.temperature(  # we just calculate poa inside
            cell_or_mod="cell",
            weather_df=weather_df,
            meta=meta,
            temp_model=temp_model,
            conf=conf,
            wind_factor=wind_factor,
            irradiance_kwarg=irradiance_kwarg,
            model_kwarg=model_kwarg,
        )

    temp_amplitude, temp_max_avg = _avg_daily_temp_change(time_range, temp_cell)

    temp_max_avg = convert_temperature(temp_max_avg, "Celsius", "Kelvin")

    num_changes_temp_hist = _times_over_reversal_number(temp_cell, reversal_temp)

    damage = (
        C1
        * (temp_amplitude**n)
        * (num_changes_temp_hist**b)
        * np.exp(-(Q / (k * temp_max_avg)))
    )

    # Convert pascals to kilopascals
    damage = damage / 1000

    return damage


# TODO: add gaps functionality


def thermomechanical_driven_rate(
    weather_df=None,
    meta=None,
    weather_kwarg=None,
    A_Tm=2.04,  # <- Outdoors | Inddors -> 9.10e-5,
    E_Tm=0.43,  # <- Outdoors | Inddors -> 0.4,
    theta=2.24,
    const_Boltzmann=8.6171e-5,  # eV/K
    reversal_temp=None,
    tilt=None,
    azimuth=None,
    temp_model="sapm",
    sky_model="isotropic",
    conf_0="insulated_back_glass_polymer",
    wind_factor=0.33,
):
    """
    Calculates the hydrolysis driven rate of degradation.

    Parameters
    ----------
    A_Tm : float
        Pre-exponential factor
    E_Tm : float
        Activation energy
    const_Boltzmann : float
        Boltzmann constant 8.62E-5 [eV/K]
    tilt : float
        Tilt angle of the module [degrees]
    azimuth : float
        Azimuth angle of the module [degrees]
    temp_model : str
        Temperature model to use
    sky_model : str
        Sky model to use
    conf_0 : str
        Configuration of the PV module architecture and mounting
    wind_factor : float
        Wind speed correction exponent to account for different wind speed measurement heights

    Returns
    -------
    thermomechanical_rate : float
        Thermo-mechanical driven rate of degradation [%/s]

    References
    ----------
    doi: 10.1109/JPHOTOV.2019.2916197

    """

    parameters = ["temp_air", "wind_speed", "dhi", "ghi", "dni", "relative_humidity"]

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
    module_temperature = temperature.module(
        weather_df=weather_df,
        meta=meta,
        poa=poa,
        temp_model=temp_model,
        conf=conf_0,
        wind_factor=wind_factor,
    )

    temp_range, temp_max_avg = _avg_daily_temp_change(
        weather_df.index, module_temperature
    )
    temp_max_avg = convert_temperature(temp_max_avg, "Celsius", "Kelvin")

    if reversal_temp is None:
        reversal_temp = temp_range
    num_changes_temp_hist = _times_over_reversal_number(
        module_temperature, reversal_temp
    )

    # cycling_rate = num_changes_temp_hist  # cycles per year
    cycling_rate = 0.5  # TODO: Figure out what this should be

    # Calculate the thermo-mechanical driven rate of degradation
    thermomechanical_rate = (
        A_Tm
        * (temp_range**theta)
        * cycling_rate
        * np.exp(-E_Tm / (const_Boltzmann * temp_max_avg))
        * 100  # Convert to %
    )

    res = {"k_tm": thermomechanical_rate}
    df_res = pd.DataFrame.from_dict(res, orient="index").T

    return df_res
