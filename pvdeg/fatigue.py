"""fatigue.py."""

import numpy as np
import pandas as pd
from scipy.constants import convert_temperature

from pvdeg import temperature, decorators


def _avg_daily_temp_change(time_range, temp_cell):
    """Daily temp change, helper function.

    Get the average of a year for the daily maximum temperature
    change.

    For every 24hrs this function will find the delta between the maximum
    temperature and minimun temperature.  It will then take the deltas for
    every day of the year and return the average delta.

    Parameters
    ----------
    time_range : timestamp series
        Local time of specific site by the hour
        year-month-day hr:min:sec . (Example) 2002-01-01 01:00:00
    temp_cell : float series
        Photovoltaic module cell temperature(Celsius) for every hour of a year

    Returns
    -------
    avg_daily_temp_change : float
        Average Daily Temerature Change for 1-year (Celsius)
    avg_max_temp_cell : float
        Average of Daily Maximum Temperature for 1-year (Celsius)
    """
    if time_range.dtype == "object":
        time_range = pd.to_datetime(time_range)

    # Setup frame for vector processing
    timeAndTemp_df = pd.DataFrame(columns=["Cell Temperature"])
    timeAndTemp_df["Cell Temperature"] = temp_cell
    timeAndTemp_df.index = time_range
    timeAndTemp_df["month"] = timeAndTemp_df.index.month
    timeAndTemp_df["day"] = timeAndTemp_df.index.day

    # Group by month and day to determine the max and min cell Temperature [°C]
    # for each day
    dailyMaxCellTemp_series = timeAndTemp_df.groupby(["month", "day"])[
        "Cell Temperature"
    ].max()
    dailyMinCellTemp_series = timeAndTemp_df.groupby(["month", "day"])[
        "Cell Temperature"
    ].min()
    temp_cell_change = pd.DataFrame(
        {"Max": dailyMaxCellTemp_series, "Min": dailyMinCellTemp_series}
    )
    temp_cell_change["TempChange"] = temp_cell_change["Max"] - temp_cell_change["Min"]

    # Find the average temperature change for every day of one year [°C]
    avg_daily_temp_change = temp_cell_change["TempChange"].mean()
    # Find daily maximum cell temperature average
    avg_max_temp_cell = dailyMaxCellTemp_series.mean()

    return avg_daily_temp_change, avg_max_temp_cell


def _times_over_reversal_number(temp_cell, reversal_temp):
    """Temperature reversal, helper function.

    Get the number of times a temperature increases or decreases
    over a specific temperature gradient.

    Parameters
    ----------
    temp_cell : float series
        Photovoltaic module cell temperature [C]
    reversal_temp : float
        Temperature threshold to cross above and below [C]

    Returns
    -------
    num_changes_temp_hist : int
        Number of times the temperature threshold is crossed
    """
    # Find the number of times the temperature crosses over 54.8(°C)
    temp_df = pd.DataFrame()
    temp_df["CellTemp"] = temp_cell
    temp_df["COMPARE"] = temp_cell
    temp_df["COMPARE"] = temp_df.COMPARE.shift(-1)

    # reversal_temp = 54.8

    temp_df["cross"] = (
        ((temp_df.CellTemp >= reversal_temp) & (temp_df.COMPARE < reversal_temp))
        | ((temp_df.COMPARE > reversal_temp) & (temp_df.CellTemp <= reversal_temp))
        | (temp_df.CellTemp == reversal_temp)
    )

    num_changes_temp_hist = temp_df.cross.sum()

    return num_changes_temp_hist


@decorators.geospatial_quick_shape("numeric", ["damage"])
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
    """Get the Thermomechanical Fatigue of flat plate photovoltaic module solder joints.

    Damage will be returned as the rate of solder fatigue for one year. Based on:

        Bosco, N., Silverman, T. and Kurtz, S. (2020). Climate specific thermomechanical
        fatigue of flat plate photovoltaic module solder joints. [online] Available
        at: https://www.sciencedirect.com/science/article/pii/S0026271416300609
        [Accessed 12 Feb. 2020].

    This function uses the default values for 60-min input intervals from Table 4 of the
    above paper. For other use cases, please refer to the paper for recommended values
    of C1 and the reversal temperature.

    Parameters
    ----------
    weather_df : pd.dataframe
        Must contain dni, dhi, ghi, temp_air, windspeed, and datetime index
    meta : dict
        site location meta-data
    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement
        heights between weather database (e.g. NSRDB) and the tempeature model
        (e.g. SAPM). The NSRDB provides calculations at 2 m (i.e module height) but SAPM
        uses a 10m height. It is recommended that a power-law relationship between
        height and wind speed of 0.33 be used*. This results in a wind speed that is
        1.7 times higher. It is acknowledged that this can vary significantly.
    time_range : timestamp series, optional
        Local time of specific site by the hour year-month-day hr:min:sec
        (Example) 2002-01-01 01:00:00
        If a time range is not give, function will use dt index from weather_df
    temp_cell : float series, optional
        Photovoltaic module cell temperature [C] for every hour of a year
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
        Wind speed correction exponent to account for different wind speed measurement
        heights between weather database (e.g. NSRDB) and the tempeature model
        (e.g. SAPM). The NSRDB provides calculations at 2 m (i.e module height) but SAPM
        uses a 10m height. It is recommended that a power-law relationship between
        height and wind speed of 0.33 be used*. This results in a wind speed that is
        1.7 times higher. It is acknowledged that this can vary significantly.
    irradiance_kwarg : (dict, optional)
        keyword argument dictionary used for the poa irradiance caluation.
        options: ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``.
        See ``pvdeg.spectral.poa_irradiance``.
    model_kwarg : (dict, optional)
        keyword argument dictionary used for the pvlib temperature model calculation.
        See https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html  # noqa
        for more.

    Returns
    -------
    damage : float series
        Solder fatigue damage for a time interval depending on time_range [kPa]
    """
    # TODO this, and many other functions with temp_cell or temp_module would benefit
    # from an optional parameter "conf = 'open_rack_glass_glass' or equivalent"

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
