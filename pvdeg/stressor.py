"""
Collection of Classes and Functions To Create Chamber Exposure Stressor Conditions1
"""

import numpy as np
import pandas as pd
from numba import njit

from typing import (
    Union, 
    List
)

@njit
def num_steps(
    step_time: Union[int, float], 
    resolution: Union[int, float],
    )->int:
    """
    Find number of time steps required for a ramp rate.
    """
    return int(np.ceil( step_time / resolution )) 
    # is this what we want. the numbers being inputteed could cause weird things but rouning up is correct?

@njit
def linear_ramp_time(
    y_0: Union[int, float],
    y_f: Union[int, float],
    rate: Union[int, float],
    )->float:
    """
    Calculate total time to go from an intial condition to the final condition
    a rate in the same units with respect to time. Where rate is the absolute value
    """
    time = (y_f - y_0) / rate
    return np.abs( time )    

# TODO: make this work with instant rate (use 0 for this)
def fill_linear_region(
    step_time: Union[int, float],
    step_time_resolution: Union[int, float], 
    set_0: Union[int, float],
    set_f: Union[int, float],
    rate: Union[int, float],
    )->np.array:
    """
    Populate a setpoint timeseries of chamber setpoints with variable ramp rates.

    Parameters:
    -----------
    step_time: Union[int, float]
        duration of current step
    step_time_resolution:
        precision to use for each step (deltaT)
    set_0: Union[int, float]
        intitial condition value
    set_f: Union[int, float]
        set point value
    rate: Union[int, float]
        speed at which to ramp from the intial condition
        to the new setpoint with respect to time.
    """
    if rate != 0:
        ramp_time = linear_ramp_time(
            y_0=set_0,
            y_f=set_f,
            rate=rate
        )

        if ramp_time > step_time:
            raise ValueError("Ramp speed is too slow, will not finish ramping up before next set point")

        ramp_steps = num_steps(
        step_time=ramp_time,
        resolution=step_time_resolution

        )
    else:
        ramp_steps = 0

    total_steps = num_steps(
        step_time=step_time,
        resolution=step_time_resolution
    )

    flat_steps = total_steps - ramp_steps

    if ramp_steps != 0:
        set_points = np.linspace(set_0, set_f, ramp_steps)
    else:
        set_points = np.array([])

    if flat_steps > 0:
        set_points_flat = np.full(flat_steps, set_f)
        set_points = np.concatenate((set_points, set_points_flat))

    return set_points

# could add intial conditions instead of nans where for previous values at row 0
def add_previous_setpoints(
    df: pd.DataFrame
    )-> pd.DataFrame:
    """
    Update the setpoint dataframe to contain previous setpoint information in current row
    """
    mapping = ['temperature', 'relative_humidity','irradiance','voltage']
    col_names = ['previous_temperature', 'previous_relative_humidity', 'previous_irradiance', 'previous_voltage']
    df[col_names] = df[mapping].shift(1) 
    return df

# instant change or linear change
def _apply_fill_linear_region(
    row,
    column_name: str,
    )-> np.ndarray:
    """
    Create a 1d numpy array containing a linear region of setpoint values

    Parameters:
    -----------
    row: 
        row of a dataframe using df.apply
    column_name: str
        name of the column to target in the dataframe
    """

    values = fill_linear_region(
        step_time=row['step_length'],
        step_time_resolution=row['time_resolution'],
        set_0=row[f'previous_{column_name}'],
        set_f=row[column_name],
        rate=row[f'{column_name}_ramp']
    )
    return values

def series_index(
    start_time: Union[int, float], 
    step_length: Union[int, float], 
    resoultion: Union[int, float]
    )-> np.ndarray[int, float]:
    """
    Generate time indexes in a fixed resoultion for a setpoint.

    Parameters:
    -----------
    start_time: Union[int, float]
    step_length: Union[int, float]
    resolution: Union[int, float]

    Return:
    -------
    values : np.ndarray
    """
    values = np.arange(
        start=start_time,
        stop=start_time + step_length,
        step=resoultion
        )

    return values

def flat_restore_index(series):
    indicies = np.array([])
    for index, sub_series in series.items():
        indicies = np.append(indicies, sub_series.index.values)

    flat_setpoints = series.explode()

    flat_setpoints.index = indicies

    return flat_setpoints 


def chamber_set_points_timeseries(
    df: pd.DataFrame,
    setpoint_name: str,
    )-> np.array:
    """
    Create a pandas series containing values for a specific setpoint using 
    set points and ramp values.

    Parameters:
    -----------
    df: pd.DataFrame
        pandas dataframe containing columns named for each setpoint and previous setpoint and setpoint ramp values
        >>> ex) temperature set point series:
        >>> dataframe should contain the following columns 
        >>> 'temperature', 'previous_temperature', 'temperature_ramp'
    """
    df = add_previous_setpoints(df)

    setpoints_np_series = df.apply(lambda row: _apply_fill_linear_region(row, column_name=setpoint_name), axis=1)

    setpoints_series = pd.Series()

    for index, row in setpoints_np_series.items():
        # if row.any(): # maybe failing because of zeros
        if row.size > 0:
            new_index = series_index(
                start_time=df.loc[index, 'start_time'],
                step_length=df.loc[index, 'step_length'],
                resoultion=df.loc[index, 'time_resolution']
            )
            setpoints_series[index] = pd.Series(row, index=new_index)

    setpoints_timesereies = flat_restore_index(setpoints_series)

    return setpoints_timesereies

# RENAME THIS, everything up to here should be encapsulated in one function 
def apply_chamber_set_points(
    df: pd.DataFrame,
    setpoint_names: List[str] = ['temperature', 'relative_humidity', 'irradiance', 'voltage']
    )-> pd.DataFrame:
    """
    Apply the chamber_set_points_timeseries function for multiple setpoints.

    Parameters:
    -----------
    df: pd.DataFrame
        pandas dataframe containing the setpoint data.
    setpoint_names: List[str]
        List of setpoint names to process.

    Returns:
    ---------
    setpoints_time_df: pd.DataFrame
        dataframe of linearly ramped and instant change setpoints from
        user defined set points.
    """

    setpoints_time_df = pd.DataFrame()

    for name in setpoint_names:
        series = chamber_set_points_timeseries(df=df, setpoint_name=name)
        series.name = name # i hate this
        setpoint_df = series.to_frame()
    
        setpoints_time_df = pd.concat([setpoints_time_df, setpoint_df], axis=1)

    return setpoints_time_df


# TODO: add initial conditions check, can come from csv or args
def create_set_point_df(
    t_0: Union[int, float], 
    rh_0: Union[int, float], 
    irrad_0: Union[int, float], 
    v_0: Union[int, float], 
    fp: str
    )-> pd.DataFrame:
    """
    Create a dataframe of chamber set points from a csv and set of initial conditions.

    Parameters:
    -----------
    t_0 : Union[int, float]
        inital temperature [C]
    rh_0 : Union[int, float]
        inital relative humidity [unitless]
    irrad_0 : Union[int, float]
        inital irradiance at 340 nm [?]
    v_0 : Union[int, float]
        intial voltage [V]
    fp : str
        file path to csv with set points
    
    Returns:
    --------
    set_points_df : pd.DataFrame
        dataframe containing inital conditons and set points
    """
    columns=['step_length', 'temperature', 'temperature_ramp', 'relative_humidity', 'relative_humidity_ramp', 'irradiance', 'irradiance_ramp', 'voltage', 'voltage_ramp', 'time_resolution']
    data = {0 : [0, t_0, 0, rh_0, 0, irrad_0, 0, v_0, 0, 1]}
    
    initial_conditions = pd.DataFrame.from_dict(data, orient='index', columns=columns)
    set_points_directive = pd.read_csv(fp)
    set_points_directive = set_points_directive[columns]

    set_point_df = pd.concat([initial_conditions, set_points_directive], axis=0, ignore_index=True)

    return set_point_df

# TODO: Fix intial condition behavior
def start_times(
    df: pd.DataFrame
    )-> pd.DataFrame:
    """
    Add a column called `'start_time'` to the dataframe containing the start time 
    for each set point in minutes
    """

    df.loc[0, 'start_time'] = 0

    for i in range(1, len(df.index)):
        df.loc[i, 'start_time'] = df.loc[i - 1, 'start_time'] + df.loc[i - 1, 'step_length']

    return df

# TODO: change to datetime or timedelta index?
def chamber_setpoints(
    fp: str,
    t_0: Union[int, float], 
    rh_0: Union[int, float], 
    irrad_0: Union[int, float], 
    v_0: Union[int, float], 
    setpoint_names: List[str] = ['temperature', 'relative_humidity', 'irradiance', 'voltage']
    )-> pd.DataFrame:

    """
    Parameters:
    -----------
    fp : str
        file path to csv with set points
    t_0 : Union[int, float]
        inital temperature [C]
    rh_0 : Union[int, float]
        inital relative humidity [unitless]
    irrad_0 : Union[int, float]
        inital irradiance at 340 nm [?]
    v_0 : Union[int, float]
        intial voltage [V]
    setpoint_names: List[str]
        list of column names to create setpoint timeseries for.
        all list entries must exist in dataframe/csv/intial values.
    """

    set_point_df = create_set_point_df(
        t_0=t_0,
        rh_0=rh_0,
        irrad_0=irrad_0,
        v_0=v_0,
        fp=fp
    )

    set_point_starts_df = start_times(set_point_df)

    set_values_df = apply_chamber_set_points(
        df=set_point_starts_df,
        setpoint_names=setpoint_names
    )

    return set_values_df