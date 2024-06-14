"""
Collection of Classes and Functions To Create Chamber Exposure Stressor Conditions1
"""

import numpy as np
import pandas as pd
from typing import Union


def chamber_set_points_timeseries(row):
    ...

def ramp(x_0, x_f, rate, time):
    ...

# TODO: add initial conditions check, can come from csv or args
def create_set_point_df(
        t_0: Union[int, float], 
        rh_0: Union[int, float], 
        irrad_0: Union[int, float], 
        v_0: Union[int, float], 
        fp: str
        )->pd.DataFrame:
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