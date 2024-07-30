import pandas as pd
import numpy as np
from numba import njit


def start_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column called `'start_time'` to the dataframe containing the start time
    for each set point in minutes
    """

    df.loc[0, "start_time"] = 0

    for i in range(1, len(df.index)):
        df.loc[i, "start_time"] = (
            df.loc[i - 1, "start_time"] + df.loc[i - 1, "step_length"]
        )
    return df


def add_previous_setpoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Update the setpoint dataframe to contain previous setpoint information in current row
    """
    mapping = [
        "temperature",
        "relative_humidity",
        "voltage",
        "irradiance_340",
        "irradiance_300-400",
        "irradiance_full",
    ]
    col_names = [
        "previous_temperature",
        "previous_relative_humidity",
        "previous_voltage",
        "previous_irradiance_340",
        "previous_irradiance_300-400",
        "previous_irradiance_full",
    ]
    df[col_names] = df[mapping].shift(1)
    return df


def setpoints_index(start_time, step_length, step_divisions) -> pd.RangeIndex:
    index = pd.RangeIndex(
        start=start_time,
        stop=start_time + step_length,
        step=(step_length / step_divisions),
    )
    return index


@njit
def num_steps(
    step_time: float,
    resolution: float,
) -> int:
    """
    Find number of time steps required for a ramp rate.
    """
    return int(np.ceil(step_time / resolution))


@njit
def linear_ramp_time(
    y_0: float,
    y_f: float,
    rate: float,
) -> float:
    """
    Calculate total time to go from an intial condition to the final condition
    a rate in the same units with respect to time. Where rate is the absolute value
    """
    time = (y_f - y_0) / rate
    return np.abs(time)


def fill_linear_region(
    step_time: float,
    step_time_resolution: float,
    set_0: float,
    set_f: float,
    rate: float,
) -> np.array:
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
        ramp_time = linear_ramp_time(y_0=set_0, y_f=set_f, rate=rate)

        if ramp_time > step_time:
            raise ValueError(
                "Ramp speed is too slow, will not finish ramping up before next set point"
            )

        ramp_steps = num_steps(step_time=ramp_time, resolution=step_time_resolution)
    else:
        ramp_steps = 0

    total_steps = num_steps(step_time=step_time, resolution=step_time_resolution)

    flat_steps = total_steps - ramp_steps

    if ramp_steps != 0:
        set_points = np.linspace(set_0, set_f, ramp_steps)
    else:
        set_points = np.array([])

    if flat_steps > 0:
        set_points_flat = np.full(flat_steps, set_f)
        set_points = np.concatenate((set_points, set_points_flat))

    return set_points


def _apply_fill_linear_region(
    row,
    column_name: str,
) -> np.ndarray:
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
        step_time=row["step_length"],
        step_time_resolution=(
            row["step_length"] / row["step_divisions"]
        ),  # find the time resolution for within the step
        set_0=row[f"previous_{column_name}"],
        set_f=row[column_name],
        rate=row[f"{column_name}_ramp"],
    )
    return values


def flat_restore_index(series):
    indicies = np.array([])
    for index, sub_series in series.items():
        indicies = np.append(indicies, sub_series.index.values)

    flat_setpoints = series.explode()

    flat_setpoints.index = indicies

    return flat_setpoints


def setpoint_series(df: pd.DataFrame, setpoint_name: str) -> pd.Series:
    if df[setpoint_name].isnull().sum():
        return ValueError(
            f"column: {setpoint_name} contains NaN values. Remove from setpoint list or remove NaN's in input."
        )

    setpoints_np_series = df.apply(
        lambda row: _apply_fill_linear_region(row, column_name=setpoint_name),
        axis=1,
    )

    setpoints_2d = pd.Series()

    for i in range(df.shape[0]):
        index = setpoints_index(
            start_time=df.loc[i, "start_time"],
            step_length=df.loc[i, "step_length"],
            step_divisions=df.loc[i, "step_divisions"],
        )

        setpoints_2d[i] = pd.Series(setpoints_np_series[i], index)

    return flat_restore_index(setpoints_2d)


def apply_setpoints_series(
    df: pd.DataFrame,
    setpoint_names: list[str] = [
        "temperature",
        "relative_humidity",
        "voltage",
        "irradiance_340",
        "irradiance_300-400",
        "irradiance_full",
    ],
) -> pd.DataFrame:
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
        series = setpoint_series(df=df, setpoint_name=name)
        series.name = name
        setpoint_df = series.to_frame()

        setpoints_time_df = pd.concat([setpoints_time_df, setpoint_df], axis=1)

    return setpoints_time_df


def setpoints_timeseries_from_csv(
    fp: str,
    setpoint_names: list[str] = [
        "temperature",
        "relative_humidity",
        "voltage",
        "irradiance_340",
        "irradiance_300-400",
        "irradiance_full",
    ],
    skiprows: list[int] = [1],
) -> pd.DataFrame:
    set_points_directive = pd.read_csv(fp, skiprows=skiprows)  # skip row with units

    set_points_directive = start_times(set_points_directive)  # add start time column
    set_points_complete = add_previous_setpoints(
        set_points_directive
    )  # add shifted columns

    timeseries_setpoints = apply_setpoints_series(
        set_points_complete, setpoint_names=setpoint_names
    )

    minutes = timeseries_setpoints.index.values.astype(int)
    timedeltas_index = minutes.astype("timedelta64[m]")
    timeseries_setpoints.index = timedeltas_index

    return timeseries_setpoints
