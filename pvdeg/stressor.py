"""
Collection of Classes and Functions To Create Chamber Exposure Stressor Conditions
"""

import numpy as np
import pandas as pd
from numba import njit

from pvdeg.humidity import (
    water_vapor_pressure,
    chamber_dew_point_from_vapor_pressure,
    rh_at_sample_temperature,
    rh_internal_cell_backside,
    equilibrium_eva_water,
)

from pvdeg.temperature import chamber_sample_temperature
from pvdeg.utilities import _shift

from typing import Union, List

# TODO: vectorize numpy calcs

### Questions for Mike:

# Where do the constants in the water vapor pressure equation come from

# do we need set point temp -> real chamber temp -> real sample temp?
# what is special about 0.4 w/m^2/nm at 340 nm in the temperature.chamber_sample_temp function also should there be a seconary air temperature function?


@njit
def num_steps(
    step_time: Union[int, float],
    resolution: Union[int, float],
) -> int:
    """
    Find number of time steps required for a ramp rate.
    """
    return int(np.ceil(step_time / resolution))
    # is this what we want. the numbers being inputteed could cause weird things but rouning up is correct?


@njit
def linear_ramp_time(
    y_0: Union[int, float],
    y_f: Union[int, float],
    rate: Union[int, float],
) -> float:
    """
    Calculate total time to go from an intial condition to the final condition
    a rate in the same units with respect to time. Where rate is the absolute value
    """
    time = (y_f - y_0) / rate
    return np.abs(time)


# TODO: make this work with instant rate (use 0 for this)
def fill_linear_region(
    step_time: Union[int, float],
    step_time_resolution: Union[int, float],
    set_0: Union[int, float],
    set_f: Union[int, float],
    rate: Union[int, float],
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


# could add intial conditions instead of nans where for previous values at row 0
def add_previous_setpoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Update the setpoint dataframe to contain previous setpoint information in current row
    """
    mapping = ["temperature", "relative_humidity", "irradiance", "voltage"]
    col_names = [
        "previous_temperature",
        "previous_relative_humidity",
        "previous_irradiance",
        "previous_voltage",
    ]
    df[col_names] = df[mapping].shift(1)
    return df


# instant change or linear change
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
        step_time_resolution=row["time_resolution"],
        set_0=row[f"previous_{column_name}"],
        set_f=row[column_name],
        rate=row[f"{column_name}_ramp"],
    )
    return values


def series_index(
    start_time: Union[int, float],
    step_length: Union[int, float],
    resoultion: Union[int, float],
) -> np.ndarray[int, float]:
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
    values = np.arange(start=start_time, stop=start_time + step_length, step=resoultion)

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
) -> np.array:
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

    setpoints_np_series = df.apply(
        lambda row: _apply_fill_linear_region(row, column_name=setpoint_name), axis=1
    )

    setpoints_series = pd.Series()

    for index, row in setpoints_np_series.items():
        # if row.any(): # maybe failing because of zeros
        if row.size > 0:
            new_index = series_index(
                start_time=df.loc[index, "start_time"],
                step_length=df.loc[index, "step_length"],
                resoultion=df.loc[index, "time_resolution"],
            )
            setpoints_series[index] = pd.Series(row, index=new_index)

    setpoints_timesereies = flat_restore_index(setpoints_series)

    return setpoints_timesereies


def apply_chamber_set_points(
    df: pd.DataFrame,
    setpoint_names: List[str] = [
        "temperature",
        "relative_humidity",
        "irradiance",
        "voltage",
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
        series = chamber_set_points_timeseries(df=df, setpoint_name=name)
        series.name = name  # i hate this
        setpoint_df = series.to_frame()

        setpoints_time_df = pd.concat([setpoints_time_df, setpoint_df], axis=1)

    return setpoints_time_df


# TODO: add initial conditions check, can come from csv or args
def create_set_point_df(
    t_0: Union[int, float],
    rh_0: Union[int, float],
    irrad_0: Union[int, float],
    v_0: Union[int, float],
    fp: str,
) -> pd.DataFrame:
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
    columns = [
        "step_length",
        "temperature",
        "temperature_ramp",
        "relative_humidity",
        "relative_humidity_ramp",
        "irradiance",
        "irradiance_ramp",
        "voltage",
        "voltage_ramp",
        "time_resolution",
    ]
    data = {0: [0, t_0, 0, rh_0, 0, irrad_0, 0, v_0, 0, 1]}

    initial_conditions = pd.DataFrame.from_dict(data, orient="index", columns=columns)
    set_points_directive = pd.read_csv(fp)
    set_points_directive = set_points_directive[columns]

    set_point_df = pd.concat(
        [initial_conditions, set_points_directive], axis=0, ignore_index=True
    )

    return set_point_df


# TODO: Fix intial condition behavior
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


def chamber_setpoints(
    fp: str,
    t_0: Union[int, float],
    rh_0: Union[int, float],
    irrad_0: Union[int, float],
    v_0: Union[int, float],
    setpoint_names: List[str] = [
        "temperature",
        "relative_humidity",
        "irradiance",
        "voltage",
    ],
) -> pd.DataFrame:
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
        t_0=t_0, rh_0=rh_0, irrad_0=irrad_0, v_0=v_0, fp=fp
    )

    set_point_starts_df = start_times(set_point_df)

    set_values_df = apply_chamber_set_points(
        df=set_point_starts_df, setpoint_names=setpoint_names
    )

    minutes = set_values_df.index.values.astype(int)
    timedeltas_index = minutes.astype("timedelta64[m]")
    set_values_df.index = timedeltas_index

    return set_values_df


@njit
def _calc_water_vap_pres(temp_numpy: np.ndarray, rh_numpy: np.ndarray) -> np.ndarray:
    water_vap_pres = np.empty_like(temp_numpy)

    for i in range(water_vap_pres.shape[0]):
        water_vap_pres[i] = water_vapor_pressure(temp=temp_numpy[i], rh=rh_numpy[i])

    return water_vap_pres


@njit
def _calc_dew_point(water_vap_pres_numpy: np.ndarray) -> np.ndarray:
    dew_point = np.empty_like(water_vap_pres_numpy)

    for i in range(dew_point.shape[0]):
        dew_point[i] = chamber_dew_point_from_vapor_pressure(
            water_vap_pres=water_vap_pres_numpy[i]
        )

    return dew_point


# tau and times should both be in minutes
# this cannot be vectorized but the others could
def _calc_sample_temperature(
    irradiance_340: np.ndarray[float],
    temp_set: np.ndarray[float],
    times: np.ndarray[np.timedelta64],
    tau: float,
    chamber_irrad_0: float,
    sample_temp_0: float,
) -> np.ndarray[float]:
    """
    Apply finite difference temperatures to chamber conditions.

    Parameters:
    -----------
    irradiance_340: np.ndarray[float]
        UV irradiance [W/m^2/nm at 340 nm]
    temp_set: np.ndarray[float]
        chamber temperature setpoint
    times: np.ndarray[float]
        length of timestep (end time - start time) [min]
    tau: float
        Characteristic thermal equilibration time [min]
    chamber_irrad_0: float
        inital chamber UV irradiance [W/m^2/nm at 340 nm]
    sample_temp_0: float
        intial sample temperature [C]

    Returns:
    --------
    sample_temperatures: np.ndarray[float]
        array of sample temperatures

    """

    previous_times = _shift(times, 1)
    differences = np.subtract(times, previous_times)
    minute_times = differences.astype(
        "timedelta64[m]"
    ).astype(
        int
    )  # do we want to do this, they might be in minutes already so we can drop this part

    sample_temperatures = np.empty_like(minute_times)

    sample_temperatures[0] = sample_temp_0
    irradiance_340[0] = chamber_irrad_0

    for i in range(1, minute_times.shape[0]):
        sample_temperatures[i] = chamber_sample_temperature(
            irradiance_340=irradiance_340[i],
            previous_sample_temp=sample_temperatures[i - 1],
            temp_set=temp_set[i],
            delta_t=minute_times[i],
            tau=tau,
        )

    return sample_temperatures


@njit
def _calc_rh(
    temp_set: np.ndarray[float],
    rh_set: np.ndarray[float],
    sample_temp: np.ndarray[float],
) -> np.ndarray[float]:
    rh = np.empty_like(sample_temp)

    for i in range(sample_temp.shape[0]):
        rh[i] = rh_at_sample_temperature(
            temp_set=temp_set[i], rh_set=rh_set[i], sample_temp=sample_temp[i]
        )

    return rh


# TODO: fix intial conditons entered during the set point dataframe, they could just be moved here instead of having them above.
# could also combine into one function it would be much easier that way potentially. explore this as an option.
def chamber_properties(
    set_point_df: pd.DataFrame,
    tau: float,
    chamber_irrad_0: float,
    sample_temp_0: float,
    eva_solubility: float,
    solubility_prefactor: float,
) -> pd.DataFrame:
    """
    Create a dataframe with sample properties at each time step.
    Includes `'water_vapor_pressure'`, `'dew_point'`, '`sample_temperature'`

    Parameters:
    -----------
    set_point_df: pd.DataFrame
        dataframe containing a setpoint timeseries with a timdelta index.
        generated using `pvdeg.stressor.chamber_setpoints`
    tau: float
        Characteristic thermal equilibration time [min]
    chamber_irrad_0: float
        inital chamber UV irradiance [W/m^2/nm at 340 nm]
    sample_temp_0: float
        intial sample temperature [C]
    eva_solubility:
        activation energy for solubility in EVA [eV]
    solubility_prefactor:
        amount of substance already present [g/cm^3]
        >>> should this just say water present at t=0


    Returns:
    --------
    properties_df: pd.DataFrame
        DataFrame containing chamber properties at each timedelta present
        in the setpoints dataframe. Contains columns for:
        >>> 'water_vapor_pressure', 'dew_point', 'sample_temperature'
    """

    properties_df = pd.DataFrame(index=set_point_df.index)

    water_vapor_pressures = _calc_water_vap_pres(
        temp_numpy=set_point_df["temperature"].to_numpy(dtype=np.float64),
        rh_numpy=set_point_df["relative_humidity"].to_numpy(dtype=np.float64),
    )

    dew_points = _calc_dew_point(water_vap_pres_numpy=water_vapor_pressures)

    sample_temperatures = _calc_sample_temperature(
        irradiance_340=set_point_df["irradiance"].to_numpy(dtype=np.float64),
        temp_set=set_point_df["temperature"].to_numpy(dtype=np.float64),
        times=set_point_df.index.to_numpy().astype("timedelta64[m]"),
        sample_temp_0=sample_temp_0,
        chamber_irrad_0=chamber_irrad_0,
        tau=tau,
    )

    rh_sample_temp = _calc_rh(
        temp_set=set_point_df["temperature"].to_numpy(dtype=np.float64),
        rh_set=set_point_df["relative_humidity"].to_numpy(dtype=np.float64),
        sample_temp=sample_temperatures,
    )

    eq_eva_water = equilibrium_eva_water(
        sample_temp=sample_temperatures,
        rh_at_sample_temp=rh_sample_temp,
        eva_solubility=eva_solubility,
        solubility_prefactor=solubility_prefactor,
    )

    # where does this come from
    back_eva_moisture_content = ...

    rh_backside_cells = rh_internal_cell_backside(
        back_eva_moisture=back_eva_moisture_content, # this line wont work
        equilibrium_eva_water=eq_eva_water,
        rh_at_sample_temp=rh_sample_temp
    )


    properties_df["water_vapor_pressure"] = water_vapor_pressures
    properties_df["dew_point"] = dew_points
    properties_df["sample_temperature"] = sample_temperatures
    properties_df["rh_at_sample_temp"] = rh_sample_temp
    properties_df["equilibrium_eva_water"] = eq_eva_water

    # this is not right, where do these values come from 
    properties_df["back_eva_moisture_content"] = ... # eq_eva_water 

    return properties_df


class ChamberStressor:

    def __init__(
        self,
        eva_diffusivity,
        eva_diffusivity_prefactor,
        eva_thickness,
        eva_solubility,
        eva_solubility_prefactor,
        pet_permeability,
        pet_permeability_prefactor,
        pet_thickness,
        cell_edge_lengths,
        k,
    ):
        """Define Chamber Settings"""
        self.eva_diffusivity = eva_diffusivity
        self.eva_diffusivity_prefactor = eva_diffusivity_prefactor
        self.eva_thickness = eva_thickness
        self.eva_solubility = eva_solubility
        self.eva_solubility_prefactor = eva_solubility_prefactor
        self.pet_permeability = pet_permeability
        self.pet_permeability_prefactor = pet_permeability_prefactor
        self.pet_thickness = pet_thickness
        self.cell_edge_lengths = cell_edge_lengths
        self.k = k

    def add_setpoints(
        self,
        fp,
        t_0,
        rh_0,
        irrad_0,
        setpoint_names=["temperature", "relative_humidity", "irradiance", "voltage"],
    ):
        self.set_points = chamber_setpoints(
            fp=fp, t_0=t_0, rh_0=rh_0, irrad_0=irrad_0, setpoint_names=setpoint_names
        )

    def calculate_chamber_properties(self, tau, chamber_irrad_0, sample_temp_0):
        self.chamber_properties = chamber_properties(
            self.set_points,
            tau=tau,
            chamber_irrad_0=chamber_irrad_0,
            sample_temp_0=sample_temp_0,
        )
