"""
Collection of classes, methods and functions to calculate chamber stress test and chamber sample conditions.
"""

# TODO: cleanup all chamber methods, they are heinous to look at

import pandas as pd
import numpy as np
from numba import njit
from typing import Union
import json
import os
import matplotlib.pyplot as plt
from IPython.display import display, HTML

import io
import base64

from pvdeg import (
    humidity,
    utilities,
    spectral,
    DATA_DIR,
)

from pvdeg.temperature import (
    fdm_temperature,
    fdm_temperature_irradiance,
)


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


# this tries to shift all of these even if they dont exist
# def add_previous_setpoints(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Update the setpoint dataframe to contain previous setpoint information in current row
#     """
#     mapping = [
#         "temperature",
#         "relative_humidity",
#         "voltage",
#         "irradiance_340",
#         "irradiance_300-400",
#         "irradiance_full",
#     ]
#     col_names = [
#         "previous_temperature",
#         "previous_relative_humidity",
#         "previous_voltage",
#         "previous_irradiance_340",
#         "previous_irradiance_300-400",
#         "previous_irradiance_full",
#     ]
#     for setpoint, shifted_setpoint in zip(mapping, col_names):
#         if setpoint in df.columns:
#             df[shifted_setpoint] = df[setpoint].shift(1)

#     return df


def add_previous_setpoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create setpoints columns with values shifted by 1.
    At row k, the shifted value will come from row k - 1.
    """

    ignore_columns = {
        "step_length",
        "step_divisions",
    }

    # only care about setpoints, not ramp rates
    setpoints = set(df.columns).difference(ignore_columns)
    setpoint_names = {name.rstrip("_ramp") for name in setpoints}

    for name in setpoint_names:
        df[f"previous_{name}"] = df[name].shift(1)

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

        if ramp_time > step_time:  # adding a tolerance
            # not np.isclose(ramp_time, step_time, rtol=0.01):
            # if ramp_time > step_time:
            raise ValueError(
                f"""
                Ramp speed is too slow, will not finish ramping up before next set point.
                ramp will take {ramp_time} minutes but step is only {step_time} minutes.
                """
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

    if pd.isna(row[f"previous_{column_name}"]):
        row[f"previous_{column_name}"] = row[column_name]

    values = fill_linear_region(
        step_time=row["step_length"],
        step_time_resolution=(
            row["step_length"] / row["step_divisions"]
        ),  # find the time resolution for within the step
        set_0=row[
            f"previous_{column_name}"
        ],  # this will be wrong when we shift the first row for the nan
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
        raise ValueError(
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

    timeseries_setpoints.columns = [
        f"setpoint_{name}" for name in timeseries_setpoints.columns.tolist()
    ]

    return timeseries_setpoints


# @njit
def _temp_calc_no_irradiance(
    temps: np.ndarray, times: np.ndarray, tau: float, temp_0: float
) -> np.ndarray:
    res = np.empty_like(temps)
    res[0] = temp_0

    for i in range(0, temps.shape[0] - 1):
        delta_t = times[i + 1] - times[i]
        delta_t = delta_t.astype("timedelta64[m]").astype(int)

        res[i + 1] = fdm_temperature(
            t_current=res[i],
            t_set=temps[i + 1],
            delta_time=delta_t,
            tau=tau,
        )

    return res


# we can move a calculation outside of the loop
# the temperature increase from irradiance at a timestep has a constant factor of
# K = surface area * absorptance so we can bring this out of the loop if we refactor the temperature irradiance

# @njit
def _temp_calc_irradiance(
    temps: np.ndarray,
    times: np.ndarray,
    irradiances: np.ndarray,
    tau: float,
    temp_0: float,
    surface_area: float,
    absorptance: float,
) -> np.ndarray:
    res = np.empty_like(temps)
    res[0] = (
        temp_0  # initial irradiance has no effect on intial temperature because it is at an instant so the accumulated irradiance (integral of G = 0)
    )

    for i in range(0, temps.shape[0] - 1):
        delta_t = times[i + 1] - times[i]
        delta_t = delta_t.astype("timedelta64[m]").astype(int)

        res[i + 1] = fdm_temperature_irradiance(
            t_current=res[i],
            t_set=temps[i + 1],
            irradiance=irradiances[i],
            delta_time=delta_t,
            tau=tau,
            surface_area=surface_area,
            absorptance=absorptance,
        )

    return res


# calculate the air temperature in a chamber using finite difference method
def air_temperature(
    setpoints_df: pd.DataFrame, tau_c: float, air_temp_0: float
) -> np.ndarray:
    air_temp = _temp_calc_no_irradiance(
        temps=setpoints_df["setpoint_temperature"].to_numpy(),
        times=setpoints_df.index.to_numpy(),
        tau=tau_c,
        temp_0=air_temp_0,
    )

    return pd.Series(air_temp, index=setpoints_df.index, name="Air Temperature")


def sample_temperature(
    setpoints_df: pd.DataFrame,
    air_temperature: Union[pd.Series, np.ndarray],
    tau_s: float,
    sample_temp_0: float,
    surface_area: float = None,
    absorptance: float = None,
) -> np.ndarray:
    if isinstance(air_temperature, pd.Series):
        air_temperature = air_temperature.to_numpy()

    if not isinstance(air_temperature, np.ndarray):
        raise ValueError("Air_temperature must be a numpy array or pandas series")

    if "setpoint_irradiance_full" in setpoints_df.columns:
        sample_temp = _temp_calc_irradiance(
            temps=air_temperature,
            irradiances=setpoints_df["setpoint_irradiance_full"].to_numpy(),
            times=setpoints_df.index.to_numpy(),
            tau=tau_s,
            temp_0=sample_temp_0,
            surface_area=surface_area,
            absorptance=absorptance,
        )

    else:
        print(f"""
              "setpoint_irradiance_full" not in setpoints_df.columns 
              Current column names {setpoints_df.columns}.
              calculating sample temperature without irradiance"
              """)

        sample_temp = _temp_calc_no_irradiance(
            temps=air_temperature,
            times=setpoints_df.index.to_numpy(),
            tau=tau_s,
            temp_0=sample_temp_0,
        )

    return pd.Series(sample_temp, index=setpoints_df.index, name="Sample Temperature")


# @njit
def _calc_water_vap_pres(temp_numpy: np.ndarray, rh_numpy: np.ndarray) -> np.ndarray:
    water_vap_pres = np.empty_like(temp_numpy)

    for i in range(water_vap_pres.shape[0]):
        water_vap_pres[i] = humidity.water_vapor_pressure(
            temp=temp_numpy[i], rh=rh_numpy[i]
        )

    return water_vap_pres


@njit
def _calc_rh(
    temp_set: np.ndarray[float],
    rh_set: np.ndarray[float],
    sample_temp: np.ndarray[float],
) -> np.ndarray[float]:
    rh = np.empty_like(sample_temp)

    for i in range(sample_temp.shape[0]):
        rh[i] = humidity.rh_at_sample_temperature(
            temp_set=temp_set[i], rh_set=rh_set[i], sample_temp=sample_temp[i]
        )
    return rh


class Sample:
    def __init__(
        self,
        backsheet=None,
        backsheet_thickness=None,
        encapsulant=None,
        encapsulant_thickness=None,
        absorptance=None,
        length=None,
        width=None,
    ):
        # f = utilities.read_material()
        # f = open(os.path.join(DATA_DIR, "materials.json"))
        # self.materials = json.load(f)

        self.absorptance = absorptance
        self.length = length
        self.width = width

        if backsheet:
            self.setBacksheet(backsheet, backsheet_thickness)

        if encapsulant:
            self.setEncapsulant(encapsulant_thickness)

    def setEncapsulant(self, pvdeg_file: str, key: str, thickness: float, fp: str = None) -> None:
        """
        Set encapsulant diffusivity activation energy, prefactor and solubility activation energy, prefactor.

        Activation Energies will be in kJ/mol

        Parameters:
        -----------
        pvdeg_file: str
            keyword for material json file in `pvdeg/data`. Options:
            >>> "AApermeation", "H2Opermeation", "O2permeation"
        fp: str
            file path to material parameters json with same schema as material parameters json files in `pvdeg/data`. `pvdeg_file` will override `fp` if both are provided.
        key: str
            key corresponding to specific material in the file. In the pvdeg files these have arbitrary names. Inspect the files or use `display_json` or `search_json` to identify the key for desired material.
        thickness: float
            thickness of encapsulant [mm]
        """

        material_dict = utilities.read_material(
            pvdeg_file=pvdeg_file,
            fp=fp,
            key=key,
        )

        self.diffusivity_encap_ea = material_dict["Ead"]
        self.diffusivity_encap_pre = material_dict["Do"]
        self.solubility_encap_ea = material_dict["Eas"]
        self.solubility_encap_pre = material_dict["So"]

        # self.diffusivity_encap_ea = (self.materials)[key]["Ead"]
        # self.diffusivity_encap_pre = (self.materials)[key]["Do"]
        # self.solubility_encap_ea = (self.materials)[key]["Eas"]
        # self.solubility_encap_pre = (self.materials)[key]["So"]
        self.encap_thickness = thickness

    def setBacksheet(self, pvdeg_file: str, key: str, thickness: float, fp: str = None ) -> None:
        """
        Set backsheet permiability activation energy and prefactor.

        Activation Energies will be in kJ/mol

        Parameters:
        -----------
        id: str
            name of material from `PVDegradationTools/data/materials.json`
        thickness: float
            thickness of backsheet [mm]

        Parameters:
        -----------
        pvdeg_file: str
            keyword for material json file in `pvdeg/data`. Options:
            >>> "AApermeation", "H2Opermeation", "O2permeation"
        fp: str
            file path to material parameters json with same schema as material parameters json files in `pvdeg/data`. `pvdeg_file` will override `fp` if both are provided.
        key: str
            key corresponding to specific material in the file. In the pvdeg files these have arbitrary names. Inspect the files or use `display_json` or `search_json` to identify the key for desired material.
        thickness: float
            thickness of backsheet [mm]
        """

        material_dict = utilities.read_material(
            pvdeg_file=pvdeg_file,
            fp=fp,
            key=key,
        )

        self.permiability_back_ea = material_dict["Eap"]
        self.permiability_back_pre = material_dict["Po"]

        # self.permiability_back_ea = (self.materials)[id]["Eap"]
        # self.permiability_back_pre = (self.materials)[id]["Po"]
        self.back_thickness = thickness

    def setDimensions(self, length: float = None, width: float = None) -> None:
        """
        Set dimensions of a rectangular test sample.

        Parameters:
        -----------
        length: float
            dimension of side A [m]
        width: float
            dimension of side B [m]

        Modifies:
        ----------
        self.length
            sets to length
        self.width
            sets to width
        """
        self.length = length
        self.width = width

    def setAbsorptance(self, absorptance: float):
        """
        Set the absorptance of a test sample.

        Parameters:
        -----------
        absorptance: float
            Fraction of light absorbed by the sample. [unitless]

        Modifies:
        ----------
        self.absorptance: float
            sets to absorptance arg
        """
        self.absorptance = absorptance


class Chamber(Sample):
    def __init__(self, fp: str = None, setpoint_names: list[str] = None, **kwargs):
        """
        Create a chamber stress test object.

        This will contain information about the chamber and the sample before running calculations for chamber and sample conditions.

        Parameters:
        -----------
        fp: str
            filepath to CSV of setpoints following the schema defined in the [docs]()
        """

        super().__init__()
        self.setpoint_timeseries(fp, setpoint_names=setpoint_names, **kwargs)

    def setpoint_timeseries(
        self, fp: str, setpoint_names: list[str] = None, **kwargs
    ) -> None:
        """
        Read a setpoints CSV and create a timeseries of setpoint values
        """
        self.setpoints = setpoints_timeseries_from_csv(fp, setpoint_names, **kwargs)

    def plot_setpoints(self) -> None:
        """
        Plot setpoint timeseries values
        """
        self.setpoints.plot(title="Chamber Setpoints")

    def calc_temperatures(
        self, air_temp_0: float, sample_temp_0: float, tau_c: float, tau_s: float
    ) -> None:
        """
        Calculate sample and air temperatures.

        Parameters:
        -----------
        air_temp_0: float
            initial air temperature in chamber [$\degree C$]
        sample_temp_0: float
            initial air temperature in chamber [$\degree C$]
        tau_c: float
            $\tau_C$ thermal equilibration time of the chamber [min]
        tau_s: float
            $\tau_S$ thermal equilibration time of the test sample [min]

        Modifies:
        ---------
        self.air_temperature: pd.Series
            pandas series of air temperatures inside of test chamber [C]
        self.sample_temperature: pd.Series
            pandas series of sample temperatures inside of test chamber [C]
        """
        self.air_temperature = air_temperature(
            self.setpoints, tau_c=tau_c, air_temp_0=air_temp_0
        )

        if (
            "setpoint_irradiance_340" in self.setpoints.columns
            and "setpoint_irradiance_full" not in self.setpoints.columns
        ):
            # gti calculation is very slow because of integration
            print("Calculating GTI...")
            # should this be setpoint irradiance full, it is not a setpoint
            self.setpoints["setpoint_irradiance_full"] = spectral.get_GTI_from_irradiance_340(
                self.setpoints["setpoint_irradiance_340"]
            )  # this may be misleading
            print('Saved in self.setpoints as "setpoints_irradiance_full')

        self.sample_temperature = sample_temperature(
            self.setpoints,
            tau_s=tau_s,
            air_temperature=self.air_temperature,
            sample_temp_0=sample_temp_0,
            surface_area=(self.length * self.width),
            absorptance=self.absorptance,
        )

    def plot_temperatures(self):
        """Plot sample and air temperature over the course of the chamber test"""
        self.air_temperature.plot(label="air temperature")
        self.sample_temperature.plot(label="sample temperature")
        plt.legend()

    def calc_water_vapor_pressure(self):
        """Calculate chamber water vapor pressure"""
        res = humidity.water_vapor_pressure(
            self.air_temperature.to_numpy(dtype=np.float64),
            self.setpoints["setpoint_relative_humidity"].to_numpy(dtype=np.float64),
        )
        self.water_vapor_pressure = pd.Series(
            res, index=self.setpoints.index, name="Water Vapor Pressure"
        )

    def calc_sample_relative_humidity(self):
        """Calculate sample percent relative humidity"""
        res = humidity.rh_at_sample_temperature(
            self.air_temperature.to_numpy(dtype=np.float64),  # C
            self.setpoints["setpoint_relative_humidity"].to_numpy(
                dtype=np.float64
            ),  # %
            self.sample_temperature.to_numpy(dtype=np.float64),  # C
        )
        self.sample_relative_humidity = pd.Series(
            res, index=self.setpoints.index, name="Sample Relative Humidity"
        )

    def calc_equilibrium_ecapsulant_water(self):
        """Calculate the equilibrium state of the water in the encapsulant"""
        res = humidity.equilibrium_eva_water(
            self.sample_temperature.to_numpy(dtype=np.float64),  # C
            self.sample_relative_humidity.to_numpy(np.float64),  # %
            utilities.kj_mol_to_ev(self.solubility_encap_ea),  # kJ/mol -> eV
            self.solubility_encap_pre,  # cm^2/s
        )
        self.equilibrium_encapsulant_water = pd.Series(
            res, index=self.setpoints.index, name="Equilibrium Encapsulant Water"
        )

    def calc_back_encapsulant_moisture(self, n_steps: int = 20):
        """Calculate the moisture on the backside of the encapsulant"""

        # kJ/mol -> eV
        permiability = utilities.kj_mol_to_ev(self.permiability_back_ea)

        res, _ = humidity.moisture_eva_back(
            eva_moisture_0=self.equilibrium_encapsulant_water.iloc[0],
            sample_temp=self.sample_temperature,
            rh_at_sample_temp=self.sample_relative_humidity,
            equilibrium_eva_water=self.equilibrium_encapsulant_water,
            # pet_permiability=utilities.kj_mol_to_ev(
            #     self.permiability_back_ea
            # ),  # kJ/mol -> eV
            pet_permiability=permiability,
            pet_prefactor=self.permiability_back_pre,
            thickness_eva=self.encap_thickness,  # mm
            thickness_pet=self.back_thickness,  # mm
            n_steps=n_steps,
        )

        self.back_encapsulant_moisture = pd.Series(
            # res, index=self.setpoints.index, name="Back Encapsulant Moisture"
            res, 
            index=self.setpoints.index, 
            name="Back Encapsulant Moisture"
        )

    def calc_relative_humidity_internal_on_back_of_cells(self):
        """Calculate the relative humidity inside the module on the backside of the cells"""
        res = humidity.rh_internal_cell_backside(
            back_eva_moisture=self.back_encapsulant_moisture.to_numpy(dtype=np.float64),
            equilibrium_eva_water=self.equilibrium_encapsulant_water.to_numpy(
                dtype=np.float64
            ),
            rh_at_sample_temp=self.sample_relative_humidity.to_numpy(dtype=np.float64),
        )

        self.relative_humidity_internal_on_back_of_cells = pd.Series(
            res, 
            self.setpoints.index, 
            name="Relative Humidity Internal Cells Backside"
        )

    def calc_dew_point(self):
        """Calculate the chamber dew point"""
        res = humidity.chamber_dew_point_from_vapor_pressure(
            self.water_vapor_pressure.to_numpy(dtype=np.float64)
        )
        self.dew_point = pd.Series(res, self.setpoints.index, name="Dew Point")

    def chamber_conditions(self, tau_c: float, air_temp_0: float) -> pd.DataFrame:
        """
        Calculate the chamber conditions

        Parameters:
        -----------
        tau_c: float
            $\tau_C$ thermal equilibration time of the chamber [min]
        air_temp_0: float
            initial air temperature in chamber [$\degree C$]

        Returns:
        --------
        chamber_and_set_df: pd.DataFrame
            pandas dataframe containing all chamber setpoints, chamber air temperature,
            water vapor pressure and dew point conditions for each timestep.
        """

        self.air_temperature = air_temperature(
            self.setpoints, tau_c=tau_c, air_temp_0=air_temp_0
        )
        self.calc_water_vapor_pressure()
        self.calc_dew_point()

        chamber_df = pd.DataFrame(
            [self.air_temperature, self.water_vapor_pressure, self.dew_point]
        ).T
        chamber_and_set_df = pd.concat([self.setpoints, chamber_df], axis=1)
        return chamber_and_set_df

    def sample_conditions(
        self, tau_s: float, sample_temp_0: float, n_steps: int = 20
    ) -> pd.DataFrame:
        """
        Calculate the sample conditions

        Parameters:
        -----------
        tau_s: float
            $\tau_S$ thermal equilibration time of the test sample [min]
        sample_temp_0: float
            initial air temperature in chamber [$\degree C$]
        n_steps: int
            number of stubsteps to calculate for numerical stability.
            4-5 works for most cases but quick changes error can accumulate
            quickly so 20 is a good value for numerical safety.
            *Used to calculate backside encapsulant moisture*

        Returns:
        --------
        sample_df: pd.DataFrame
            pandas dataframe containing
            [sample temperature, sample relative humidity, equilibrium encapsulant water, back encapsulant moisture, relative humidity internal on back of cells]
        """
        self.sample_temperature = sample_temperature(
            self.setpoints,
            air_temperature=self.air_temperature,
            tau_s=tau_s,
            sample_temp_0=sample_temp_0,
            surface_area=(self.length * self.width),
            absorptance=self.absorptance,
        )
        self.calc_sample_relative_humidity()
        self.calc_equilibrium_ecapsulant_water()
        self.calc_back_encapsulant_moisture(n_steps=n_steps)
        self.calc_relative_humidity_internal_on_back_of_cells()

        sample_df = pd.DataFrame(
            [
                self.sample_temperature,
                self.sample_relative_humidity,
                self.equilibrium_encapsulant_water,
                self.back_encapsulant_moisture,
                self.relative_humidity_internal_on_back_of_cells,
            ]
        ).T
        return sample_df

    def gti_from_irradiance_340(self) -> pd.Series:
        """
        Calculate full spectrum GTI from irradiance at 340 nm

        Returns:
        --------
        gti: pd.Series
            full spectrum irradiance using ASTM G173-03 AM1.5 spectrum.
        """
        self.setpoints["setpoint_irradiance_full"] = spectral.get_GTI_from_irradiance_340(
            self.setpoints["setpoint_irradiance_340"]
        )

        return self.setpoints["setpoint_irradiance_full"]

    def _ipython_display_(self):
        """
        Display the setpoints of the chamber instance.
        """

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 4))
        self.setpoints.plot(ax=ax)
        ax.set_title("Setpoints Plot (Units Not Demonstrated on Y Axis)")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        plt.tight_layout()

        # Save the plot to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close(fig)
        img_buffer.seek(0)

        # Encode the image as base64
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

        html_content = f"""
        <div style="border:1px solid #ddd; border-radius: 5px; padding: 3px; margin-top: 5px;">
            <h2>Chamber Simulation</h2>

            <div id="setpoints_dataframe" onclick="toggleVisibility('content_setpoints_dataframe')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_content_setpoints_dataframe" style="color: #b676c2;">►</span>
                     Setpoints Dataframe
                </h4>
            </div>
            <div id="content_setpoints_dataframe" style="display:none; margin-left: 20px; padding: 5px;">
                {self.setpoints._repr_html_()}
            </div>

            <div id="setpoints_plot" onclick="toggleVisibility('content_setpoints_plot')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_content_setpoints_plot" style="color: #b676c2;">►</span>
                     Setpoints Plot
                </h4>
            </div>
            <div id="content_setpoints_plot" style="display:none; margin-left: 20px; padding: 5px;">
                <h3>Setpoints Plot</h3>
                <img src="data:image/png;base64,{img_base64}" alt="Setpoints Plot" style="max-width:100%; height:auto;">
            </div>

        </div>
        <script>
            function toggleVisibility(id) {{
                var content = document.getElementById(id);
                var arrow = document.getElementById('arrow_' + id);
                if (content.style.display === 'none') {{
                    content.style.display = 'block';
                    arrow.innerHTML = '▼';
                }} else {{
                    content.style.display = 'none';
                    arrow.innerHTML = '►';
                }}
            }}
        </script>
        """
        display(HTML(html_content))
