"""
Collection of classes and functions to calculate different temperatures.
"""

import pvlib
import pvdeg

import numpy as np
from numba import njit
from typing import Union

def module(
    weather_df,
    meta,
    poa=None,
    temp_model="sapm",
    conf="open_rack_glass_polymer",
    wind_factor=0.33,
):
    """
    Calculate module surface temperature using pvlib.

    Parameters
    ----------
    weather_df : (pd.dataframe)
        Data Frame with minimum requirements of 'temp_air' and 'wind_speed'
    poa : pandas.DataFrame
         Contains keys/columns 'poa_global', 'poa_direct', 'poa_diffuse',
         'poa_sky_diffuse', 'poa_ground_diffuse'.
    temp_model : str, optional
        The temperature model to use, Sandia Array Performance Model 'sapm'
        from pvlib by default.
    conf : str, optional
        The configuration of the PV module architecture and mounting
        configuration.
        Options:
            'sapm': 'open_rack_glass_polymer' (default),
            'open_rack_glass_glass', 'close_mount_glass_glass',
            'insulated_back_glass_polymer'

    Returns
    -------
    module_temperature : pandas.DataFrame
        The module temperature in degrees Celsius at each time step.
    """
    parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[temp_model][conf]

    if poa is None:
        poa = pvdeg.spectral.poa_irradiance(weather_df, meta)
    if "wind_height" not in meta.keys():
        wind_speed_factor = 1
    else:
        if temp_model == "sapm":
            wind_speed_factor = (10 / float(meta["wind_height"])) ** wind_factor
        elif temp_model == "pvsyst":
            wind_speed_factor = (2 / float(meta["wind_height"])) ** wind_factor
        elif temp_model == "faiman":
            wind_speed_factor = (2 / float(meta["wind_height"])) ** wind_factor
        elif temp_model == "faiman_rad":
            wind_speed_factor = (2 / float(meta["wind_height"])) ** wind_factor
        elif temp_model == "fuentes":
            wind_speed_factor = (5 / float(meta["wind_height"])) ** wind_factor
        elif temp_model == "ross":
            wind_speed_factor = (
                10 / float(meta["wind_height"])
            ) ** wind_factor  # I had to guess what this one was
        elif temp_model == "noct_sam":
            if meta["wind_height"] > 3:
                wind_speed_factor = 2
            else:
                wind_speed_factor = (
                    1  # The wind speed height is managed weirdly for this one.
                )
        elif temp_model == "prilliman":
            wind_speed_factor = (
                1  # this model will take the wind speed height in and do an adjustment.
            )
        elif temp_model == "generic_linear":
            wind_speed_factor = (10 / float(meta["wind_height"])) ** wind_factor
        elif temp_model == "GenericLinearModel":
            wind_speed_factor = (2 / float(meta["wind_height"])) ** wind_factor
            # this one does a linear conversion from the other models, faiman, pvsyst, noct_sam, sapm_module and generic_linear.
            # An appropriate facter will need to be figured out.
        else:
            wind_speed_factor = 1  # this is just hear for completeness.
    # TODO put in code for the other models, PVSYS, Faiman,

    if temp_model == "sapm":
        module_temperature = pvlib.temperature.sapm_module(
            poa_global=poa["poa_global"],
            temp_air=weather_df["temp_air"],
            wind_speed=weather_df["wind_speed"] * wind_speed_factor,
            a=parameters["a"],
            b=parameters["b"],
        )
    else:
        # TODO: add options for temperature model
        print("There are other models but they haven't been implemented yet!")
    return module_temperature


def cell(
    weather_df,
    meta,
    poa=None,
    temp_model="sapm",
    conf="open_rack_glass_polymer",
    wind_factor=0.33,
):
    """
    Calculate the PV cell temperature using PVLIB
    Currently this only supports the SAPM temperature model.

    Parameters:
    -----------
    weather_df : (pd.dataframe)
        Data Frame with minimum requirements of 'temp_air' and 'wind_speed'
    meta : (dict)
        Weather meta-data dictionary (location info)
    poa : (dataframe or series, optional)
        Dataframe or series with minimum requirement of 'poa_global'
    temp_model : (str, optional)
        Specify which temperature model from pvlib to use. Current options:
        'sapm'
    conf : (str)
        The configuration of the PV module architecture and mounting
        configuration.
        Options: 'open_rack_glass_polymer' (default), 'open_rack_glass_glass',
                 'close_mount_glass_glass', 'insulated_back_glass_polymer'
    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement heights
        between weather database (e.g. NSRDB) and the tempeature model (e.g. SAPM)
        The NSRDB provides calculations at 2 m (i.e module height) but SAPM uses a 10 m height.
        It is recommended that a power-law relationship between height and wind speed of 0.33
        be used*. This results in a wind speed that is 1.7 times higher. It is acknowledged that
        this can vary significantly.

    R. Rabbani, M. Zeeshan, "Exploring the suitability of MERRA-2 reanalysis data for wind energy
        estimation, analysis of wind characteristics and energy potential assessment for selected
        sites in Pakistan", Renewable Energy 154 (2020) 1240-1251.

    Return:
    -------
    temp_cell : pandas.DataFrame
        This is the temperature of the cell in a module at every time step.[°C]
    """

    if "wind_height" not in meta.keys():
        wind_speed_factor = 1
    else:
        if temp_model == "sapm":
            wind_speed_factor = (10 / float(meta["wind_height"])) ** wind_factor
        elif temp_model == "pvsyst":
            wind_speed_factor = (2 / float(meta["wind_height"])) ** wind_factor
        elif temp_model == "faiman":
            wind_speed_factor = (2 / float(meta["wind_height"])) ** wind_factor
        elif temp_model == "faiman_rad":
            wind_speed_factor = (2 / float(meta["wind_height"])) ** wind_factor
        elif temp_model == "fuentes":
            wind_speed_factor = (5 / float(meta["wind_height"])) ** wind_factor
        elif temp_model == "ross":
            wind_speed_factor = (
                10 / float(meta["wind_height"])
            ) ** wind_factor  # I had to guess what this one was
        elif temp_model == "notc_sam":
            if float(meta["wind_height"]) > 3:
                wind_speed_factor = 2
            else:
                wind_speed_factor = (
                    1  # The wind speed height is managed weirdly for this one.
                )
        elif temp_model == "prilliman":
            wind_speed_factor = (
                1  # this model will take the wind speed height in and do an adjustment.
            )
        elif temp_model == "generic_linear":
            wind_speed_factor = (10 / float(meta["wind_height"])) ** wind_factor
        elif temp_model == "GenericLinearModel":
            wind_speed_factor = (2 / float(meta["wind_height"])) ** wind_factor
            # this one does a linear conversion from the other models, faiman, pvsyst, noct_sam, sapm_module and generic_linear.
            # An appropriate facter will need to be figured out.
        else:
            wind_speed_factor = 1  # this is just here for completeness.
    parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[temp_model][conf]

    if poa is None:
        poa = pvdeg.spectral.poa_irradiance(weather_df, meta)

    if temp_model == "sapm":
        temp_cell = pvlib.temperature.sapm_cell(
            poa_global=poa["poa_global"],
            temp_air=weather_df["temp_air"],
            wind_speed=weather_df["wind_speed"] * wind_speed_factor,
            **parameters,
        )
    else:
        # TODO: add options for temperature model
        print("There are other models but they haven't been implemented yet!")

    return temp_cell

# @njit
def chamber_sample_temperature(
    irradiance_340: float,
    temp_set: float,
    previous_sample_temp: float,
    delta_t: float,
    tau: float
    )-> float:
    """
    Finite difference method for chamber sample temperature.

    .. math::
        
        T_2 = T_0 + (T_1 - T_0)e^\frac{- \Delta t}{\tau}

    Parameters:
    -----------
    irradiance_340: float
        UV irradiance [W/m^2/nm at 340 nm]
    temp_set: float
        chamber temperature setpoint
    previous_sample_temp: float
        temperature of chamber sample during previous timestep [C]
    delta_t: float
        length of timestep (end time - start time) [min]
    tau: float
        Characteristic thermal equilibration time [min]
    """

    if irradiance_340 == 0:
        sample_temp = (
            temp_set + (previous_sample_temp-temp_set) 
            * np.exp(-(delta_t) / tau) 
        )
    
    else:
        if irradiance_340 == 0.4: # what is specical about 0.4
            sample_temp = (
                temp_set + 16 + (previous_sample_temp-temp_set-16)
                * np.exp(-(delta_t) / tau) 
            )

        else:
            sample_temp = (
                temp_set + 40 + (previous_sample_temp-temp_set-40) 
                * np.exp(-(delta_t) / tau) 
            )

    return sample_temp

@njit
def fdm_temperature(t_current: float, t_set: float, delta_time: float, tau: float):
    """
    Calculate next timestep of temperature using finite difference method without utilizing irradiance.

    Parameters:
    -----------
    t_current: float
        current temperature [C]
    t_set: float
        temperature we are approaching [C]
    delta_time: float
        length of timestep, units should match time unit of tau [time]
    tau: float
        thermal equilibration time, units should match time unit of delta_time [time]

    Returns:
    --------
    t_next: float
        temperature at next timestep [C]
    """

    t_next = t_current + (t_set - t_current) * (1 - np.exp( (-delta_time) / (tau) ))
    return t_next

# @njit
def fdm_temperature_irradiance(
    t_current: float, 
    t_set: float, 
    irradiance: float,
    delta_time: float, 
    tau: float, 
    surface_area: float,
    absorptance: float,
) -> float:

    """
    Calculate next timestep of temperature using finite difference method utilizing irradiance.

    Parameters:
    -----------
    t_current: float
        current temperature [C]
    t_set: float
        temperature we are approaching [C]
    irradiance: float
        full spectrum irradiance at current time [w / m^2]
    delta_time: float
        length of timestep, units should match time unit of tau [time]
    tau: float
        thermal equilibration time, units should match time unit of delta_time [time]
    surface_area: float
        surface area of sample [m^2]
    absorptance: float
        Fraction of light absorbed by the sample. [unitless]
    Returns:
    --------
    t_next: float
        temperature at next timestep [C]
    """
    
    # new term
    # we wan to add the increased irradiance caused by the temperature
    temp_increase = (delta_time / tau) * irradiance * surface_area * absorptance

    t_next = t_current + (t_set - t_current) * (1 - np.exp( (-delta_time) / (tau) )) + temp_increase
    return t_next



