"""
Collection of classes and functions to calculate different temperatures.
"""

import pvlib
import pvdeg


def module(
    weather_df,
    meta,
    poa=None,
    temp_model="sapm",
    conf="open_rack_glass_polymer",
    wind_speed_factor=1,
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
        The temperature model to use, Sandia Array Performance Model 'sapm' from pvlib by default.
    conf : str, optional
        The configuration of the PV module architecture and mounting configuration.
        Options:
            'sapm': 'open_rack_glass_polymer' (default), 'open_rack_glass_glass',
                    'close_mount_glass_glass', 'insulated_back_glass_polymer'

    Returns
    -------
    module_temperature : pandas.DataFrame
        The module temperature in degrees Celsius at each time step.
    """
    parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[temp_model][conf]

    if poa is None:
        poa = pvdeg.spectral.poa_irradiance(weather_df, meta)

    module_temperature = pvlib.temperature.sapm_module(
        poa_global=poa["poa_global"],
        temp_air=weather_df["temp_air"],
        wind_speed=weather_df["wind_speed"] * wind_speed_factor,
        a=parameters["a"],
        b=parameters["b"],
    )

    return module_temperature


def cell(weather_df, meta, poa=None, temp_model="sapm", conf="open_rack_glass_polymer"):
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
        DO NOT CHANGE UNTIL UPDATED
    conf : (str)
        The configuration of the PV module architecture and mounting configuration.
        Options: 'open_rack_glass_polymer' (default), 'open_rack_glass_glass',
                 'close_mount_glass_glass', 'insulated_back_glass_polymer'
    """

    parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[temp_model][conf]

    if poa is None:
        poa = pvdeg.spectral.poa_irradiance(weather_df, meta)

    temp_cell = pvlib.temperature.sapm_cell(
        poa_global=poa["poa_global"],
        temp_air=weather_df["temp_air"],
        wind_speed=weather_df["wind_speed"],
        **parameters
    )
    return temp_cell
