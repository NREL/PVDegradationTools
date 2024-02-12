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
