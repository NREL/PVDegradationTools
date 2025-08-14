"""Collection of classes and functions to calculate different temperatures."""

import pvlib

# import pvdeg

from pvdeg import (
    spectral,
    decorators,
)
import pandas as pd
from typing import Union
import inspect


def map_model(temp_model: str, cell_or_mod: str) -> callable:
    """Map string to pvlib function.

    References
    ----------
    https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html
    """
    # sapm_cell_from_module?
    # prillman diverges from others, used for smoothing/interp

    # double check that models are in correct maps
    module = {  # only module
        "sapm": pvlib.temperature.sapm_module,
        "sapm_mod": pvlib.temperature.sapm_module,
    }

    cell = {  # only cell
        "sapm": pvlib.temperature.sapm_cell,
        "sapm_cell": pvlib.temperature.sapm_cell,
        "pvsyst": pvlib.temperature.pvsyst_cell,
        "ross": pvlib.temperature.ross,
        "noct_sam": pvlib.temperature.noct_sam,
        "generic_linear": pvlib.temperature.generic_linear,
    }

    agnostic = {  # module or cell
        "faiman": pvlib.temperature.faiman,
        "faiman_rad": pvlib.temperature.faiman_rad,
        "fuentes": pvlib.temperature.fuentes,
    }

    super_map = {"module": module, "cell": cell}

    if cell_or_mod:
        agnostic.update(super_map[cell_or_mod])  # bad naming
    else:
        agnostic.update(module)  # if none then we use all
        agnostic.update(cell)

    return agnostic[temp_model]


def _wind_speed_factor(temp_model: str, meta: dict, wind_factor: float):
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
        # this one does a linear conversion from the other models, faiman, pvsyst,
        # noct_sam, sapm_module and generic_linear.
        # An appropriate facter will need to be figured out.
    else:
        wind_speed_factor = 1  # this is just hear for completeness.

    return wind_speed_factor


@decorators.geospatial_quick_shape("timeseries", ["module_temperature"])
def module(
    weather_df,
    meta,
    poa=None,
    temp_model="sapm",
    conf="open_rack_glass_polymer",
    wind_factor=0.33,
):
    """Calculate module surface temperature using pvlib.

    Parameters
    ----------
    weather_df : pd.dataframe
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
    module_temperature : pandas.Series
        The module temperature in degrees Celsius at each time step.
    """
    parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[temp_model][conf]

    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta)
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
            ) ** wind_factor  # guessed the temperature model height on this one, Kempe
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
            # this one does a linear conversion from the other models, faiman, pvsyst,
            # noct_sam, sapm_module and generic_linear.
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


@decorators.geospatial_quick_shape("timeseries", ["cell_temperature"])
def cell(
    weather_df: pd.DataFrame,
    meta: dict,
    poa: Union[pd.DataFrame, pd.Series] = None,
    temp_model: str = "sapm",
    conf: str = "open_rack_glass_polymer",
    wind_factor: float = 0.33,
) -> pd.DataFrame:
    """Calculate the PV cell temperature using pvlib-python.

    Currently this only supports the SAPM temperature model.

    Parameters
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
        Wind speed correction exponent to account for different wind speed measurement
        heights between weather database (e.g. NSRDB) and the tempeature model
        (e.g. SAPM). The NSRDB provides calculations at 2 m (i.e module height) but SAPM
        uses a 10m height.
        It is recommended that a power-law relationship between height and wind speed
        of 0.33 be used*. This results in a wind speed that is 1.7 times higher. It is
        acknowledged that this can vary significantly.

    R. Rabbani, M. Zeeshan, "Exploring the suitability of MERRA-2 reanalysis data for
    wind energy estimation, analysis of wind characteristics and energy potential
    assessment for selected sites in Pakistan", Renewable Energy 154 (2020) 1240-1251.

    Return:
    -------
    temp_cell : pandas.Series
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
            ) ** wind_factor  # I had to guess what the wind height for this temperature
            # model was on this one, Kempe.
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
            # this one does a linear conversion from the other models, faiman, pvsyst,
            # noct_sam, sapm_module and generic_linear.
            # An appropriate facter will need to be figured out.
        else:
            wind_speed_factor = 1  # this is just here for completeness.
    parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[temp_model][conf]

    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta)

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


# test not providing poa
# what if we dont need the cell or mod param, only matters for sapm
# to add more temperature model options just add them to the model_map function with the
# value as a reference to your function
# genaric linear is a little weird, we need to calculate the values outside of the
# function using pvlib.temp.genearilinearmodel and converting, can we take a reference
# to the model or args for the model instead
def temperature(
    weather_df,
    meta,
    poa=None,
    temp_model="sapm",
    cell_or_mod=None,
    conf="open_rack_glass_polymer",
    wind_factor=0.33,
    irradiance_kwarg={},
    model_kwarg={},
):
    """
    Calculate the PV cell or module temperature using PVLIB.

    Current supports the following temperature models:

    Parameters
    -----------
    cell_or_mod : (str)
        choose to calculate the cell or module temperature. Use
        ``cell_or_mod == 'mod' or 'module'`` for module temp calculation.
        ``cell_or_mod == 'cell'`` for cell temp calculation.
    weather_df : (pd.dataframe)
        Data Frame with minimum requirements of 'temp_air' and 'wind_speed'
    meta : (dict)
        Weather meta-data dictionary (location info)
    poa : (dataframe or series, optional)
        Dataframe or series with minimum requirement of 'poa_global'. Will be calculated
        rom weather_df, meta if not provided
    temp_model : (str, optional)
        Specify which temperature model from pvlib to use. Current options:

        ``sapm_cell``,``sapm_module``,``pvsyst_cell``,``faiman``,``faiman_rad``,
        ``ross``,``noct_sam``, ``fuentes``, ``generic_linear``

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
        uses a 10m height.
        It is recommended that a power-law relationship between height and wind speed
        of 0.33 be used*. This results in a wind speed that is 1.7 times higher. It is
        acknowledged that this can vary significantly.
    irradiance_kwarg : (dict, optional)
        keyword argument dictionary used for the poa irradiance caluation.
        options: ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``. See
        ``pvdeg.spectral.poa_irradiance``.
    model_kwarg : (dict, optional)
        keyword argument dictionary used for the pvlib temperature model calculation.
        See https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html  # noqa
        for more.

    Return
    -------
    temp_cell : pandas.DataFrame
        This is the temperature of the cell in a module at every time step.[°C]

    References
    -----------
    R. Rabbani, M. Zeeshan, "Exploring the suitability of MERRA-2 reanalysis data for
    wind energy estimation, analysis of wind characteristics and energy potential
    assessment for selected sites in Pakistan", Renewable Energy 154 (2020) 1240-1251.
    """
    cell_or_mod = "module" if cell_or_mod == "mod" else cell_or_mod  # mod->module

    if "wind_height" not in meta.keys():
        wind_speed_factor = 1
    else:
        wind_speed_factor = _wind_speed_factor(temp_model, meta, wind_factor)

    if temp_model in ["sapm", "pvsyst"]:
        parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[temp_model][conf]

        if cell_or_mod != "cell" and temp_model == "sapm":
            # strip the 'deltaT' for calculations that will not need it
            parameters = {k: v for k, v in parameters.items() if k != "deltaT"}

    if poa is None:
        poa = spectral.poa_irradiance(weather_df, meta, **irradiance_kwarg)

    # irrelevant key,value pair will be ignored (NO ERROR)
    weather_args = {
        "poa_global": poa["poa_global"],
        "temp_air": weather_df["temp_air"],
        "wind_speed": weather_df["wind_speed"] * wind_speed_factor,
    }  # but this will ovewrite the model default always, so will have to provide
    # default wind speed in the kwargs

    # only apply nessecary values to the model,
    func = map_model(temp_model, cell_or_mod)
    sig = inspect.signature(func)

    # unpack signature into list for merging all dictionaries
    model_args = {
        k: (v.default if v.default is not inspect.Parameter.empty else None)
        for k, v in sig.parameters.items()
    }

    # if key is present update the value in the function signature
    for key in model_args:
        # we only want to update the values with matching keys
        if key in weather_args:
            model_args[key] = weather_args[key]

    try:
        model_args.update(parameters)
    except NameError:
        pass  # hits when not sapm or pvsyst

    # add optional kwargs, overwrites copies
    model_args.update(**model_kwarg)

    temperature = func(**model_args)

    return temperature
