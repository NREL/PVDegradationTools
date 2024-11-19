"""
Collection of classes and functions to obtain spectral parameters.
"""

import pvlib
import pandas as pd
import inspect
import warnings
from pvdeg.decorators import geospatial_quick_shape


@geospatial_quick_shape(
    1,
    [
        "apparent_zenith",
        "zenith",
        "apparent_elevation",
        "elevation",
        "azimuth",
        "equation_of_time",
    ],
)
def solar_position(weather_df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Calculate solar position using pvlib based on weather data from the
    National Solar Radiation Database (NSRDB) for a given location (gid).

    Parameters
    ----------
    weather_df : pandas.DataFrame
        Weather data for given location.
    meta : dict
        Meta data of location.

    Returns
    -------
    solar_position : pandas.DataFrame
        Solar position like zenith and azimuth.
    """

    # location = pvlib.location.Location(
    #     latitude=meta['latitude'],
    #     longitude=meta['longitude'],
    #     altitude=meta['elevation'])

    # TODO: check if timeshift is necessary
    # times = weather_df.index
    # solar_position = location.get_solarposition(times)
    solar_position = pvlib.solarposition.get_solarposition(
        time=weather_df.index,
        latitude=meta["latitude"],
        longitude=meta["longitude"],
        altitude=meta["altitude"],
    )

    return solar_position


@geospatial_quick_shape(
    1,
    [
        "poa_global",
        "poa_direct",
        "poa_diffuse",
        "poa_sky_diffuse",
        "poa_ground_diffuse",
    ],
)
def poa_irradiance(
    weather_df: pd.DataFrame,
    meta: dict,
    module_mount="fixed",
    sol_position=None,
    dni_extra=None,
    airmass=None,
    albedo=0.25,
    surface_type=None,
    sky_model="isotropic",
    model_perez="allsitescomposite1990",
    **kwargs_mount,
) -> pd.DataFrame:
    """
    Calculate plane-of-array (POA) irradiance using `pvlib.irradiance.get_total_irradiance` for different module mounts
     as fixed tilt systems or tracked systems.

    Parameters
    ----------
    weather_df : pd.DataFrame
        The file path to the NSRDB file.
    meta : dict
        The geographical location ID in the NSRDB file.
    module_mount: string
        Module mounting configuration. Can either be `fixed` for fixed tilt systems or
        `single_axis` for single-axis tracker systems.
    sol_position : pd.DataFrame, optional
        pvlib.solarposition.get_solarposition Dataframe. If none is given, it will be calculated.
    dni_extra : pd.Series, optional
        Extra-terrestrial direct normal irradiance. If None, it will be calculated.
    airmass : pd.Series, optional
        Airmass values. If None, it will be calculated.
    albedo : float, optional
        Ground reflectance. Default is 0.25.
    surface_type : str, optional
        Type of surface for albedo calculation. If None, a default value will be used.
    sky_model : str, optional
        Sky diffuse model to use. Default is `isotropic`.
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    model_perez : str, optional
        Perez model to use for diffuse irradiance. Default is `allsitescomposite1990`.
        Only used when sky_model is 'perez'.
    **kwargs_mount : dict
        Additional keyword arguments for the POA model based on mounting configuration. See
        `poa_irradiance_fixed` or `poa_irradiance_tracker` for details.

        - For `module_mount='fixed'`:
            - surface_tilt : float
                Tilt angle of the modules (degrees).
            - surface_azimuth : float
                Azimuth angle of the modules (degrees).

        - For `module_mount='single_axis'`:
            - axis_tilt : float
                Tilt angle of the tracker axis (degrees).
            - axis_azimuth : float
                Azimuth angle of the tracker axis (degrees).
            - max_angle : float
                Maximum rotation angle of the tracker (degrees).
            - backtrack : bool
                Whether to enable backtracking for single-axis trackers.
            - gcr : float
                Ground coverage ratio of the tracker system.


    Returns
    -------
    poa : pandas.DataFrame
         Contains keys/columns 'poa_global', 'poa_direct', 'poa_diffuse',
         'poa_sky_diffuse', 'poa_ground_diffuse'. [W/m2]

    Notes
    -----
    This function uses pvlib to calculate the plane-of-array irradiance based on the provided weather data and
    mounting configuration. See the pvlib documentation for further information on the parameters.
    """

    # Allow legacy keys for tilt and azimuth
    if "tilt" in kwargs_mount:
        kwargs_mount["surface_tilt"] = kwargs_mount.pop("tilt")
    if "azimuth" in kwargs_mount:
        kwargs_mount["surface_azimuth"] = kwargs_mount.pop("azimuth")

    def check_keys_in_list(dictionary, key_list):
        extra_keys = [key for key in dictionary.keys() if key not in key_list]
        for key in extra_keys:
            dictionary.pop(key)
            warnings.warn(
                f"Key '{extra_keys}' cannot be used for the selected `module_mount` and are ignored."
            )

    if sol_position is None:
        sol_position = solar_position(weather_df, meta)

    if module_mount == "fixed":
        arg_names = ["surface_tilt", "surface_azimuth"]
        check_keys_in_list(kwargs_mount, arg_names)

        poa = poa_irradiance_fixed(
            weather_df=weather_df,
            meta=meta,
            sol_position=sol_position,
            dni_extra=dni_extra,
            airmass=airmass,
            albedo=albedo,
            surface_type=surface_type,
            model=sky_model,
            model_perez=model_perez,
            **kwargs_mount,
        )
    elif module_mount == "single_axis":
        args = inspect.signature(pvlib.tracking.singleaxis).parameters
        arg_names = [param.name for param in args.values()]
        check_keys_in_list(kwargs_mount, arg_names)

        poa = poa_irradiance_tracker(
            weather_df=weather_df,
            meta=meta,
            sol_position=sol_position,
            dni_extra=dni_extra,
            airmass=airmass,
            albedo=albedo,
            surface_type=surface_type,
            model=sky_model,
            model_perez=model_perez,
            **kwargs_mount,
        )
    else:
        raise NotImplementedError(
            f"The input module_mount '{module_mount}' is not implemented"
        )

    return poa


@geospatial_quick_shape(
    1,
    [
        "poa_global",
        "poa_direct",
        "poa_diffuse",
        "poa_sky_diffuse",
        "poa_ground_diffuse",
    ],
)
def poa_irradiance_fixed(
    weather_df: pd.DataFrame,
    meta: dict,
    sol_position=None,
    surface_tilt=None,
    surface_azimuth=None,
    dni_extra=None,
    airmass=None,
    albedo=0.25,
    surface_type=None,
    model="isotropic",
    model_perez="allsitescomposite1990",
) -> pd.DataFrame:
    """
    Calculate plane-of-array (POA) irradiance using `pvlib.irradiance.get_total_irradiance` for a fixed tilt system.

    Parameters
    ----------
    weather_df : pd.DataFrame
        The file path to the NSRDB file.
    meta : dict
        The geographical location ID in the NSRDB file.
    sol_position : pd.DataFrame, optional
        pvlib.solarposition.get_solarposition Dataframe. If none is given, it will be calculated.
    dni_extra : pd.Series, optional
        Extra-terrestrial direct normal irradiance. If None, it will be calculated.
    airmass : pd.Series, optional
        Airmass values. If None, it will be calculated.
    albedo : float, optional
        Ground reflectance. Default is 0.25.
    surface_type : str, optional
        Type of surface for albedo calculation. If None, a default value will be used.
    sky_model : str, optional
        Sky diffuse model to use. Default is `isotropic`.
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    model_perez : str, optional
        Perez model to use for diffuse irradiance. Default is `allsitescomposite1990`.
        Only used when sky_model is 'perez'.
    surface_tilt : float, optional
        Tilt angle of the modules (degrees).
    surface_azimuth : float, optional
        Azimuth angle of the modules (degrees).

    Returns
    -------
    poa : pd.DataFrame
        DataFrame containing the calculated plane-of-array irradiance components with columns:
        'poa_global', 'poa_direct', 'poa_diffuse', 'poa_sky_diffuse', 'poa_ground_diffuse'. [W/m2]


    Notes
    -----
    This function uses pvlib to calculate the plane-of-array irradiance based on the provided weather data and
    mounting configuration for fixed tilt systems. See the pvlib documentation for further information on the parameters.
    """

    if surface_tilt is None:
        try:
            surface_tilt = float(meta["tilt"])
        except:
            surface_tilt = float(abs(meta["latitude"]))
            print(
                f"The array surface_tilt angle was not provided, therefore the latitude tilt of {surface_tilt:.1f} was used."
            )

    if surface_azimuth is None:  # Sets the default orientation to equator facing.
        try:
            surface_azimuth = float(meta["azimuth"])
        except:
            if float(meta["latitude"]) < 0:
                surface_azimuth = 0
            else:
                surface_azimuth = 180
                print(
                    f"The array azimuth was not provided, therefore an azimuth of {surface_azimuth:.1f} was used."
                )

    if sol_position is None:
        sol_position = solar_position(weather_df, meta)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=weather_df["dni"],
        ghi=weather_df["ghi"],
        dhi=weather_df["dhi"],
        solar_zenith=sol_position["apparent_zenith"],
        solar_azimuth=sol_position["azimuth"],
        dni_extra=dni_extra,
        airmass=airmass,
        albedo=albedo,
        surface_type=surface_type,
        model=model,
        model_perez=model_perez,
    )

    return poa


@geospatial_quick_shape(
    1,
    [
        "poa_global",
        "poa_direct",
        "poa_diffuse",
        "poa_sky_diffuse",
        "poa_ground_diffuse",
    ],
)
def poa_irradiance_tracker(
    weather_df: pd.DataFrame,
    meta: dict,
    sol_position=None,
    axis_tilt=0,
    axis_azimuth=None,
    max_angle=90,
    backtrack=True,
    gcr=0.2857142857142857,
    cross_axis_tilt=0,
    dni_extra=None,
    airmass=None,
    albedo=0.25,
    surface_type=None,
    model="isotropic",
    model_perez="allsitescomposite1990",
) -> pd.DataFrame:
    """
    Calculate plane-of-array (POA) irradiance using pvlib based on weather data from the
    National Solar Radiation Database (NSRDB) for a given location (gid).

    Parameters
    ----------
    weather_df : pd.DataFrame
        The file path to the NSRDB file.
    meta : dict
        The geographical location ID in the NSRDB file.
    sol_position : pd.DataFrame, optional
        pvlib.solarposition.get_solarposition Dataframe. If none is given, it will be calculated.
    dni_extra : pd.Series, optional
        Extra-terrestrial direct normal irradiance. If None, it will be calculated.
    airmass : pd.Series, optional
        Airmass values. If None, it will be calculated.
    albedo : float, optional
        Ground reflectance. Default is 0.25.
    surface_type : str, optional
        Type of surface for albedo calculation. If None, a default value will be used.
    sky_model : str, optional
        Sky diffuse model to use. Default is `isotropic`.
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    model_perez : str, optional
        Perez model to use for diffuse irradiance. Default is `allsitescomposite1990`.
        Only used when sky_model is 'perez'.
    axis_tilt : float, optional
        Tilt angle of the tracker axis (degrees). Default is 0.0.
    axis_azimuth : float, optional
        Azimuth angle of the tracker axis (degrees). Default is 0.0.
    max_angle : float, optional
        Maximum rotation angle of the tracker (degrees). Default is 45.0.
    backtrack : bool, optional
        Whether to enable backtracking for single-axis trackers. Default is True.
    gcr : float, optional
        Ground coverage ratio of the tracker system. Default is 0.3.

    Returns
    -------
    tracker_poa : pandas.DataFrame
         Contains keys/columns 'poa_global', 'poa_direct', 'poa_diffuse',
         'poa_sky_diffuse', 'poa_ground_diffuse'. [W/m2]

    Notes
    -----
    This function uses pvlib to calculate the plane-of-array irradiance based on the provided weather data and
    mounting configuration for single-axis tracker systems. See the pvlib documentation for further information on the parameters.
    """

    if axis_azimuth is None:  # Sets the default orientation to north-south.
        try:
            axis_azimuth = float(meta["axis_azimuth"])
        except:
            if float(meta["latitude"]) < 0:
                axis_azimuth = 0
            else:
                axis_azimuth = 180
                print(
                    f"The array axis_azimuth was not provided, therefore an azimuth of {axis_azimuth:.1f} was used."
                )

    if sol_position is None:
        sol_position = solar_position(weather_df, meta)

    tracker_data = pvlib.tracking.singleaxis(
        sol_position["apparent_zenith"],
        sol_position["azimuth"],
        axis_tilt=axis_tilt,
        axis_azimuth=axis_azimuth,
        max_angle=max_angle,
        backtrack=backtrack,
        gcr=gcr,
        cross_axis_tilt=cross_axis_tilt,
    )

    tracker_poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tracker_data["surface_tilt"],
        surface_azimuth=tracker_data["surface_azimuth"],
        dni=weather_df["dni"],
        ghi=weather_df["ghi"],
        dhi=weather_df["dhi"],
        solar_zenith=sol_position["apparent_zenith"],
        solar_azimuth=sol_position["azimuth"],
        dni_extra=dni_extra,
        airmass=airmass,
        albedo=albedo,
        surface_type=surface_type,
        model=model,
        model_perez=model_perez,
    )

    return tracker_poa
