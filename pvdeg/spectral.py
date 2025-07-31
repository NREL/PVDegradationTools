"""Collection of classes and functions to obtain spectral parameters."""

import pvlib
import pandas as pd
from pvdeg import decorators


@decorators.geospatial_quick_shape(
    "timeseries",
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
    """Calculate solar position.

    Calculate solar position using pvlib based on weather data from the National
    Solar Radiation Database (NSRDB) for a given location (gid).

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


@decorators.geospatial_quick_shape(
    "timeseries",
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
    **kwargs_irradiance,
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
    module_mount: string
        Module mounting configuration. Can either be `fixed` for fixed tilt systems or
        `1_axis` for single-axis tracker systems.
    sol_position : pd.DataFrame, optional
        pvlib.solarposition.get_solarposition Dataframe. If none is given, it will be calculated.
    kwargs_irradiance : dict
        Contains kwarg arguments for the poa model based on mounting configuration. See
        `poa_irradiance_fixed` or `poa_irradiance_tracker` for details.

    Returns
    -------
    poa : pandas.DataFrame
         Contains keys/columns 'poa_global', 'poa_direct', 'poa_diffuse',
         'poa_sky_diffuse', 'poa_ground_diffuse'. [W/m2]
    """

    if sol_position is None:
        sol_position = solar_position(weather_df, meta)

    if module_mount == "fixed":
        poa = poa_irradiance_fixed(weather_df, meta, sol_position, **kwargs_irradiance)
    elif module_mount == "1_axis":
        poa = poa_irradiance_tracker(
            weather_df, meta, sol_position, **kwargs_irradiance
        )
    else:
        raise NotImplementedError(
            f"The input module_mount '{module_mount}' is not implemented"
        )

    return poa.fillna(0)  # Fill NaN values with 0 for irradiance values


@decorators.geospatial_quick_shape(
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
    tilt=None,
    azimuth=None,
    sky_model="isotropic",
    **kwargs_irradiance,
) -> pd.DataFrame:
    """Calculate plane-of-array (POA) irradiance.

    Calculate plane-of-array (POA) irradiance using pvlib based on weather data from
    the National Solar Radiation Database (NSRDB) for a given location (gid).

    Parameters
    ----------
    weather_df : pd.DataFrame
        The file path to the NSRDB file.
    meta : dict
        The geographical location ID in the NSRDB file.
    sol_position : pd.DataFrame, optional
        pvlib.solarposition.get_solarposition Dataframe. If none is given, it will be
        calculated.
    tilt : float, optional
        The tilt angle of the PV panels in degrees, if None, the latitude of the
        location is used.
    azimuth : float, optional
        The azimuth angle of the PV panels in degrees. Equatorial facing by default.
    sky_model : str, optional
        The pvlib sky model to use, 'isotropic' by default.

    Returns
    -------
    poa : pandas.DataFrame
         Contains keys/columns 'poa_global', 'poa_direct', 'poa_diffuse',
         'poa_sky_diffuse', 'poa_ground_diffuse'. [W/m2]
    """
    # TODO: change for handling HSAT tracking passed or requested
    if tilt is None:
        try:
            tilt = float(meta["tilt"])
        except:
            tilt = float(abs(meta["latitude"]))
            print(
                f"The array tilt angle was not provided, therefore the latitude tilt \
                    of {tilt:.1f} was used."
            )
    if azimuth is None:  # Sets the default orientation to equator facing.
        try:
            azimuth = float(meta["azimuth"])
        except:
            if float(meta["latitude"]) < 0:
                azimuth = 0
            else:
                azimuth = 180
                print(
                    f"The array azimuth was not provided, therefore an azimuth of \
                        {azimuth:.1f} was used."
                )

    if sol_position is None:
        sol_position = solar_position(weather_df, meta)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=weather_df["dni"],
        ghi=weather_df["ghi"],
        dhi=weather_df["dhi"],
        solar_zenith=sol_position["apparent_zenith"],
        solar_azimuth=sol_position["azimuth"],
        model=sky_model,
    )

    return poa


@decorators.geospatial_quick_shape(
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
    sky_model="isotropic",
    **kwargs_irradiance,
) -> pd.DataFrame:
    """
    Calculate plane-of-array (POA) irradiance using pvlib based on supplied weather data.

    Parameters
    ----------
    weather_df : pd.DataFrame
        The weather data.
    meta : dict
        The geographical location information.
    sol_position : pd.DataFrame, optional
        pvlib.solarposition.get_solarposition Dataframe. If none is given, it will be calculated.
    axis_tilt : float, optional
        The tilt angle of the array along the long axis of a single axis tracker [degrees]. 
        If None, horizontal is used.
    axis_azimuth : float, optional
        The azimuth angle of the long, non-rotating axis of the array panels [degrees]. 
        North-south orientation by default.
    max_angle : float
        This is the maximum angle of the rotating axis achievable relative to the horizon.
    backtrack : boolean
        If true, the tilt will backtrack to avoid row to row shading.
    gcr : float
        Ground coverage ratio (GCR). The ratio of the width of the PV array to the distance between rows. 
        This affects the backtracking funciton.
    cross_axis_tilt : float
        angle, relative to horizontal, of the line formed by the intersection between the slope containing 
        the tracker axes and a plane perpendicular to the tracker axes [degrees]
        Fixes backtracking for a slope not parallel with the axis azimuth. 
    sky_model : str, optional
        The pvlib sky model to use, 'isotropic' by default.
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.

    Returns
    -------
    tracker_poa : pandas.DataFrame
         Contains keys/columns 'poa_global', 'poa_direct', 'poa_diffuse',
         'poa_sky_diffuse', 'poa_ground_diffuse'. [W/m2]
    """

    if axis_azimuth is None:  # Sets the default orientation to north-south.
        try:
            axis_azimuth = float(meta["axis_azimuth"])
        except:
            if float(meta["latitude"]) < 0:
                axis_azimuth = 0
            else:
                axis_azimuth = 180
                print(f"The array axis_azimuth was not provided, therefore an azimuth of {axis_azimuth:.1f} was used.")
                
    if axis_tilt is None:  # Sets the default orientation to horizontal.
        try:
            axis_tilt = float(meta["axis_tilt"])
        except:
            axis_tilt = 0
            print(f"The array axis_tilt was not provided, therefore an axis tilt of 0° was used.")

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
        model=sky_model,
    )

    return tracker_poa
