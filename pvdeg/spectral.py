"""
Collection of classes and functions to obtain spectral parameters.
"""

import pvlib


def solar_position(weather_df, meta):
    """
    Calculate solar position using pvlib based on weather data from the
    National Solar Radiation Database (NSRDB) for a given location (gid).

    Parameters
    ----------
    weather_df : pandas.DataFrame
        Weather data for given location.
    meta : pandas.Series
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


def poa_irradiance(
    weather_df, meta, sol_position=None, tilt=None, azimuth=None, sky_model="isotropic"
):
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
         'poa_sky_diffuse', 'poa_ground_diffuse'.
    """

    # TODO: change for handling HSAT tracking passed or requested
    if tilt is None:
        try:
            tilt = float(meta["tilt"])
        except:
            tilt = float(meta["latitude"])
            print(
                f"The array tilt angle was not provided, therefore the latitude tilt of {tilt:.1f} was used."
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
                    f"The array azimuth was not provided, therefore an azimuth of {azimuth:.1f} was used."
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
