"""
Collection of classes and functions to obtain spectral parameters.
"""

import pvlib
from numba import njit, prange, cuda
import numpy as np
import pandas as pd


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


# Deprication Warning: pvlib.spectrum.get_am15g was depricated in pvlib v0.11.0
# will be removed in v0.12.0, currently pvdeg should be using v0.10.3
def get_GTI_from_irradiance_340(irradiance_340: pd.Series) -> pd.Series:
    """
    Calculate the Global Tilt Irradiance of a module with 37 degrees of tile from irradiance at 340 nm.

    Parameters:
    -----------
    irradiance_340: pd.Series
        array of UV irradiance at 340 nm [W/m^2/nm @ 340nm]

    Returns:
    --------
    GTI: pd.Series
        Full spectrum Global Tilt Irradiance for a module at 37 degrees in [W/m^2]
    """

    am15 = pvlib.spectrum.get_am15g()
    wavelengths = am15.index.to_numpy(dtype=np.float64)
    spectrum = am15.to_numpy(dtype=np.float64)

    gti = _GTI_from_irradiance_340(
        irradiance_340=irradiance_340.to_numpy(dtype=np.float64),
        wavelengths=wavelengths,
        spectrum=spectrum,
    )

    return pd.Series(gti, index=irradiance_340.index, name="GTI")


# this is very inneficient
# we probably dont need to integrate here
# according to the spectrum, we know that the ratio of 340 to full spectrum is the same compared to the refenece so this is just a proportion calculation
@njit(parallel=True, cache=True)
def _GTI_from_irradiance_340(
    irradiance_340: np.ndarray, wavelengths: np.ndarray, spectrum: np.ndarray
) -> np.ndarray:
    gti = np.empty_like(irradiance_340)
    spectrum_340 = spectrum[120]  # 340nm at 120th index

    for i in prange(irradiance_340.shape[0]):
        scaling_factor = irradiance_340[i] / spectrum_340
        scaled_irradiance = scaling_factor * spectrum
        total_irradiance = np.trapz(scaled_irradiance, wavelengths)
        gti[i] = total_irradiance

    return gti
