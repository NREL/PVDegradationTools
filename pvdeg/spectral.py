"""Collection of classes and functions to obtain spectral parameters."""

import pvlib
import pandas as pd
import inspect
import warnings
from pvdeg import decorators
import os
import re
from datetime import datetime as dt
from datetime import date
import bifacial_radiance as br
import numpy as np
import pandas as pd
from itertools import product
import bifacialvf
from pathlib import Path
from pvlib import iotools
import sys, platform
from pvdeg.utilities import read_material


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
    dni_extra=None,
    airmass=None,
    albedo=0.25,
    surface_type=None,
    sky_model="isotropic",
    model_perez="allsitescomposite1990",
    **kwargs_mount,
) -> pd.DataFrame:
    """Calculate plane-of-array (POA) irradiance.

    Calculate plane-of-array (POA) irradiance using
    `pvlib.irradiance.get_total_irradiance` for different module mounts
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
        pvlib.solarposition.get_solarposition Dataframe. If none is given, it will be
        calculated.
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
        Additional keyword arguments for the POA model based on mounting configuration.
        See `poa_irradiance_fixed` or `poa_irradiance_tracker` for details.

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
    This function uses pvlib to calculate the plane-of-array irradiance based on the
    provided weather data and mounting configuration. See the pvlib documentation for
    further information on the parameters.
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
                "Key '{extra_keys}' cannot be used for the selected `module_mount` "
                "and are ignored."
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
    Calculate plane-of-array (POA) irradiance using
    `pvlib.irradiance.get_total_irradiance` for a fixed tilt system.

    Parameters
    ----------
    weather_df : pd.DataFrame
        The file path to the NSRDB file.
    meta : dict
        The geographical location ID in the NSRDB file.
    sol_position : pd.DataFrame, optional
        pvlib.solarposition.get_solarposition Dataframe. If none is given, it will be
        calculated.
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
        DataFrame containing the calculated plane-of-array irradiance components with
        columns:
        'poa_global', 'poa_direct', 'poa_diffuse', 'poa_sky_diffuse',
        'poa_ground_diffuse'. [W/m2]


    Notes
    -----
    This function uses pvlib to calculate the plane-of-array irradiance based on the
    provided weather data and mounting configuration for fixed tilt systems. See the
    pvlib documentation for further information on the parameters.
    """

    if surface_tilt is None:
        try:
            surface_tilt = float(meta["tilt"])
        except Exception:
            surface_tilt = float(abs(meta["latitude"]))
            print(
                "The array surface_tilt angle was not provided, therefore the latitude "
                f"of {surface_tilt: .1f} was used."
            )

    if surface_azimuth is None:  # Sets the default orientation to equator facing.
        try:
            surface_azimuth = float(meta["azimuth"])
        except Exception:
            if float(meta["latitude"]) < 0:
                surface_azimuth = 0
            else:
                surface_azimuth = 180
                print(
                    "The array azimuth was not provided, therefore an azimuth of "
                    f"{surface_azimuth: .1f} was used."
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
    dni_extra=None,
    airmass=None,
    albedo=0.25,
    surface_type=None,
    model="isotropic",
    model_perez="allsitescomposite1990",
) -> pd.DataFrame:
    """
    Calculate plane-of-array (POA) irradiance using pvlib based on supplied weather
    data.

    Parameters
    ----------
    weather_df : pd.DataFrame
        The weather data.
    meta : dict
        The geographical location information.
    sol_position : pd.DataFrame, optional
        pvlib.solarposition.get_solarposition Dataframe. If none is given, it will be
        calculated.
    axis_tilt : float, optional
        The tilt angle of the array along the long axis of a single axis tracker
        [degrees].
        If None, horizontal is used.
    axis_azimuth : float, optional
        The azimuth angle of the long, non-rotating axis of the array panels [degrees].
        North-south orientation by default.
    max_angle : float
        This is the maximum angle of the rotating axis achievable relative to the
        horizon.
    backtrack : boolean
        If true, the tilt will backtrack to avoid row to row shading.
    gcr : float
        Ground coverage ratio (GCR). The ratio of the width of the PV array to the
        distance between rows. This affects the backtracking function.
    cross_axis_tilt : float
        angle, relative to horizontal, of the line formed by the intersection between
        the slope containing the tracker axes and a plane perpendicular to the tracker
        axes [degrees]
        Fixes backtracking for a slope not parallel with the axis azimuth.
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
    This function uses pvlib to calculate the plane-of-array irradiance based on the
    provided weather data and mounting configuration for single-axis tracker systems.
    See the pvlib documentation for further information on the parameters.
    """

    if axis_azimuth is None:  # Sets the default orientation to north-south.
        try:
            axis_azimuth = float(meta["axis_azimuth"])
        except Exception:
            if float(meta["latitude"]) < 0:
                axis_azimuth = 0
            else:
                axis_azimuth = 180
                print(
                    "The array axis_azimuth was not provided, therefore an azimuth of "
                    f"{axis_azimuth: .1f} was used."
                )

    if axis_tilt is None:  # Sets the default orientation to horizontal.
        try:
            axis_tilt = float(meta["axis_tilt"])
        except Exception:
            axis_tilt = 0
            print(
                "The array axis_tilt was not provided, therefore an axis tilt of 0° "
                "was used."
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

def spectrally_resolved_irradiance(weather_df: pd.DataFrame, meta: dict, wavelengths: np.ndarray, testfolder: Path = None, 
                                   spectra_folder: Path = None, 
                                   frontResultsOnly: bool = None,
                                   module_mount=None,
                                   tilt=None,
                                   azimuth=None,
                                   axis_tilt=None,
                                   axis_azimuth=None,
                                   custom_albedo_summer: str | dict = None,
                                   custom_albedo_winter: str | dict = None,
                                   min_wavelength=280, 
                                   max_wavelength=4000,
                                   **kwargs_irradiance: dict) -> pd.DataFrame:
    """
    Calculate spectrally resolved irradiance using SMARTS spectra

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data for given location.
    meta : dict
        Meta data of location.
    wavelengths : np.ndarray
        Wavelengths for which to calculate the irradiance [nm].
    testfolder : Path, optional
        Folder to save the spectra files. If None, no files are saved.
    irradiance_kwargs : dict
        Additional keyword arguments for the irradiance calculation.

    Returns
    -------
    spectral_irradiance : pd.DataFrame
        Spectrally resolved irradiance values.
    """
    if testfolder is not None:
        if not os.path.exists(testfolder):
            os.makedirs(testfolder)

    demo = br.RadianceObj('demo')
    # meta['TZ'] = meta['tz']  # Ensure timezone is set correctly
    metdata = demo.readWeatherData(metadata=meta, metdata=weather_df)
    print(metdata)
    myTMY3 = demo.metdata.tmydata
    print(myTMY3.columns)
    myTMY3.rename(columns={'dni': 'DNI',
                            'dhi': 'DHI',
                            'ghi': 'GHI',
                            'temp_air': 'DryBulb',
                            'wind_speed': 'Wspd',
                            'albedo': 'Alb'
                            }, inplace=True)       
    print(myTMY3)
    #Run once to generate spectra files, then comment out
    if spectra_folder is None:
        cwd = os.getcwd()
        alb, dni, dhi, ghi = demo.generate_spectra(ground_material='Grass', min_wavelength=min_wavelength, max_wavelength=max_wavelength)
        os.chdir(cwd)
    else: 
        print("Using existing spectra folder: ", spectra_folder)

    location_name = meta['State']+ '_' +  meta['City']
    
    demo.generate_spectral_tmys(wavelengths=wavelengths, source = "SAM", spectra_folder=r'spectra', location_name=location_name)
    #myTMY3 = weather_df
    # if custom_albedo_df is not None:
    #     # If a custom albedo DataFrame is provided, use it to set the albedo: 
    #     for alb_name in custom_albedo_df.columns:
    #         if alb_name in myTMY3.columns:
    #             myTMY3[alb_name] = custom_albedo_df[alb_name]
    custom_albedo_dict = {}
    if isinstance(custom_albedo_summer, str):
        custom_albedo_dict['Summer'] = read_material('Albedo', key=custom_albedo_summer)
    elif isinstance(custom_albedo_summer, dict):
        custom_albedo_dict['Summer'] = custom_albedo_summer
    else:
        custom_albedo_dict['Summer'] = None
    if isinstance(custom_albedo_winter, str):
        custom_albedo_dict['Winter'] = read_material('Albedo', key=custom_albedo_winter)
    elif isinstance(custom_albedo_winter, dict):
        custom_albedo_dict['Winter'] = custom_albedo_winter
    else:
        custom_albedo_dict['Winter'] = None

    meta = meta
    print(myTMY3)
    deltastyle = 'SAM'  # 

    # Variables
    if (module_mount == 'fixed'):
        # TODO: change for handling HSAT tracking passed or requested
        if tilt is None:
            try:
                tilt = float(meta["tilt"])
            except:
                tilt = float(abs(meta["latitude"]))
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
        tilt = tilt                 # PV tilt (deg)
        sazm = azimuth                  # PV Azimuth(deg) or tracker axis direction
        tracking = False
    elif (module_mount == '1_axis'):
        
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
            tilt = axis_tilt
            sazm = axis_azimuth
            tracking=True
            backtrack=False
            limit_angle = 50
    else:
        raise NotImplementedError(f"The input module_mount '{module_mount}' is not implemented")

    albedo = 0.62               # ground albedo
    clearance_height=0.4
    pitch = 1.5                   # row to row spacing in normalized panel lengths. 
    rowType = "interior"        # RowType(first interior last single)
    transFactor = 0.013         # TransmissionFactor(open area fraction)
    sensorsy = 6                # sensorsy(# hor rows in panel)   <--> THIS ASSUMES LANDSCAPE ORIENTATION 
    PVfrontSurface = "glass"    # PVfrontSurface(glass or ARglass)
    PVbackSurface = "glass"     # PVbackSurface(glass or ARglass)

    # Calculate PV Output Through Various Methods    
    # This variables are advanced and explored in other tutorials.
    #calculateBilInterpol = True         # Only works with landscape at the moment.
    #calculatePVMismatch = True
    #portraitorlandscape='landscape'   # portrait or landscape
    #cellsnum = 72
    #bififactor = 1.0
    #agriPV = True                       # Returns ground irradiance values

    if testfolder is None:
        testfolder = Path.cwd()
    else:
        testfolder = Path(testfolder)
    composite_file = os.path.join(testfolder, 'Spectrally_resolved_irradiance.csv')
    #spectra_folder = "C:\\Users\\mprillim\\Documents\\Spectral_example\\NSRDB_example\\Tutorial_01\\data\\spectral_tmys"

    spectrum_sums = demo.integrated_spectrum(spectra_folder=r'spectra')
    spectrum_sums.to_csv('spectrum_sums.csv')

    print(weather_df)
    composite_data = bifacialvf.skycomposition_method(myTMY3=myTMY3, spectral_file_path='data/spectral_tmys', lambda_range=wavelengths, integrated_spectrum=spectrum_sums, meta=meta, 
            custom_albedo_dict=custom_albedo_dict, frontResultsOnly=frontResultsOnly, writefiletitle=composite_file, 
            tilt=tilt, sazm=sazm, pitch=pitch, clearance_height=clearance_height, 
            rowType=rowType, transFactor=transFactor, sensorsy=sensorsy, 
            PVfrontSurface=PVfrontSurface, PVbackSurface=PVbackSurface, 
            albedo=albedo, tracking=tracking, backtrack=backtrack, 
            limit_angle=limit_angle, deltastyle=deltastyle)
    composite_data['RH'] = weather_df['relative_humidity']
    return composite_data


