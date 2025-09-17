"""Collection of classes and functions for standard development."""

import numpy as np
import pandas as pd
import dask.dataframe as dd
from typing import Union

# from gaps import ProjectPoints
from pvdeg import (
    temperature,
    spectral,
    weather,
    decorators,
)

# passing all tests after updating temperature models but this should be checked
# throughly before final release


@decorators.geospatial_quick_shape("timeseries", ["T_0", "T_inf", "poa"])
def eff_gap_parameters(
    weather_df=None,
    meta=None,
    weather_kwarg=None,
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    tilt=None,
    azimuth=None,
    wind_factor=0.33,
    model_kwarg={},
):
    """Calculate and set up data necessary to calculate the effective standoff distance.

    Calculation is for rooftop mounted PV system according to IEC TS 63126. The
    temperature is modeled for T_0 and T_inf and the corresponding test module
    temperature must be provided in the weather data.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    weather_kwarg : dict
        other variables needed to access a particular weather dataset.
    sky_model : str, optional
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    temp_model : str, optional
        Options: 'sapm'.  'pvsyst' and 'faiman' and others from PVlib.
        Performs the calculation for the cell temperature.
    conf_0 : str, optional
        Model for the high temperature module on the exponential decay curve.
        Default: 'insulated_back_glass_polymer'
    conf_inf : str, optional
        Model for the lowest temperature module on the exponential decay curve.
        Default: 'open_rack_glass_polymer'
    tilt : float, optional
        Tilt angle of PV system relative to horizontal. [°]
    azimuth : float, optional
        Azimuth angle of PV system relative to north. [°]
    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement
        heights between weather database (e.g. NSRDB) and the tempeature model
        (e.g. SAPM). The NSRDB provides calculations at 2 m (i.e module height) but
        SAPM uses a 10m height. It is recommended that a power-law relationship between
        height and wind speed of 0.33 be used*. This results in a wind speed that is
        1.7 times higher. It is acknowledged that this can vary significantly.

    References
    ----------
    R. Rabbani, M. Zeeshan, "Exploring the suitability of MERRA-2 reanalysis data for
    wind energy estimation, analysis of wind characteristics and energy potential
    assessment for selected sites in Pakistan", Renewable Energy 154 (2020) 1240-1251.

    Returns
    -------
    T_0 : float
        An array of temperature values for a module with an insulated back or an
        alternatively desired small or zero standoff, [°C]. Used as the basis for the
        maximum achievable temperature.
    T_inf : float
        An array of temperature values for a module that is rack mounted, [°C].
    poa : float
        An array of values for the plane of array irradiance, [W/m²]
    """
    parameters = ["temp_air", "wind_speed", "dhi", "ghi", "dni"]

    if isinstance(weather_df, dd.DataFrame):
        weather_df = weather_df[parameters].compute()
        weather_df.set_index("time", inplace=True)
    elif isinstance(weather_df, pd.DataFrame):
        weather_df = weather_df[parameters]
    elif weather_df is None:
        weather_df, meta = weather.get(**weather_kwarg)

    solar_position = spectral.solar_position(weather_df, meta)
    poa = spectral.poa_irradiance(
        weather_df,
        meta,
        sol_position=solar_position,
        tilt=tilt,
        azimuth=azimuth,
        sky_model=sky_model,
    )

    T_0 = temperature.temperature(
        cell_or_mod="cell",
        weather_df=weather_df,
        meta=meta,
        poa=poa,
        temp_model=temp_model,
        conf=conf_0,
        wind_factor=wind_factor,
        model_kwarg=model_kwarg,
    )

    T_inf = temperature.temperature(
        cell_or_mod="cell",
        weather_df=weather_df,
        meta=meta,
        poa=poa,
        temp_model=temp_model,
        conf=conf_inf,
        wind_factor=wind_factor,
        model_kwarg=model_kwarg,
    )

    return T_0, T_inf, poa


def eff_gap(T_0, T_inf, T_measured, T_ambient, poa, x_0=6.5, poa_min=400, t_amb_min=0):
    """Calculate the effective standoff distance.

    Calculate the effective standoff distance for rooftop mounted PV system according
    to IEC TS 63126. The 98ᵗʰ percentile calculations for T_0 and T_inf are also
    calculated.

    Parameters
    ----------
    T_0 : pd.series
        An array of temperature values for a module with an insulated back or an
        alternatively desired small or zero standoff. Used as the basis for the
        maximum achievable temperature, [°C].
    T_inf : pd.series
        An array of temperature values for a module that is rack mounted, [°C].
    T_measured : pd.series
        An array of values for the measured module temperature, [°C].
    T_ambient : pd.series
        An array of values for the ambient temperature, [°C].
    poa : pd.series
        An array of values for the plane of array irradiance, [W/m²]
    x_0 : float, optional
        Thermal decay constant [cm], [Kempe, PVSC Proceedings 2023].
        According to edition 2 of IEC TS 63126 a value of 6.5 cm is recommended.
    poa_min : float, optional
        Minimum iradiance
    t_ambient_min : floa, optional
        Minimum am

    Returns
    -------
    x_eff : float
        Effective module standoff distance. While not the actual physical standoff,
        this can be thought of as a heat transfer coefficient that produces results
        that are similar to the modeled singl module temperature with that gap.

    References
    ----------
    M. Kempe, et al. Close Roof Mounted System Temperature Estimation for Compliance
    to IEC TS 63126, PVSC Proceedings 2023
    """
    n = 0
    summ = 0
    for i in range(0, len(T_0)):
        if T_ambient.iloc[i] > t_amb_min:
            if poa.iloc[i] > poa_min:
                n = n + 1
                summ = summ + (T_0.iloc[i] - T_measured.iloc[i]) / (
                    T_0.iloc[i] - T_inf.iloc[i]
                )
    try:
        x_eff = -x_0 * np.log(1 - summ / n)
        # x_eff = np.multiply(
        # np.negative(x_0), np.log(np.subtract(1, np.divide(summ, n))))
    except RuntimeWarning:
        x_eff = (
            np.nan
        )  # results if the specified T₉₈ is cooler than an open_rack temperature
    if x_eff < 0:
        x_eff = 0

    return x_eff


# test conf for other temperature models
@decorators.geospatial_quick_shape(
    "numeric", ["x", "T98_0", "T98_inf"]
)  # numeric result, with corresponding datavariable names
def standoff(
    weather_df: pd.DataFrame = None,
    meta: dict = None,
    weather_kwarg: dict = None,
    tilt: Union[float, int, str] = None,
    azimuth: Union[float, int] = None,
    sky_model: str = "isotropic",
    temp_model: str = "sapm",
    conf_0: str = "insulated_back_glass_polymer",
    conf_inf: str = "open_rack_glass_polymer",
    conf_0_kwarg={},
    conf_inf_kwarg={},
    T98: float = 70,  # [°C]
    x_0: float = 6.5,  # [cm]
    wind_factor: float = 0.33,
    irradiance_kwarg={},
    tracker_irradiance_kwarg={},
    model_kwarg={},
) -> pd.DataFrame:
    """Calculate a minimum standoff distance for roof mounded PV systems.

    Will default
    to horizontal tilt. If the azimuth is not provided, it will use equator facing. You
    can use customized temperature models for the building integrated and the rack
    mounted configuration, but it will still assume an exponential decay.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    weather_kwarg : dict
        other variables needed to access a particular weather dataset.
    tilt : float, optional
        Tilt angle of rack mounted PV system relative to horizontal. [°]
        If single-axis tracker mounted, specify keyword 'single_axis'
    azimuth : float, optional
        Azimuth angle of PV system relative to north. [°]
    sky_model : str, optional
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    temp_model : str, optional
        Performs the calculations for the cell temperature.
        Options:
        `'sapm_cell'`,`'sapm_module'`,`'pvsyst_cell'`,`'faiman'`,`'faiman_rad'`,
        `'ross'`,`'noct_sam'`, `'fuentes'`, `'generic_linear'`.
        Note: we cannot simply drop in `pvsyst` using `conf_0=insulated` and
        `conf_inf=freestanding`. This will yield erroneous results as these
        configurtions represent different cases. Must provide equivalent
        `conf_0_kwarg` and `conf_inf_kwarg` between temperature models.
    conf_0 : str, optional
        Model for the high temperature module on the exponential decay curve.
        Default: 'insulated_back_glass_polymer'
    conf_inf : str, optional
        Model for the lowest temperature module on the exponential decay curve.
        Default: 'open_rack_glass_polymer'
    conf_0_kwarg : dict, optional
        keyword arguments for the high tempeature module on the exponential
        decay curve. Use for temperature models other than ``sapm`` model
        arguments representing an 'insulated_back_glass_polymer' module.
    conf_inf_kwarg : dict, optional
        keyword arguments for the lowest tempeature module on the exponential
        decay curve. Use for temperature models other than ``sapm`` model
        arguments representing an 'open_rack_glass_polymer' module.
    x_0 : float, optional
        Thermal decay constant (cm), [Kempe, PVSC Proceedings 2023]
    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement
        heights between weather database (e.g. NSRDB) and the tempeature model
        (e.g. SAPM). The NSRDB provides calculations at 2 m (i.e module height) but
        SAPM uses a 10m height. It is recommended that a power-law relationship between
        height and wind speed of 0.33 be used*. This results in a wind speed that is
        1.7 times higher. It is acknowledged that this can vary significantly.
    irradiance_kwarg : (dict, optional)
        keyword argument dictionary used for the poa irradiance caluation.
        options: ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``. See
        ``pvdeg.spectral.poa_irradiance``.
        Used in place of dedicated arguments in the case of a top down scenario
        method call.
    model_kwarg : dict, optional
        dictionary to provide to the temperature model, see temperature.temperature for
        more information

    R. Rabbani, M. Zeeshan, "Exploring the suitability of MERRA-2 reanalysis data for
    wind energy estimation, analysis of wind characteristics and energy potential
    assessment for selected sites in Pakistan", Renewable Energy 154 (2020) 1240-1251.

    Returns
    -------
    x : float [cm]
        Minimum installation distance in centimeter per IEC TS 63126 when the default
        settings are used. Effective gap "x" for the lower limit for Level 1 or Level 0
        modules (IEC TS 63216)
    T98_0 : float [°C]
        This is the 98ᵗʰ percential temperature of a theoretical module with no
        standoff.
    T98_inf : float [°C]
        This is the 98ᵗʰ percential temperature of a theoretical rack mounted module.

    References
    ----------
    M. Kempe, et al. Close Roof Mounted System Temperature Estimation for Compliance
    to IEC TS 63126, PVSC Proceedings 2023
    """
    parameters = ["temp_air", "wind_speed", "dhi", "ghi", "dni"]

    if isinstance(weather_df, dd.DataFrame):
        weather_df = weather_df[parameters].compute()
        weather_df.set_index("time", inplace=True)
    elif isinstance(weather_df, pd.DataFrame):
        weather_df = weather_df[parameters]
    elif weather_df is None:
        weather_df, meta = weather.get(**weather_kwarg)

    solar_position = spectral.solar_position(weather_df, meta)

    if tilt == "single_axis":
        irradiance_dict = {
            "sol_position": solar_position,
            "axis_azimuth": azimuth,
            "sky_model": sky_model,
        }
        poa = spectral.poa_irradiance_tracker(
            weather_df=weather_df,
            meta=meta,
            **irradiance_dict | tracker_irradiance_kwarg,
        )

    else:
        irradiance_dict = {
            "sol_position": solar_position,
            "tilt": tilt,
            "azimuth": azimuth,
            "sky_model": sky_model,
        }

        poa = spectral.poa_irradiance(
            weather_df=weather_df, meta=meta, **irradiance_dict | irradiance_kwarg
        )

    T_0 = temperature.temperature(
        cell_or_mod="cell",
        weather_df=weather_df,
        meta=meta,
        poa=poa,
        temp_model=temp_model,
        conf=conf_0,
        wind_factor=wind_factor,
        model_kwarg=model_kwarg | conf_0_kwarg,  # may lead to undesired behavior, test
    )
    T98_0 = T_0.quantile(q=0.98, interpolation="linear")

    T_inf = temperature.temperature(
        cell_or_mod="cell",
        weather_df=weather_df,
        meta=meta,
        poa=poa,
        temp_model=temp_model,
        conf=conf_inf,
        wind_factor=wind_factor,
        model_kwarg=model_kwarg
        | conf_inf_kwarg,  # may lead to undesired behavior, test
    )
    T98_inf = T_inf.quantile(q=0.98, interpolation="linear")

    try:
        x = -x_0 * np.log(1 - (T98_0 - T98) / (T98_0 - T98_inf))
    except RuntimeWarning:
        x = np.nan
        # results if the specified T₉₈ is cooler than an open_rack temperature
    if x < 0:
        x = 0

    res = {"x": x, "T98_0": T98_0, "T98_inf": T98_inf}
    df_res = pd.DataFrame.from_dict(res, orient="index").T

    return df_res


def interpret_standoff(standoff_1=None, standoff_2=None):
    """Interpret results of standoff calculations.

    This is a set of statments designed to provide a printable output to interpret
    the results of standoff calculations. At a minimum, data for Standoff_1 must be
    included.

    Parameters
    ----------
    Standoff_1 and Standoff_2 : df
    This is the dataframe output from the standoff calculation method for calculations
    at 70°C and 80°C, respectively.
        x : float [°C]
            Minimum installation distance in centimeter per IEC TS 63126 when the
            default settings are used. Effective gap "x" for the lower limit for Level 1
            or Level 0 modules (IEC TS 63216)
        T98_0 : float [°C]
            This is the 98ᵗʰ percential temperature of a theoretical module with no
            standoff.
        T98_inf : float [°C]
            This is the 98ᵗʰ percential temperature of a theoretical rack mounted
            module.

    Returns
    -------
    Output: str
        This is an interpretation of the accepatble effective standoff values suitable
        or presentation.
    """
    if standoff_1 is not None:
        x70 = standoff_1["x"].iloc[0]
        T98_0 = standoff_1["T98_0"].iloc[0]
        T98_inf = standoff_1["T98_inf"].iloc[0]
        if standoff_2 is not None:
            x80 = standoff_2["x"].iloc[0]
        else:
            try:
                x80 = -(-x70 / (np.log(1 - (T98_0 - 70) / (T98_0 - T98_inf)))) * np.log(
                    1 - (T98_0 - 80) / (T98_0 - T98_inf)
                )
            except RuntimeWarning:
                x80 = None
    else:
        x70 = None

    if x70 is None:
        Output = "Insufficient data for IEC TS 63126 Level determination."
    else:
        if T98_0 is not None:
            Output = (
                "The estimated T₉₈ of an insulated-back module is "
                + "%.1f" % T98_0
                + "°C. \n"
            )
        if T98_inf is not None:
            Output = (
                Output
                + "The estimated T₉₈ of an open-rack module is "
                + "%.1f" % T98_inf
                + "°C. \n"
            )
        if x80 is None:
            Output = (
                Output
                + "The minimum standoff for Level 0 certification and T₉₈<70°C is "
                + "%.1f" % x70
                + " cm."
            )
        else:
            Output = (
                Output
                + "Level 0 certification is valid for a standoff greather than "
                + "%.1f" % x70
                + " cm. \n"
            )
            if x70 > 0:
                if x80 > 0:
                    Output = (
                        Output
                        + "Level 1 certification is required for a standoff between "
                        + "%.1f" % x70
                        + " cm, and "
                        + "%.1f" % x80
                        + " cm. \n"
                    )
                    Output = (
                        Output
                        + "Level 2 certification is required for a standoff less than "
                        + "%.1f" % x80
                        + " cm."
                    )
                else:
                    Output = (
                        Output
                        + "Level 1 certification is required for a standoff less than "
                        + "%.1f" % x70
                        + " cm. \n"
                    )
                    Output = (
                        Output
                        + "Level 2 certification is never required for this temperature\
                            profile."
                    )

    return Output


@decorators.geospatial_quick_shape("numeric", ["T98"])
def T98_estimate(
    weather_df=None,
    meta=None,
    weather_kwarg=None,
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    wind_factor=0.33,
    tilt=None,
    azimuth=None,
    x_eff=None,
    x_0=6.5,
    model_kwarg={},
):
    """Estimate 98th percentile module temperature for given tilt, azimuth, and x_eff.

    Estimate the 98ᵗʰ percentile temperature for the module at the given tilt,
    azimuth, and x_eff. If any of these factors are supplied, it default to latitide
    tilt, equatorial facing, and open rack mounted, respectively.

    Parameters
    ----------
    x_eff : float
        This is the effective module standoff distance according to the model. [cm]
    x_0 : float, optional
        Thermal decay constant. [cm]
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    weather_kwarg : dict
        other variables needed to access a particular weather dataset.
    tilt : float,
        Tilt angle of PV system relative to horizontal. [°]
    azimuth : float, optional
        Azimuth angle of PV system relative to north. [°]
    sky_model : str, optional
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    temp_model : str, optional
        Options: 'sapm'.  'pvsyst' and 'faiman' will be added later.
        Performs the calculations for the cell temperature.
    conf_0 : str, optional
        Model for the high temperature module on the exponential decay curve.
        Default: 'insulated_back_glass_polymer'
    conf_inf : str, optional
        Model for the lowest temperature module on the exponential decay curve.
        Default: 'open_rack_glass_polymer'
    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement
        heights between weather database (e.g. NSRDB) and the tempeature model
        (e.g. SAPM). The NSRDB provides calculations at 2 m (i.e module height) but
        SAPM uses a 10m height. It is recommended that a power-law relationship between
        height and wind speed of 0.33 be used*. This results in a wind speed that is
        1.7 times higher. It is acknowledged that this can vary significantly.
    model_kwarg : dict, optional
        keyword argument dictionary to provide other arguments to the temperature model.
        See temperature.temperature for more information.

    R. Rabbani, M. Zeeshan, "Exploring the suitability of MERRA-2 reanalysis data for
    wind energy estimation, analysis of wind characteristics and energy potential
    assessment for selected sites in Pakistan", Renewable Energy 154 (2020) 1240-1251.

    Returns
    -------
    T98: float
        This is the 98ᵗʰ percential temperature for the module at the given tilt,
        azimuth, and x_eff.
    """
    parameters = ["temp_air", "wind_speed", "dhi", "ghi", "dni"]

    if isinstance(weather_df, dd.DataFrame):
        weather_df = weather_df[parameters].compute()
        weather_df.set_index("time", inplace=True)
    elif isinstance(weather_df, pd.DataFrame):
        weather_df = weather_df[parameters]
    elif weather_df is None:
        weather_df, meta = weather.get(**weather_kwarg)

    solar_position = spectral.solar_position(weather_df, meta)
    poa = spectral.poa_irradiance(
        weather_df=weather_df,
        meta=meta,
        sol_position=solar_position,
        tilt=tilt,
        azimuth=azimuth,
        sky_model=sky_model,
    )
    T_inf = temperature.temperature(
        cell_or_mod="cell",
        weather_df=weather_df,
        meta=meta,
        poa=poa,
        temp_model=temp_model,
        conf=conf_inf,
        wind_factor=wind_factor,
        model_kwarg=model_kwarg,
    )

    T98_inf = T_inf.quantile(q=0.98, interpolation="linear")

    if x_eff is None:
        return T98_inf
    else:
        T_0 = temperature.temperature(
            cell_or_mod="cell",
            weather_df=weather_df,
            meta=meta,
            poa=poa,
            temp_model=temp_model,
            conf=conf_0,
            wind_factor=wind_factor,
            model_kwarg=model_kwarg,
        )
        T98_0 = T_0.quantile(q=0.98, interpolation="linear")
        T98 = T98_0 - (T98_0 - T98_inf) * (1 - np.exp(-x_eff / x_0))

        return T98


def standoff_x(
    weather_df,
    meta,
    tilt,
    azimuth,
    sky_model,
    temp_model=None,
    conf_0=None,
    conf_inf=None,
    T98=None,
    x_0=None,
    wind_factor=None,
    model_kwarg={},
):
    """Calculate a minimum standoff distance for roof mounded PV systems.

    Will default
    to horizontal tilt and return only that value. It just passes through the calling
    function and returns a single value.

    Parameters
    ----------
    See Standoff() documentation

    Returns
    -------
    x : float [cm]
        Minimum installation distance in centimeter per IEC TS 63126 when the default
        settings are used.
        Effective gap "x" for the lower limit for Level 1 or Level 0 modules
        (IEC TS 63216)
    """
    temp_df = standoff(
        weather_df=weather_df,
        meta=meta,
        tilt=tilt,
        azimuth=azimuth,
        sky_model=sky_model,
        temp_model=temp_model,
        model_kwarg=model_kwarg,
        conf_0=conf_0,
        conf_inf=conf_inf,
        T98=T98,
        x_0=x_0,
        wind_factor=wind_factor,
    ).x[0]

    return temp_df


def x_eff_temperature_estimate(
    weather_df=None,
    meta=None,
    weather_kwarg=None,
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    wind_factor=0.33,
    module_mount=None,
    tilt=None,
    azimuth=None,
    x_eff=None,
    x_0=6.5,
    model_kwarg={},
):
    """
    Estimate the temperature for the module at the given tilt, azimuth, and x_eff.
    If any of these factors are not supplied, it default to latitude tilt, equatorial
    facing, and open rack mounted, respectively.

    Parameters
    ----------
    x_eff : float
        This is the effective module standoff distance according to the model. [cm]
    x_0 : float, optional
        Thermal decay constant. [cm]
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    weather_kwarg : dict
        other variables needed to access a particular weather dataset.
    tilt : float,
        Tilt angle of PV system relative to horizontal. [°]
    azimuth : float, optional
        Azimuth angle of PV system relative to north. [°]
    sky_model : str, optional
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    temp_model : str, optional
        Options: 'sapm'.  'pvsyst' and 'faiman' will be added later.
        Performs the calculations for the cell temperature.
    conf_0 : str, optional
        Model for the high temperature module on the exponential decay curve.
        Default: 'insulated_back_glass_polymer'
    conf_inf : str, optional
        Model for the lowest temperature module on the exponential decay curve.
        Default: 'open_rack_glass_polymer'
    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement
        heights between weather database (e.g. NSRDB) and the temperature model
        (e.g. SAPM). The NSRDB provides calculations at 2 m (i.e module height) but SAPM
        uses a 10 m height. It is recommended that a power-law relationship between
        height and wind speed of 0.33
        be used*. This results in a wind speed that is 1.7 times higher. It is
        acknowledged that this can vary significantly.
    model_kwarg : dict, optional
        keyword argument dictionary to provide other arguments to the temperature model.
        See temperature.temperature for more information.

    R. Rabbani, M. Zeeshan, "Exploring the suitability of MERRA-2 reanalysis data for
    wind energy estimation, analysis of wind characteristics and energy potential
    assessment for selected sites in Pakistan", Renewable Energy 154 (2020) 1240-1251.

    Returns
    -------
    T_x_eff: Pandas Dataframe
        This is the estimate for the module temperature at the given tilt, azimuth, and
        x_eff.

    """

    parameters = ["temp_air", "wind_speed", "dhi", "ghi", "dni"]

    if isinstance(weather_df, dd.DataFrame):
        weather_df = weather_df[parameters].compute()
        weather_df.set_index("time", inplace=True)
    elif isinstance(weather_df, pd.DataFrame):
        weather_df = weather_df[parameters]
    elif weather_df is None:
        weather_df, meta = weather.get(**weather_kwarg)

    solar_position = spectral.solar_position(weather_df, meta)
    poa = spectral.poa_irradiance(
        weather_df=weather_df,
        meta=meta,
        sol_position=solar_position,
        module_mount=module_mount,
        tilt=tilt,
        azimuth=azimuth,
        sky_model=sky_model,
    )
    T_inf = temperature.temperature(
        cell_or_mod="cell",
        weather_df=weather_df,
        meta=meta,
        poa=poa,
        temp_model=temp_model,
        conf=conf_inf,
        wind_factor=wind_factor,
        model_kwarg=model_kwarg,
    )

    if x_eff is None:
        return T_inf
    else:
        T_0 = temperature.temperature(
            cell_or_mod="cell",
            weather_df=weather_df,
            meta=meta,
            poa=poa,
            temp_model=temp_model,
            conf=conf_0,
            wind_factor=wind_factor,
            model_kwarg=model_kwarg,
        )
        T_x_eff = T_0 - (T_0 - T_inf) * (1 - np.exp(-x_eff / x_0))

        return T_x_eff
