"""Collection of classes and functions for humidity calculations."""

import numpy as np
import pandas as pd
from numba import jit

from pvdeg import temperature, spectral, decorators, utilities

# Constants
R_GAS = 0.00831446261815324  # Gas constant in kJ/(mol·K)


# TODO: When is dew_yield used?
@jit
def dew_yield(elevation, dew_point, dry_bulb, wind_speed, n):
    """Estimate the dew yield in [mm/day].

       Calculation taken from: Beysens,
       "Estimating dew yield worldwide from a few meteo data", Atmospheric Research 167
       (2016) 146-155.

    Parameters
    ----------
    elevation : int
        Site elevation [km]
    dew_point : float
        Dewpoint temperature in Celsius [°C]
    dry_bulb : float
        Air temperature "dry bulb temperature" [°C]
    wind_speed : float
        Air or windspeed measure [m/s]
    n : float
        Total sky cover(okta)
        This is a quasi emperical scale from 0 to 8 used in meterology which corresponds
        to 0-sky completely clear, to 8-sky completely cloudy. Does not account for
        cloud type or thickness.

    Returns
    -------
    dew_yield : float
        Amount of dew yield in [mm/day]
    """
    wind_speed_cut_off = 4.4
    dew_yield = (1 / 12) * (
        0.37
        * (
            1
            + (0.204323 * elevation)
            - (0.0238893 * elevation**2)
            - (18.0132 - (1.04963 * elevation**2) + (0.21891 * elevation**2))
            * (10 ** (-3) * dew_point)
        )
        * ((((dew_point + 273.15) / 285) ** 4) * (1 - (n / 8)))
        + (0.06 * (dew_point - dry_bulb))
        * (1 + 100 * (1 - np.exp(-((wind_speed / wind_speed_cut_off) ** 20))))
    )

    return dew_yield


def psat(temp, average=True):
    """Calculate water saturation temperature or dew point for given vapor pressure.

    Water vapor pressure model created from an emperical fit of ln(Psat) vs temperature
    using a 6th order polynomial fit. The fit produced
    R^2=0.999813. Calculation created by Michael Kempe, unpublished data.

    Parameters:
    -----------
    temp : series, float
        The air temperature (dry bulb) as a time-indexed series [C]
    average : boolean, default = True
        If true, return both psat serires and average psat (used for certain calcs)
    Returns:
    --------
    psat : array, float
        Saturation point
    avg_psat : float, optional
        mean saturation point for the series given
    """
    psat = np.exp(
        (3.2575315268e-13 * temp**6)
        - (1.5680734584e-10 * temp**5)
        + (2.2213041913e-08 * temp**4)
        + (2.3720766595e-7 * temp**3)
        - (4.0316963015e-04 * temp**2)
        + (7.9836323361e-02 * temp)
        - (5.6983551678e-1)
    )
    if average:
        return psat, psat.mean()
    else:
        return psat


def surface_outside(rh_ambient, temp_ambient, temp_module):
    """Calculate the Relative Humidity of a Solar Panel Surface at module temperature.

    Parameters
    ----------
    rh_ambient : float
        The ambient outdoor environmnet relative humidity [%].
    temp_ambient : float
        The ambient outdoor environmnet temperature [°C]
    temp_module : float
        The surface temperature of the solar panel module [°C]

    Returns
    --------
    rh_Surface : float
        The relative humidity of the surface of a solar module as a fraction or percent
        depending on input.
    """
    rh_Surface = rh_ambient * (psat(temp_ambient)[0] / psat(temp_module)[0])

    return rh_Surface


def _diffusivity_weighted_water(
    rh_ambient, temp_ambient, temp_module, So=1.81390702, Eas=16.729, Ead=38.14
):
    """Calculate weighted average module surface RH, helper function.
    """
    # Get the relative humidity of the surface
    rh_surface = surface_outside(rh_ambient, temp_ambient, temp_module)

    # Generate a series of the numerator values "prior to summation"
    numerator = (
        So
        * np.exp(-(Eas / (R_GAS * (temp_module + 273.15))))
        * rh_surface
        * np.exp(-(Ead / (R_GAS * (temp_module + 273.15))))
    )
    # get the summation of the numerator
    numerator = numerator.sum(axis=0, skipna=True)

    denominator = np.exp(-(Ead / (R_GAS * (temp_module + 273.15))))

    # get the summation of the denominator
    denominator = denominator.sum(axis=0, skipna=True)

    diffuse_water = (numerator / denominator) / 100

    return diffuse_water


def front_encap(
    rh_ambient, temp_ambient, temp_module, So=None, Eas=None, Ead=None,
    encapsulant="W001"
):
    """Calculate diffusivity weighted average Relative Humidity of the module surface.

    Parameters
    ----------
    rh_ambient : series (float)
        Ambient outdoor relative humidity. [%] Example: 50 = 50%, NOT .5 = 50%
    temp_ambient : series (float)
        Ambient outdoor temperature [°C]
    temp_module : pandas series (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    So : float
        Encapsulant solubility prefactor in [g/cm3]
        Will default to 1.81390702(g/cm3) which is the suggested value for EVA 001 if
        not specified.
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729(kJ/mol) is the suggested value for EVA.
    Ead : float
        Encapsulant diffusivity activation energy in [kJ/mol]
        Ead = 38.14(kJ/mol) is the suggested value for EVA.
    encapsulant : str
        This is the code number for the encapsulant. The default is EVA 'W001'.

    Return
    ------
    RHfront_series : pandas series (float)
        Relative Humidity of the photovoltaic module  frontside encapsulant. [%]
    """
    if So is None or Eas is None or Ead is None:
        So = utilities._read_material(
            name=encapsulant, fname="H2Opermeation", item=None, fp=None)["So"]
        Eas = utilities._read_material(
            name=encapsulant, fname="H2Opermeation", item=None, fp=None)["Eas"]
        Ead = utilities._read_material(
            name=encapsulant, fname="H2Opermeation", item=None, fp=None)["Ead"]
    diffuse_water = _diffusivity_weighted_water(
        rh_ambient=rh_ambient, temp_ambient=temp_ambient, temp_module=temp_module
    )

    RHfront_series = (
        diffuse_water
        / (So * np.exp(-(Eas / (R_GAS * (temp_module + 273.15)))))
    ) * 100

    return RHfront_series


def _csat(temp_module, So=1.81390702, Eas=16.729):
    """Return saturation of Water Concentration [g/cm³].

    Calculation is used in determining Relative Humidity of Backside Solar Module
    Encapsulant, and returns saturation of Water Concentration [g/cm³]

    Parameters
    ----------
    temp_module : pandas series (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    So : float
        Encapsulant solubility prefactor in [g/cm3]
        So = 1.81390702(g/cm3) is the suggested value for EVA.
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729(kJ/mol) is the suggested value for EVA.

    Returns
    -------
    Csat : pandas series (float)
        Saturation of Water Concentration [g/cm³]
    """
    # Saturation of water concentration
    Csat = So * np.exp(-(Eas / (R_GAS * (273.15 + temp_module))))

    return Csat


def _ceq(Csat, rh_SurfaceOutside):
    """
    Return Equilibration water concentration (g/cm³).

    Calculation is used in determining Relative Humidity of Backside Solar Module
    Encapsulant, and returns Equilibration water concentration (g/cm³)

    Parameters
    ----------
    Csat : pandas series (float)
        Saturation of Water Concentration (g/cm³)
    rh_SurfaceOutside : pandas series (float)
        The relative humidity of the surface of a solar module (%)

    Returns
    -------
    Ceq : pandas series (float)
        Equilibration water concentration (g/cm³)
    """
    Ceq = Csat * (rh_SurfaceOutside / 100)

    return Ceq


def Ce(
    temp_module,
    rh_surface,
    start=None,
    Po_b=None,
    Ea_p_b=None,
    t=None,
    So_e=None,
    Ea_s_e=None,
    back_encap_thickness=None,
    backsheet="W017",
    encapsulant="W001",
    output="rh",
):
    """Return water concentration in encapsulant.

    Calculation is used in determining Relative Humidity of Backside Solar Module
    Encapsulant. This function returns a numpy array of the Concentration of water in
    the encapsulant at every time step.

    This calculation uses a quasi-steady state approximation of the diffusion equation
    to calculate the concentration of water in the encapsulant. For this, it is assumed
    that the diffusion in the encapsulant is much larger than the diffusion in the
    backsheet, and it ignores the transients in the backsheet.

    Numba was used to isolate recursion requiring a for loop
    Numba Functions are very fast because they compile and run in machine code but can
    not use pandas dataframes.

    Parameters
    -----------
    temp_module : pandas series (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    rh_surface : list (float)
        The relative humidity of the surface of a solar module [%]
        EXAMPLE: "50 = 50% NOT .5 = 50%"
    start : float
        Initial value of the Concentration of water in the encapsulant.
        by default, the function will use half the equilibrium value as the first value
    Po_b : float
        Water permeation rate prefactor [g·mm/m²/day].
        The suggested value for PET W17 is Po = 1319534666.90318 [g·mm/m²/day].
    Ea_p_b : float
        Backsheet permeation  activation energy [kJ/mol] .
        For PET backsheet W017, Ea_p_b=55.4064573018373 [kJ/mol]
    t : float
        Thickness of the backsheet [mm].
        The suggested default for a PET backsheet is t=0.3 [mm]
    So_e : float
        Encapsulant solubility prefactor in [g/cm³]
        So = 1.81390702(g/cm³) is the suggested value for EVA W001.
    Ea_s_e : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729[kJ/mol] is the suggested value for EVA W001.
    back_encap_thickness : float
        Thickness of the backside encapsulant [mm].
        The suggested value for EVA encapsulant is 0.46 mm
    backsheet : str
        This is the code number for the backsheet.
        The default is PET 'W017'.
    encapsulant : str
        This is the code number for the encapsulant.
        The default is EVA 'W001'.
    output : str
        The default is "rh" which is the relative humidity in the encapsulant in [%],
        any other value, e.g. "Ce" will return the concentration in [g/cm³].


    Returns
    --------
    Ce_list : Pandas series (float)
        Concentration of water in the encapsulant at every time step in [g/cm³],
        or the relative humidity in [%] depending on the output parameter.

    """

    Ce_list = np.zeros(len(temp_module))
    Ce_out = temp_module
    if not isinstance(temp_module, np.ndarray):
        temp_module = temp_module.to_numpy()
    if not isinstance(rh_surface, np.ndarray):
        rh_surface = rh_surface.to_numpy()

    if Po_b is None or Ea_p_b is None:
        Po_b = utilities._read_material(
            name=backsheet, fname="H2Opermeation", item=None, fp=None
        )["Po"]
        Ea_p_b = utilities._read_material(
            name=backsheet, fname="H2Opermeation", item=None, fp=None
        )["Eap"]
        if t is None:
            if "t" in utilities._read_material(
                name=backsheet, fname="H2Opermeation", item=None, fp=None
            ):
                t = utilities._read_material(
                    name=backsheet, fname="H2Opermeation", item=None, fp=None
                )["t"]
            else:
                t = 0.3
    if So_e is None or Ea_s_e is None:
        So_e = utilities._read_material(
            name=encapsulant, fname="H2Opermeation", item=None, fp=None
        )["So"]
        Ea_s_e = utilities._read_material(
            name=encapsulant, fname="H2Opermeation", item=None, fp=None
        )["Eas"]
        if back_encap_thickness is None:
            if "t" in utilities._read_material(
                name=encapsulant, fname="H2Opermeation", item=None, fp=None
            ):
                back_encap_thickness = utilities._read_material(
                    name=encapsulant, fname="H2Opermeation", item=None, fp=None
                )["t"]
            else:
                back_encap_thickness = 0.46
    # Convert the parameters to the correct and convenient units
    WVTRo = Po_b / 100 / 100 / 24 / t
    EaWVTR = Ea_p_b / R_GAS
    So = So_e * back_encap_thickness / 10
    Eas = Ea_s_e / R_GAS
    # Ce is the initial start of concentration of water
    if start is None:
        Ce_start = (
            So * np.exp(-(Eas / (temp_module[0] + 273.15))) * rh_surface[0] / 100 / 2
        )
    else:
        Ce_start = start
        Ce_list[0] = _Ce(WVTRo, EaWVTR, temp_module, So, Eas, Ce_start, rh_surface)

    if output == "rh":
        # Convert the concentration to relative humidity
        Ce_list = 100 * (Ce_list / (So * np.exp(-(Eas / (temp_module + 273.15)))))
        Ce_list = pd.Series(Ce_list, name="RH_back_encapsulant")
    else:
        Ce_list = pd.Series(Ce_list, name="Ce_back_encapsulant")

    Ce_list.index = Ce_out.index
    return Ce_list


@jit
def _Ce(
    WVTRo,
    EaWVTR,
    temp_module,
    So,
    Eas,
    Ce_start,
    rh_surface,
):
    """
    This is a helper function for the Ce function that is used to calculate the
    concentration of water in the encapsulant.

    Returns
    --------
    Ce_list : Numba array (float)
        Concentration of water in the encapsulant at every time step in [g/cm³].

    """
    Ce = Ce_start
    for i in range(1, len(rh_surface)):
        Ce = Ce + (WVTRo * np.exp(-EaWVTR / (temp_module[i] + 273.15))) / (
            So * np.exp(-Eas / (temp_module[i] + 273.15))
        ) * (rh_surface[i] / 100 * So * np.exp(-Eas / (temp_module[i] + 273.15)) - Ce)

    return Ce


@jit
def Ce_numba(
    start,
    temp_module,
    rh_surface,
    WVTRo=7970633554,
    EaWVTR=55.0255,
    So=1.81390702,
    back_encap_thickness=0.5,
    Eas=16.729,
):
    """Return water concentration in encapsulant.

    Calculation is used in determining Relative Humidity of Backside Solar Module
    Encapsulant. This function returns a numpy array of the Concentration of water in
    the encapsulant at every time step.

    Numba was used to isolate recursion requiring a for loop
    Numba Functions compile and run in machine code but can not use pandas (Very fast).

    Parameters
    -----------
    start : float
        Initial value of the Concentration of water in the encapsulant
        currently takes the first value produced from
        the _ceq(Saturation of Water Concentration) as a point
        of acceptable equilibrium
    temp_module : pandas series (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    rh_Surface : list (float)
        The relative humidity of the surface of a solar module [%]
        EXAMPLE: "50 = 50% NOT .5 = 50%"
    WVTRo : float
        Water Vapor Transfer Rate prefactor [g/m2/day].
        The suggested value for EVA is WVTRo = 7970633554(g/m2/day).
    EaWVTR : float
        Water Vapor Transfer Rate activation energy [kJ/mol] .
        It is suggested to use 0.15[mm] thick PET as a default
        for the backsheet and set EaWVTR=55.0255[kJ/mol]
    So : float
        Encapsulant solubility prefactor in [g/cm3]
        So = 1.81390702(g/cm3) is the suggested value for EVA.
    back_encap_thickness : float
        Thickness of the backside encapsulant [mm].
        The suggested value for EVA encapsulant is 0.5 mm
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729[kJ/mol] is the suggested value for EVA.

    Returns
    --------
    Ce_list : numpy array
        Concentration of water in the encapsulant at every time step
    """
    dataPoints = len(temp_module)
    Ce_list = np.zeros(dataPoints)

    for i in range(0, len(rh_surface)):
        if i == 0:
            # Ce = Initial start of concentration of water
            Ce = start
        else:
            Ce = Ce_list[i - 1]

        Ce = Ce + (
            (
                WVTRo
                / 100
                / 100
                / 24
                * np.exp(
                    -((EaWVTR) / (R_GAS * (temp_module[i] + 273.15)))
                )
            )
            / (
                So
                * back_encap_thickness
                / 10
                * np.exp(-((Eas) / (R_GAS * (temp_module[i] + 273.15))))
            )
            * (
                rh_surface[i]
                / 100
                * So
                * np.exp(-((Eas) / (R_GAS * (temp_module[i] + 273.15))))
                - Ce
            )
        )

        Ce_list[i] = Ce

    return Ce_list


def back_encap(
    rh_ambient,
    temp_ambient,
    temp_module,
    WVTRo=7970633554,
    EaWVTR=55.0255,
    So=1.81390702,
    back_encap_thickness=0.5,
    Eas=16.729,
):
    """Return RH of backside module encapsulant.

    Function to calculate the Relative Humidity of Backside Solar Module Encapsulant
    and return a pandas series for each time step

    Parameters
    ----------
    rh_ambient : pandas series (float)
        The ambient outdoor environmnet relative humidity in [%]
        EXAMPLE: "50 = 50% NOT .5 = 50%"
    temp_ambient : pandas series (float)
        The ambient outdoor environmnet temperature in Celsius
    temp_module : list (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    WVTRo : float
        Water Vapor Transfer Rate prefactor [g/m2/day].
        The suggested value for EVA is WVTRo = 7970633554[g/m2/day].
    EaWVTR : float
        Water Vapor Transfer Rate activation energy [kJ/mol] .
        It is suggested to use 0.15[mm] thick PET as a default
        for the backsheet and set EaWVTR=55.0255[kJ/mol]
    So : float
        Encapsulant solubility prefactor in [g/cm3]
        So = 1.81390702[g/cm3] is the suggested value for EVA.
    back_encap_thickness : float
        Thickness of the backside encapsulant [mm].
        The suggested value for EVA encapsulant is 0.5 mm.
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729[kJ/mol] is the suggested value for EVA.

    Returns
    -------
    RHback_series : pandas series (float)
        Relative Humidity of Backside Solar Module Encapsulant [%]
    """
    rh_surface = surface_outside(
        rh_ambient=rh_ambient, temp_ambient=temp_ambient, temp_module=temp_module
    )

    Csat = _csat(temp_module=temp_module, So=So, Eas=Eas)
    Ceq = _ceq(Csat=Csat, rh_SurfaceOutside=rh_surface)

    start = Ceq.iloc[0]

    # Need to convert these series to numpy arrays for numba function
    temp_module_numba = temp_module.to_numpy()
    rh_surface_numba = rh_surface.to_numpy()
    Ce_nparray = Ce_numba(
        start=start,
        temp_module=temp_module_numba,
        rh_surface=rh_surface_numba,
        WVTRo=WVTRo,
        EaWVTR=EaWVTR,
        So=So,
        back_encap_thickness=back_encap_thickness,
        Eas=Eas,
    )

    RHback_series = 100 * (Ce_nparray / Csat)

    return RHback_series


def backsheet_from_encap(rh_back_encap, rh_surface_outside):
    """Calculate the Relative Humidity of solar module backsheet as timeseries.

    Requires the RH of the backside encapsulant and the outside surface of
    the module.

    Parameters
    ----------
    rh_back_encap : pandas series (float)
        Relative Humidity of Frontside Solar module Encapsulant. *See rh_back_encap()
    rh_surface_outside : pandas series (float)
        The relative humidity of the surface of a solar module.
        *See rh_surface_outside()

    Returns
    -------
    RHbacksheet_series : pandas series (float)
        Relative Humidity of Backside Backsheet of a Solar Module [%]
    """
    RHbacksheet_series = (rh_back_encap + rh_surface_outside) / 2

    return RHbacksheet_series


def backsheet(
    rh_ambient,
    temp_ambient,
    temp_module,
    start=None,
    Po_b=None,
    Ea_p_b=None,
    t=None,
    So_e=None,
    Ea_s_e=None,
    back_encap_thickness=None,
    backsheet="W017",
    encapsulant="W001",
):
    """
    Calculate the relative humidity in a solar module backsheet as timeseries.
    It assume a value that is the average of the RH of the backside encapsulant and the
    outside surface of the module.

    Parameters
    ----------
    rh_ambient : pandas series (float)
        The ambient outdoor environmnet relative humidity in (%)
        EXAMPLE: "50 = 50% NOT .5 = 50%"
    temp_ambient : pandas series (float)
        The ambient outdoor environmnet temperature in Celsius
    temp_module : list (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    start : float
        Initial value of the Concentration of water in the encapsulant.
        by default, the function will use an equilibrium value as the first value
    Po_b : float
        Water permeation rate prefactor [g·mm/m²/day].
        The suggested value for PET W17 is Po = 1319534666.90318 [g·mm/m²/day].
    Ea_p_b : float
        Backsheet permeation  activation energy [kJ/mol] .
        For PET backsheet W017, Ea_p_b=55.4064573018373 [kJ/mol]
    t : float
        Thickness of the backsheet [mm].
        The suggested default for a PET backsheet is t=0.3 [mm]
    So_e : float
        Encapsulant solubility prefactor in [g/cm³]
        So = 1.81390702(g/cm³) is the suggested value for EVA W001.
    Ea_s_e : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729[kJ/mol] is the suggested value for EVA W001.
    back_encap_thickness : float
        Thickness of the backside encapsulant [mm].
        The suggested value for EVA encapsulant  is 0.46 mm.
    backsheet : str
        This is the code number for the backsheet.
        The default is PET 'W017'.
    encapsulant : str
        This is the code number for the encapsulant.
        The default is EVA 'W001'.

    Returns
    --------
    rh_backsheet : float series or array
        relative humidity of the PV backsheet as a time-series [%]
    """

    # Get the relative humidity of the surface
    surface = surface_outside(
        rh_ambient=rh_ambient, temp_ambient=temp_ambient, temp_module=temp_module
    )

    # Get the relative humidity of the back encapsulant
    RHback_series = Ce(
        rh_surface=surface,
        # temp_ambient=temp_ambient,
        temp_module=temp_module,
        start=start,
        Po_b=Po_b,
        Ea_p_b=Ea_p_b,
        t=t,
        So_e=So_e,
        Ea_s_e=Ea_s_e,
        back_encap_thickness=back_encap_thickness,
        backsheet=backsheet,
        encapsulant=encapsulant,
        output="rh",
    )

    return (RHback_series + surface) / 2


@decorators.geospatial_quick_shape(
    "timeseries",
    ["RH_surface_outside", "RH_front_encap", "RH_back_encap", "RH_backsheet"],
)
def module(
    weather_df,
    meta,
    tilt=None,
    azimuth=180,
    sky_model="isotropic",
    temp_model="sapm",
    conf="open_rack_glass_glass",
    WVTRo=7970633554,
    EaWVTR=55.0255,
    So=1.81390702,
    back_encap_thickness=0.5,
    Eas=16.729,
    wind_factor=0.33,
):
    """Calculate the Relative Humidity of solar module backsheet from timeseries data.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    tilt : float, optional
        Tilt angle of PV system relative to horizontal.
    azimuth : float, optional
        Azimuth angle of PV system relative to north.
    sky_model : str, optional
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    temp_model : str, optional
        Options: 'sapm', 'pvsyst', 'faiman', 'sandia'.
    mount_type : str, optional
        Options: 'insulated_back_glass_polymer',
                 'open_rack_glass_polymer'
                 'close_mount_glass_glass',
                 'open_rack_glass_glass'
    WVTRo : float
        Water Vapor Transfer Rate prefactor (g/m2/day).
        The suggested value for EVA is WVTRo = 7970633554(g/m2/day).
    EaWVTR : float
        Water Vapor Transfer Rate activation energy (kJ/mol) .
        It is suggested to use 0.15(mm) thick PET as a default
        for the backsheet and set EaWVTR=55.0255(kJ/mol)
    So : float
        Encapsulant solubility prefactor in [g/cm3]
        So = 1.81390702(g/cm3) is the suggested value for EVA.
    back_encap_thickness : float
        Thickness of the backside encapsulant (mm).
        The suggested value for EVA encapsulant is 0.5
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729(kJ/mol) is the suggested value for EVA.
    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement
        heights between weather database (e.g. NSRDB) and the tempeature model
        (e.g. SAPM). The NSRDB provides calculations at 2 m (i.e module height) but SAPM
        uses a 10m height. It is recommended that a power-law relationship between
        height and wind speed of 0.33 be used*. This results in a wind speed that is
        1.7 times higher. It is acknowledged that this can vary significantly.

    Returns
    --------
    rh_backsheet : float series or array
        relative humidity of the PV backsheet as a time-series
    """
    # solar_position = spectral.solar_position(weather_df, meta)
    # poa = spectral.poa_irradiance(weather_df, meta, solar_position, tilt, azimuth,
    # sky_model)
    # temp_module = temperature.module(weather_df, poa, temp_model, mount_type,
    # wind_factor)

    poa = spectral.poa_irradiance(
        weather_df=weather_df,
        meta=meta,
        tilt=tilt,
        azimuth=azimuth,
        sky_model=sky_model,
    )

    temp_module = temperature.module(
        weather_df=weather_df,
        meta=meta,
        poa=poa,
        temp_model=temp_model,
        conf=conf,
        wind_factor=wind_factor,
    )

    rh_surface_outside = surface_outside(
        rh_ambient=weather_df["relative_humidity"],
        temp_ambient=weather_df["temp_air"],
        temp_module=temp_module,
    )

    rh_front_encap = front_encap(
        rh_ambient=weather_df["relative_humidity"],
        temp_ambient=weather_df["temp_air"],
        temp_module=temp_module,
        So=So,
        Eas=Eas,
    )

    rh_back_encap = back_encap(
        rh_ambient=weather_df["relative_humidity"],
        temp_ambient=weather_df["temp_air"],
        temp_module=temp_module,
        WVTRo=WVTRo,
        EaWVTR=EaWVTR,
        So=So,
        back_encap_thickness=back_encap_thickness,
        Eas=Eas,
    )

    rh_backsheet = backsheet_from_encap(
        rh_back_encap=rh_back_encap, rh_surface_outside=rh_surface_outside
    )

    data = {
        "RH_surface_outside": rh_surface_outside,
        "RH_front_encap": rh_front_encap,
        "RH_back_encap": rh_back_encap,
        "RH_backsheet": rh_backsheet,
    }
    results = pd.DataFrame(data=data)
    return results
