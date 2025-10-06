"""Collection of classes and functions for humidity calculations."""

import numpy as np
import pandas as pd
from numba import jit
import warnings

from pvdeg import temperature, spectral, decorators, utilities

# Constants
R_GAS = 0.00831446261815324  # Gas constant in kJ/(mol·K)


def relative(temperature_air, dew_point):
    """Calculate ambient relative humidity from dry bulb air temperature and dew point.

    References
    ----------
    Alduchov, O. A., and R. E. Eskridge, 1996: Improved Magnus' form approximation of
    saturation vapor pressure. J. Appl. Meteor., 35, 601-609.
    August, E. F., 1828: Ueber die Berechnung der Expansivkraft des Wasserdunstes. Ann.
    Phys. Chem., 13, 122-137.
    Magnus, G., 1844: Versuche über die Spannkräfte des Wasserdampfs. Ann. Phys. Chem.,
    61, 225-247.

    Parameters
    ----------
    temperature_air : pd.Series or float
        Series or float of ambient air temperature. [°C]

    dew_point : pd.Series or float
        Series or float of dew point temperature. [°C]

    Notes
    -----
    Passing NaN values in either ``temperature_air`` or ``dew_point`` at any index
    position will return NaN values in the output at those same position(s) in
    ``relative_humidity``.

    Returns
    -------
    relative_humidity : pd.Series or float
        Series or float of ambient relative humidity. [%]
    """
    if (
        (isinstance(temperature_air, pd.Series) and temperature_air.isna().any())
        or (isinstance(dew_point, pd.Series) and dew_point.isna().any())
        or (isinstance(temperature_air, float) and pd.isna(temperature_air))
        or (isinstance(dew_point, float) and pd.isna(dew_point))
    ):
        warnings.warn(
            "Input contains NaN values. Output will contain NaNs at those positions."
        )

    num = np.exp(17.625 * dew_point / (243.04 + dew_point))
    den = np.exp(17.625 * temperature_air / (243.04 + temperature_air))

    return 100 * num / den


# @jit
def dew_yield(
    elevation: float, dew_point: float, dry_bulb: float, wind_speed: float, n: float
):
    """Estimate the dew yield in [mm/day].
    This may be useful for degradation modeling where the presence of water is a
    factor. E.g. much greater surface conductivity on glass promoting potential
    induced degradation (PID).

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


def water_saturation_pressure(temp, average=True):
    """Calculate the water saturation temperature or dew point for given vapor pressure.

    Water saturation pressure (psat) model created from an emperical fit of
    ln(psat) vs temperature using a 6th order polynomial fit. The fit produced
    R^2=0.999813. Calculation created by Michael Kempe, unpublished data.
    The fit used data from -40°C to 200°C.

    Parameters:
    -----------
    temp : series, float
        The air temperature (dry bulb) as a time-indexed series [°C]
    average : boolean, default = True
        If true, return both water saturation pressure serires and the average water
        saturation pressure (used for certain calcs)
    Returns:
    --------
    water_saturation_pressure : array, float
        Saturation point
    avg_water_saturation_pressure : float, optional
        Mean saturation point for the series given
    """
    water_saturation_pressure = np.exp(
        (3.2575315268e-13 * temp**6)
        - (1.5680734584e-10 * temp**5)
        + (2.2213041913e-08 * temp**4)
        + (2.3720766595e-7 * temp**3)
        - (4.0316963015e-04 * temp**2)
        + (7.9836323361e-02 * temp)
        - (5.6983551678e-1)
    )
    if average:
        return water_saturation_pressure, water_saturation_pressure.mean()
    else:
        return water_saturation_pressure


def surface_relative(rh_ambient, temp_ambient, temp_module):
    """Calculate the relative humidity on a solar panel surface at the module
    temperature.

    Parameters
    ----------
    rh_ambient : pd series, float
        The ambient outdoor environmnet relative humidity [%].
    temp_ambient : pd series, float
        The ambient outdoor environmnet temperature [°C]
    temp_module : pd series, float
        The surface temperature of the solar panel module [°C]

    Returns
    --------
    rh_Surface : float
        The relative humidity of the surface of a solar module as a fraction or percent
        depending on input.
    """
    rh_Surface = rh_ambient * (
        water_saturation_pressure(temp_ambient)[0]
        / water_saturation_pressure(temp_module)[0]
    )

    return rh_Surface


def diffusivity_weighted_water(
    rh_ambient,
    temp_ambient,
    temp_module,
    So=None,
    Eas=None,
    Ead=None,
    encapsulant="W001",
):
    """Calculate the diffusivity weighted average module surface RH.

    Parameters
    ----------
    rh_ambient : series (float)
        Ambient outdoor relative humidity. [%] Example: 50 = 50%, NOT 0.5 = 50%
    temp_ambient : series (float)
        Ambient outdoor temperature [°C]
    temp_module : pandas series (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    So : float
        Encapsulant solubility prefactor in [g/cm³]
        Will default to 1.81390702[g/cm³] which is the suggested value for EVA 'W001' if
        not specified.
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729 [kJ/mol] is the suggested value for EVA.
    Ead : float
        Encapsulant diffusivity activation energy in [kJ/mol]
        Ead = 38.14 [kJ/mol] is the suggested value for EVA.
    encapsulant : str
        This is the code number for the encapsulant. The default is EVA 'W001'.

    Return
    ------
    diffuse_weighted_water : pandas series (float)
        Average water content in equilibrium with the module surface, weighted
        by the encapsulant diffusivity in [g/cm³].
    """

    if So is None:
        So = utilities.read_material_property(
            key=encapsulant, parameters=["So"], pvdeg_file="H2Opermeation"
        )["So"]
    if Eas is None:
        Eas = utilities.read_material_property(
            key=encapsulant, parameters=["Eas"], pvdeg_file="H2Opermeation"
        )["Eas"]
    if Ead is None:
        Ead = utilities.read_material_property(
            key=encapsulant, parameters=["Ead"], pvdeg_file="H2Opermeation"
        )["Ead"]

    # Get the relative humidity of the surface
    rh_surface = surface_relative(rh_ambient, temp_ambient, temp_module)

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

    diffuse_weighted_water = (numerator / denominator) / 100
    diffuse_weighted_water = (numerator / denominator) / 100

    return diffuse_weighted_water


def front_encapsulant(
    rh_ambient,
    temp_ambient,
    temp_module,
    So=None,
    Eas=None,
    Ead=None,
    encapsulant="W001",
):
    """Calculate diffusivity weighted average Relative Humidity of the module surface.

    Parameters
    ----------
    rh_ambient : series (float)
        Ambient outdoor relative humidity. [%] Example: 50 = 50%, NOT 0.5 = 50%
    temp_ambient : series (float)
        Ambient outdoor temperature [°C]
    temp_module : pandas series (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    So : float
        Encapsulant solubility prefactor in [g/cm³]
        Will default to 1.81390702[g/cm³] which is the suggested value for EVA 001 if
        not specified.
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729 [kJ/mol] is the suggested value for EVA.
    Ead : float
        Encapsulant diffusivity activation energy in [kJ/mol]
        Ead = 38.14 [kJ/mol] is the suggested value for EVA.
    encapsulant : str
        This is the code number for the encapsulant. The default is EVA 'W001'.

    Return
    ------
    front_encapsulant : pandas series (float)
        Relative Humidity of the photovoltaic module  frontside encapsulant. [%]
    """
    if So is None:
        So = utilities.read_material_property(
            key=encapsulant, parameters=["So"], pvdeg_file="H2Opermeation"
        )["So"]
    if Eas is None:
        Eas = utilities.read_material_property(
            key=encapsulant, parameters=["Eas"], pvdeg_file="H2Opermeation"
        )["Eas"]
    if Ead is None:
        Ead = utilities.read_material_property(
            key=encapsulant, parameters=["Ead"], pvdeg_file="H2Opermeation"
        )["Ead"]

    diffuse_water = diffusivity_weighted_water(
        rh_ambient=rh_ambient,
        temp_ambient=temp_ambient,
        temp_module=temp_module,
        So=So,
        Eas=Eas,
        Ead=Ead,
    )

    front_encapsulant = (
        diffuse_water / (So * np.exp(-(Eas / (R_GAS * (temp_module + 273.15)))))
    ) * 100

    return front_encapsulant


def csat(temp_module, So=None, Eas=None, encapsulant="W001"):
    """Return saturation of Water Concentration [g/cm³].

    Calculation is used in determining Relative Humidity of Backside Solar Module
    Encapsulant, and returns saturation of Water Concentration [g/cm³].
    For most coding, it is better to just run the calculation insitu. The code is
    here for completeness and for informational purposes.

    Parameters
    ----------
    temp_module : pandas series (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    So : float
        Encapsulant solubility prefactor in [g/cm³]
    Eas : float
        Encapsulant solubility activation energy in [kJ/mol]
    encapsulant : str
        This is the code number for the encapsulant.
        The default is EVA 'W001'.

    Returns
    -------
    Csat : pandas series (float)
        Saturation of Water Concentration [g/cm³]
    """

    if So is None:
        So = utilities.read_material_property(
            key=encapsulant, parameters=["So"], pvdeg_file="H2Opermeation"
        )["So"]
    if Eas is None:
        Eas = utilities.read_material_property(
            key=encapsulant, parameters=["Eas"], pvdeg_file="H2Opermeation"
        )["Eas"]

    # Saturation of water concentration
    Csat = So * np.exp(-(Eas / (R_GAS * (273.15 + temp_module))))

    return Csat


def ceq(Csat, rh_SurfaceOutside):
    """
    Return Equilibration water concentration [g/cm³].

    Calculation is used in determining Relative Humidity of Backside Solar Module
    Encapsulant, and returns Equilibration water concentration [g/cm³].
    For most coding, it is better to just run the calculation insitu. The code is
    here for completeness and for informational purposes.

    Parameters
    ----------
    Csat : pandas series (float)
        Saturation of Water Concentration [g/cm³]
    rh_SurfaceOutside : pandas series (float)
        The relative humidity of the surface of a solar module (%)

    Returns
    -------
    Ceq : pandas series (float)
        Equilibration water concentration [g/cm³]
    """
    Ceq = Csat * (rh_SurfaceOutside / 100)

    return Ceq


def back_encapsulant_water_concentration(
    temp_module=None,
    rh_surface=None,
    rh_ambient=None,
    temp_ambient=None,
    start=None,
    Po_b=None,
    Ea_p_b=None,
    backsheet_thickness=None,
    So_e=None,
    Ea_s_e=None,
    back_encap_thickness=None,
    backsheet="W017",
    encapsulant="W001",
    output="rh",
):
    """Return water concentration in encapsulant.

    Calculation is used in determining Relative Humidity of Backside Solar Module
    Encapsulant. This function returns a numpy array of the Concentration of water
    in the encapsulant at every time step.

    This calculation uses a quasi-steady state approximation of the diffusion
    equation to calculate the concentration of water in the encapsulant. For this,
    it is assumed that the diffusion in the encapsulant is much larger than the
    diffusion in the backsheet, and it ignores the transients in the backsheet.

    Numba was used to isolate recursion requiring a for loop
    Numba Functions are very fast because they compile and run in machine code but
    can  not use pandas dataframes.

    Parameters
    -----------
    temp_module : pandas series (float)
        The surface temperature in Celsius of the solar panel module
        "module temperature [°C]"
    rh_surface : list (float)
        The relative humidity of the surface of a solar module [%]
        EXAMPLE: "50 = 50% NOT 0.5 = 50%"
        if this parameter is not provided, it will be calculated using rh_ambient and
        temp_ambient.
    rh_ambient : series (float)
        Ambient outdoor relative humidity. [%] Example: 50 = 50%, NOT 0.5 = 50%
        If rh_surface is not provided, this parameter along with temp_ambient will be
        used to calculate it.
    temp_ambient : series (float)
        Ambient outdoor temperature [°C]
        If rh_surface is not provided, this parameter along with rh_ambient will be used
        to calculate it.
    start : float
        Initial value of the Concentration of water in the encapsulant.
        by default, the function will use half the equilibrium value as the first
        value
    Po_b : float
        Water permeation rate prefactor [g·mm/m²/day].
        The suggested value for PET W17 is Po = 1319534666.90318 [g·mm/m²/day].
    Ea_p_b : float
        Backsheet permeation  activation energy [kJ/mol] .
        For PET backsheet W017, Ea_p_b=55.4064573018373 [kJ/mol]
    backsheet_thickness : float
        Thickness of the backsheet [mm].
        The suggested value for a PET backsheet_thickness=0.3.
    So_e : float
        Encapsulant solubility prefactor in [g/cm³]
        So = 1.81390702[g/cm³] is the suggested value for EVA W001.
    Ea_s_e : float
        Encapsulant solubility activation energy in [kJ/mol]
        Eas = 16.729[kJ/mol] is the suggested value for EVA W001.
    back_encap_thickness : float
        Thickness of the backside encapsulant [mm].
        The suggested value for EVA encapsulant is 0.46mm
    backsheet : str
        This is the code number for the backsheet.
        The default is PET 'W017'.
    encapsulant : str
        This is the code number for the encapsulant.
        The default is EVA 'W001'.
    output : str
        The default is "rh" which is the relative humidity in the encapsulant
        in [%], any other value, e.g. "Ce" will return the concentration in [g/cm³].


    Returns
    --------
    Ce_list : Pandas series (float)
        Concentration of water in the encapsulant at every time step in [g/cm³],
        or the relative humidity in [%] depending on the output parameter.

    """
    if rh_surface is None:
        if rh_ambient is None or temp_ambient is None:
            raise ValueError(
                "If rh_surface is not provided, both rh_ambient and temp_ambient must"
                "be provided."
            )
        # Get the relative humidity of the surface
        rh_surface = surface_relative(
            rh_ambient=rh_ambient, temp_ambient=temp_ambient, temp_module=temp_module
        )

    Ce_list = np.zeros(len(temp_module))
    index_passthrough_variable = temp_module.index
    if not isinstance(temp_module, np.ndarray):
        temp_module = temp_module.to_numpy()
    if not isinstance(rh_surface, np.ndarray):
        rh_surface = rh_surface.to_numpy()

    if Po_b is None:
        Po_b = utilities.read_material_property(
            key=backsheet, parameters=["Po"], pvdeg_file="H2Opermeation"
        )["Po"]
    if Ea_p_b is None:
        Ea_p_b = utilities.read_material_property(
            key=backsheet, parameters=["Eap"], pvdeg_file="H2Opermeation"
        )["Eap"]
    if backsheet_thickness is None:
        try:
            backsheet_thickness = utilities.read_material_property(
                key=backsheet, parameters=["t"], pvdeg_file="H2Opermeation"
            )["t"]
            if backsheet_thickness is None:
                raise ValueError()
        except (KeyError, ValueError):
            raise ValueError(
                "backsheet_thickness must be specified as a float or "
                "a backsheet material with a backsheet_thickness "
                "available should be specified."
            )
    if So_e is None:
        So_e = utilities.read_material_property(
            key=encapsulant, parameters=["So"], pvdeg_file="H2Opermeation"
        )["So"]
    if Ea_s_e is None:
        Ea_s_e = utilities.read_material_property(
            key=encapsulant, parameters=["Eas"], pvdeg_file="H2Opermeation"
        )["Eas"]
    if back_encap_thickness is None:
        try:
            back_encap_thickness = utilities.read_material_property(
                key=encapsulant, parameters=["t"], pvdeg_file="H2Opermeation"
            )["t"]
            if back_encap_thickness is None:
                raise ValueError()
        except (KeyError, ValueError):
            raise ValueError(
                "back_encap_thickness must be specified as a float or "
                "an encapsulant material with a back_encap_thickness "
                "available should be specified."
            )
    # Convert the parameters to the correct and convenient units
    WVTRo = Po_b / 100 / 100 / 24 / backsheet_thickness
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

    Ce_list = _Ce(
        WVTRo=WVTRo,
        EaWVTR=EaWVTR,
        temp_module=temp_module,
        So=So,
        Eas=Eas,
        Ce_start=Ce_start,
        rh_surface=rh_surface,
    )

    if output == "rh":
        # Convert the concentration to relative humidity
        Ce_list = 100 * (Ce_list / (So * np.exp(-(Eas / (temp_module + 273.15)))))
        Ce_list = pd.Series(
            Ce_list, index=index_passthrough_variable, name="RH_back_encapsulant"
        )
    else:
        Ce_list = pd.Series(
            Ce_list, index=index_passthrough_variable, name="Ce_back_encapsulant"
        )

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

    Parameters
    -----------
    All the parameters must be Numba compilable types. I.e. numpy arrays or floats.

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
    Encapsulant. This function returns a numpy array of the Concentration of water
    in the encapsulant at every time step.

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
                * np.exp(-((EaWVTR) / (R_GAS * (temp_module[i] + 273.15))))
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


def back_encapsulant(
    rh_ambient,
    temp_ambient,
    temp_module,
    WVTRo=7970633554,
    EaWVTR=55.0255,
    So=1.81390702,
    back_encap_thickness=0.5,
    Eas=16.729,
):
    """Return the relative humidity of backside module encapsulant.

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
    back_encapsulant : pandas series (float)
        Relative Humidity of backside solar module encapsulant [%]
    """
    rh_surface = surface_relative(
        rh_ambient=rh_ambient, temp_ambient=temp_ambient, temp_module=temp_module
    )

    Csat = csat(temp_module=temp_module, So=So, Eas=Eas)
    Ceq = ceq(Csat=Csat, rh_SurfaceOutside=rh_surface)

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

    back_encapsulant = 100 * (Ce_nparray / Csat)

    return back_encapsulant


def backsheet_from_encap(rh_back_encap, rh_surface_outside):
    """Calculate the Relative Humidity of solar module backsheet as timeseries.

    Requires the relative humidity of the backside encapsulant and the outside surface
    of the module.

    Parameters
    ----------
    rh_back_encap : pandas series (float)
        Relative Humidity of Frontside Solar module Encapsulant. *See rh_back_encap()
    rh_surface_outside : pandas series (float)
        The relative humidity of the surface of a solar module.
        *See surface_relative()

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
    backsheet_thickness=None,
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
    backsheet_thickness : float
        Thickness of the backsheet [mm].
        The suggested value for a PET backsheet is t=0.3 [mm]
    So_e : float
        Encapsulant solubility prefactor in [g/cm³]
        So = 1.81390702[g/cm³] is the suggested value for EVA W001.
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
    surface = surface_relative(
        rh_ambient=rh_ambient, temp_ambient=temp_ambient, temp_module=temp_module
    )

    # Get the relative humidity of the back encapsulant
    back_encapsulant = back_encapsulant_water_concentration(
        rh_surface=surface,
        # temp_ambient=temp_ambient,
        # rh_ambient=rh_ambient,
        temp_module=temp_module,
        start=start,
        Po_b=Po_b,
        Ea_p_b=Ea_p_b,
        backsheet_thickness=backsheet_thickness,
        So_e=So_e,
        Ea_s_e=Ea_s_e,
        back_encap_thickness=back_encap_thickness,
        backsheet=backsheet,
        encapsulant=encapsulant,
        output="rh",
    )

    return (back_encapsulant + surface) / 2


@decorators.geospatial_quick_shape(
    "timeseries",
    ["RH_surface_outside", "RH_front_encap", "RH_back_encap", "RH_backsheet"],
)
def module(
    weather_df=None,
    meta=None,
    poa=None,
    temp_module=None,
    tilt=None,
    azimuth=180,
    sky_model="isotropic",
    temp_model="sapm",
    conf="open_rack_glass_glass",
    wind_factor=0.33,
    Po_b=None,
    Ea_p_b=None,
    backsheet_thickness=None,
    So_e=None,
    Ea_s_e=None,
    Ea_d_e=None,
    back_encap_thickness=None,
    backsheet="W017",
    encapsulant="W001",
    **weather_kwargs,
):
    """Calculate the Relative Humidity of solar module backsheet from timeseries data.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    poa : pd.Series, optional
        Plane of array irradiance [W/m²]. If not provided, it will be calculated
    temp_module : pd.Series, optional
        Module temperature [°C]. If not provided, it will be calculated.
    tilt : float, optional
        Tilt angle of PV system relative to horizontal.
    azimuth : float, optional
        Azimuth angle of PV system relative to north.
    sky_model : str, optional
        Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'.
    temp_model : str, optional
        Options: 'sapm', 'pvsyst', 'faiman', 'sandia'.
    wind_factor : float, optional
        Wind speed correction exponent to account for different wind speed measurement
        heights between weather database (e.g. NSRDB) and the tempeature model
        (e.g. SAPM). The NSRDB provides calculations at 2 m (i.e module height) but SAPM
        uses a 10m height. It is recommended that a power-law relationship between
        height and wind speed of 0.33 be used*. This results in a wind speed that is
        1.7 times higher. It is acknowledged that this can vary significantly.
    Po_b : float
        Water permeation rate prefactor [g·mm/m²/day].
        The suggested value for PET W17 is Po = 1319534666.90318 [g·mm/m²/day].
    Ea_p_b : float
        Backsheet permeation  activation energy [kJ/mol].
    t : float
        Thickness of the backsheet [mm].
        The suggested value for a PET backsheet is 0.3mm.
    So_e : float
        Encapsulant solubility prefactor in [g/cm³]
    Ea_s_e : float
        Encapsulant solubility activation energy in [kJ/mol]
    Ea_d_e : float
        Encapsulant diffusivity activation energy in [kJ/mol]
    back_encap_thickness : float
        Thickness of the backside encapsulant [mm].
        The suggested value for EVA encapsulant  is 0.46mm.
    backsheet : str
        This is the code number for the backsheet.
        The default is PET 'W017'.
    encapsulant : str
        This is the code number for the encapsulant.
        The default is EVA 'W001'.
    **weather_kwargs : keyword arguments
        Additional keyword arguments passed to the weather data reader.

    Returns
    --------
    rh_surface_outside : float pandas dataframe
        relative humidity of the PV module surface as a time-series,
    rh_front_encap: float pandas dataframe
        relative humidity of the PV frontside encapsulant as a time-series,
    rh_back_encap : float pandas dataframe
        relative humidity of the PV backside encapsulant as a time-series,
    Ce_back_encap : float pandas dataframe
        concentration of water in the PV backside encapsulant as a time-series,
    rh_backsheet : float pandas dataframe
        relative humidity of the PV backsheet as a time-series
    """
    # solar_position = spectral.solar_position(weather_df, meta)
    # poa = spectral.poa_irradiance(weather_df, meta, solar_position, tilt, azimuth,
    # sky_model)
    # temp_module = temperature.module(weather_df, poa, temp_model, mount_type,
    # wind_factor)

    if poa is None:
        poa = spectral.poa_irradiance(
            weather_df=weather_df,
            meta=meta,
            tilt=tilt,
            azimuth=azimuth,
            sky_model=sky_model,
            **weather_kwargs,
        )

    if temp_module is None:
        temp_module = temperature.module(
            weather_df=weather_df,
            meta=meta,
            poa=poa,
            temp_model=temp_model,
            conf=conf,
            wind_factor=wind_factor,
            **weather_kwargs,
        )

    rh_surface_outside = surface_relative(
        rh_ambient=weather_df["relative_humidity"],
        temp_ambient=weather_df["temp_air"],
        temp_module=temp_module,
    )

    rh_front_encap = front_encapsulant(
        rh_ambient=weather_df["relative_humidity"],
        temp_ambient=weather_df["temp_air"],
        temp_module=temp_module,
        So=So_e,
        Eas=Ea_s_e,
        Ead=Ea_d_e,
        encapsulant=encapsulant,
    )

    rh_back_encap = back_encapsulant_water_concentration(
        temp_module=temp_module,
        rh_surface=None,
        rh_ambient=weather_df["relative_humidity"],
        temp_ambient=weather_df["temp_air"],
        Po_b=Po_b,
        Ea_p_b=Ea_p_b,
        backsheet_thickness=backsheet_thickness,
        So_e=So_e,
        Ea_s_e=Ea_s_e,
        back_encap_thickness=back_encap_thickness,
        backsheet=backsheet,
        encapsulant=encapsulant,
        output="rh",
    )

    Ce_back_encap = back_encapsulant_water_concentration(
        rh_ambient=weather_df["relative_humidity"],
        temp_ambient=weather_df["temp_air"],
        temp_module=temp_module,
        Po_b=Po_b,
        Ea_p_b=Ea_p_b,
        backsheet_thickness=backsheet_thickness,
        So_e=So_e,
        Ea_s_e=Ea_s_e,
        back_encap_thickness=back_encap_thickness,
        backsheet=backsheet,
        encapsulant=encapsulant,
        output="Ce",
    )

    rh_backsheet = (rh_back_encap + rh_surface_outside) / 2

    data = {
        "RH_surface_outside": rh_surface_outside,
        "RH_front_encap": rh_front_encap,
        "RH_back_encap": rh_back_encap,
        "Ce_back_encap": Ce_back_encap,
        "RH_backsheet": rh_backsheet,
    }
    results = pd.DataFrame(data=data)
    return results
