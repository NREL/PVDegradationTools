"""Collection of functions to calculate LETID or B-O LID defect states, defect state transitions,
and device degradation given device details
"""
import numpy as np
import pandas as pd
import os
from scipy.constants import convert_temperature, elementary_charge, Boltzmann
from scipy.integrate import simpson
from pvdeg import collection


def tau_now(tau_0, tau_deg, n_b):
    """
    Return carrier lifetime of a LID or LETID-degraded wafer given initial lifetime, fully
    degraded lifetime, and fraction of defects in recombination-active state B

    Parameters
    ----------
    tau_0 : numeric
        Initial lifetime. Typcially in seconds, milliseconds, or microseconds.
    tau_deg : numeric
        Lifetime when wafer is fully-degraded, i.e. 100% of defects are in state B. Same units
        as tau_0.
    n_B : numeric
        Percentage of defects in state B [%].

    Returns
    -------
    numeric
        lifetime of wafer with n_B% of defects in state B. Same units as tau_0 and tau_deg.
    """
    return tau_0 / ((tau_0 / tau_deg - 1) * n_b / 100 + 1)

def k_ij(attempt_frequency, activation_energy, temperature):
    """
    Calculates an Arrhenius rate constant given attempt frequency, activation energy, and
    temperature

    Parameters
    ----------
    attempt_frequency : numeric
        Arrhenius pre-exponential factor (or attempt frequency) [s^-1].

    activation_energy : numeric
        Arrhenius activation energy [eV].

    temperature : numeric
        Temperature [C].

    Returns
    -------
    reaction_rate : numeric
        Arrhenius reaction rate constant [s^-1].
    """
    temperature = convert_temperature(temperature, 'C', 'K')
    q = elementary_charge
    k = Boltzmann

    reaction_rate = attempt_frequency * np.exp(
        (-activation_energy * q) / (k * temperature)
    )
    return reaction_rate

def carrier_factor(
    tau,
    transition,
    temperature,
    suns,
    jsc,
    wafer_thickness,
    s_rear,
    mechanism_params,
    dn_lit=None,
):
    """
    Return the delta_n^x_ij term to modify attempt frequency by excess carrier density. See
    McPherson 2022 [1]_. Requires mechanism_params, a dict of required mechanism parameters.
    See 'MECHANISM_PARAMS' dict

    Parameters
    ----------
    tau : numeric
        Carrier lifetime [us].

    transition : str
        Transition in the 3-state defect model (A <-> B <-> C). Must be 'ab', 'bc', or 'ba'.

    temperature : numeric
        Temperature [C].

    suns : numeric
        Applied injection level of device in terms of "suns", e.g. 1 for a device held at 1-sun
        Jsc current injection in the dark, or at open-circuit with 1-sun illumination.

    jsc : numeric
        Short-circuit current density [mA/cm^2].

    wafer_thickness : numeric
        Wafer thickness [um].

    s_rear : numeric
        Rear surface recombination velocity [cm/s].

    mechanism_params : dict
        Dictionary of mechanism parameters.
        These are typically taken from literature studies of transtions in the 3-state model.
        They allow for calculation the excess carrier density of literature experiments (dn_lit)
        Parameters are coded in 'MECHANISM_PARAMS' dict.

    dn_lit : numeric, default None
        Optional, supply in lieu of a complete set of mechanism_params if experimental dn_lit
        is known.

    Returns
    -------
    numeric
        dn^x_ij term. Modified by the ratio of modeled dn to literature experiment
        dn (dn/dn_lit).

    References
    ----------
    .. [1] A. N. McPherson, J. F. Karas, D. L. Young, and I. L. Repins, 
    “Excess carrier concentration in silicon devices and wafers: How bulk properties are
    expected to accelerate light and elevated temperature degradation,” 
    MRS Advances, vol. 7, pp. 438–443, 2022, doi: 10.1557/s43580-022-00222-5.

    """
    q = elementary_charge

    if transition == "ab":
        exponent = mechanism_params[f"x_{transition}"]
        if dn_lit is None:
            meas_tau = mechanism_params[f"tau_{transition}"]
            meas_temp = mechanism_params[f"temperature_{transition}"]
            meas_temp = convert_temperature(meas_temp, 'K', 'C')  # convert Kelvin to Celsius
            meas_suns = mechanism_params[f"suns_{transition}"]
            meas_jsc = 40
            meas_wafer_thickness = mechanism_params[f"thickness_{transition}"]
            meas_srv = mechanism_params[f"srv_{transition}"]
            meas_structure = mechanism_params[f"structure_{transition}"]

            if meas_structure == "wafer":
                # unit conversions on jsc, tau, and wafer thickness:
                dn_lit = (
                    ((jsc * 0.001 * 10000 * meas_suns) * (meas_tau * 1e-6))
                    / (meas_wafer_thickness * 1e-6)
                    / q
                )

            else:
                dn_lit = calc_dn(
                    meas_tau,
                    meas_temp,
                    meas_suns,
                    meas_jsc,
                    wafer_thickness=meas_wafer_thickness,
                    s_rear=meas_srv,
                )

    elif transition == "bc":
        exponent = mechanism_params[f"x_{transition}"]
        if dn_lit is None:
            meas_tau = mechanism_params[f"tau_{transition}"]
            meas_temp = mechanism_params[f"temperature_{transition}"]
            meas_temp = convert_temperature(meas_temp, 'K', 'C')  # convert Kelvin to Celsius
            meas_suns = mechanism_params[f"suns_{transition}"]
            meas_jsc = 40
            meas_wafer_thickness = mechanism_params[f"thickness_{transition}"]
            meas_srv = mechanism_params[f"srv_{transition}"]
            meas_structure = mechanism_params[f"structure_{transition}"]

            if meas_structure == "wafer":
                # unit conversions on jsc, tau, and wafer thickness:
                dn_lit = (
                    ((jsc * 0.001 * 10000 * meas_suns) * (meas_tau * 1e-6))
                    / (meas_wafer_thickness * 1e-6)
                    / q
                )

            else:
                dn_lit = calc_dn(
                    meas_tau,
                    meas_temp,
                    meas_suns,
                    meas_jsc,
                    wafer_thickness=meas_wafer_thickness,
                    s_rear=meas_srv,
                )

    elif transition == "ba":
        exponent = mechanism_params[f"x_{transition}"]
        if dn_lit is None:
            dn_lit = 1e21  # this is hardcoded

    else:
        exponent = 0
        # or we could raise a ValueError and say transition has to be 'ab'|'bc'|'ba'
        dn_lit = 1e21

    dn = calc_dn(tau, temperature, suns, jsc, wafer_thickness, s_rear)

    return (dn / dn_lit) ** exponent

def carrier_factor_wafer(
    tau,
    transition,
    suns,
    jsc,
    wafer_thickness,
    mechanism_params,
    dn_lit=None
    ):
    r"""
    Return the delta_n^x_ij term to modify attempt frequency by excess carrier density for a
    passivated wafer, rather than a solar cell.
    
    For a passivated wafer, delta_n increases linearly with lifetime:

    .. math::
        \Delta n = \tau*J/W

    See McPherson 2022 [1]_. Requires mechanism_params, a dict of required mechanism parameters.
    See 'MECHANISM_PARAMS' dict

    Parameters
    ----------
    tau : numeric
        Carrier lifetime [us].

    transition : str
        Transition in the 3-state defect model (A <-> B <-> C). Must be 'ab', 'bc', or 'ba'.

    suns : numeric
        Applied injection level of device in terms of "suns", e.g. 1 for a device held at 1-sun
        Jsc current injection in the dark, or at open-circuit with 1-sun illumination.

    jsc : numeric
        Short-circuit current density [mA/cm^2].

    wafer_thickness : numeric
        Wafer thickness [um].

    mechanism_params : dict
        Dictionary of mechanism parameters.
        These are typically taken from literature studies of transtions in the 3-state model.
        They allow for calculation the excess carrier density of literature experiments (dn_lit)
        Parameters are coded in 'MECHANISM_PARAMS' dict.

    dn_lit : numeric, default None
        Optional, supply in lieu of a complete set of mechanism_params if experimental dn_lit
        is known.

    Returns
    -------
    numeric
        dn^x_ij term. Modified by the ratio of modeled dn to literature experiment dn
        (dn/dn_lit).

    References
    ----------
    .. [1] A. N. McPherson, J. F. Karas, D. L. Young, and I. L. Repins, 
    “Excess carrier concentration in silicon devices and wafers: How bulk properties are
    expected to accelerate light and elevated temperature degradation,” 
    MRS Advances, vol. 7, pp. 438–443, 2022, doi: 10.1557/s43580-022-00222-5.

    """
    q = elementary_charge

    if transition == "ab":
        exponent = mechanism_params[f"x_{transition}"]
        if dn_lit is None:
            meas_tau = mechanism_params[f"tau_{transition}"]
            meas_temp = mechanism_params[f"temperature_{transition}"]
            meas_temp = convert_temperature(meas_temp, 'K', 'C')  # convert Kelvin to Celsius
            meas_suns = mechanism_params[f"suns_{transition}"]
            meas_jsc = 40
            meas_wafer_thickness = mechanism_params[f"thickness_{transition}"]
            meas_srv = mechanism_params[f"srv_{transition}"]
            meas_structure = mechanism_params[f"structure_{transition}"]

            if meas_structure == "wafer":
                # unit conversions on jsc, tau, and wafer thickness:
                dn_lit = (
                    ((jsc * 0.001 * 10000 * meas_suns) * (meas_tau * 1e-6))
                    / (meas_wafer_thickness * 1e-6)
                    / q
                )

            else:
                dn_lit = calc_dn(
                    meas_tau,
                    meas_temp,
                    meas_suns,
                    meas_jsc,
                    wafer_thickness=meas_wafer_thickness,
                    s_rear=meas_srv,
                )

    elif transition == "bc":
        exponent = mechanism_params[f"x_{transition}"]
        if dn_lit is None:
            meas_tau = mechanism_params[f"tau_{transition}"]
            meas_temp = mechanism_params[f"temperature_{transition}"]
            meas_temp = convert_temperature(meas_temp, 'K', 'C')  # convert Kelvin to Celsius
            meas_suns = mechanism_params[f"suns_{transition}"]
            meas_jsc = 40
            meas_wafer_thickness = mechanism_params[f"thickness_{transition}"]
            meas_srv = mechanism_params[f"srv_{transition}"]
            meas_structure = mechanism_params[f"structure_{transition}"]

            if meas_structure == "wafer":
                # unit conversions on jsc, tau, and wafer thickness:
                dn_lit = (
                    ((jsc * 0.001 * 10000 * meas_suns) * (meas_tau * 1e-6))
                    / (meas_wafer_thickness * 1e-6)
                    / q
                )

            else:
                dn_lit = calc_dn(
                    meas_tau,
                    meas_temp,
                    meas_suns,
                    meas_jsc,
                    wafer_thickness=meas_wafer_thickness,
                    s_rear=meas_srv,
                )

    elif transition == "ba":
        exponent = mechanism_params[f"x_{transition}"]
        if dn_lit is None:
            dn_lit = 1e21  # this is hardcoded

    else:
        exponent = 0  # or we could raise a ValueError and say transition has to be 'ab'|'bc'|'ba'
        dn_lit = 1e21

    dn = (((jsc * 0.001 * 10000 * suns) * (tau * 1e-6))/ (wafer_thickness * 1e-6)/ q)

    return (dn/dn_lit)**exponent

def calc_dn(
    tau, temperature, suns, jsc, wafer_thickness, s_rear, na=7.2e21, xp=0.00000024,
    e_mobility=0.15, nc=2.8e25, nv=1.6e25, e_g=1.79444e-19
):
    """
    Return excess carrier concentration, i.e. "injection", given lifetime, temperature,
    suns-equivalent applied injection, and cell parameters

    Parameters
    ----------
    tau : numeric
        Carrier lifetime [us].

    temperature : numeric
        Cell temperature [K].

    suns : numeric
        Applied injection level of device in terms of "suns",
        e.g. 1 for a device held at 1-sun Jsc current injection.

    jsc : numeric
        Short-circuit current density of the cell [mA/cm^2].

    wafer_thickness : numeric
        Wafer thickness [um].

    s_rear : numeric
        Rear surface recombination velocity [cm/s].

    na : numeric, default 7.2e21
        Doping density [m^-3].

    xp : numeric, default 0.00000024
        width of the depletion region [m]. Treated as fixed width, as it is very small compared
        to the bulk, so injection-dependent variations will have very small effects.

    e_mobility : numeric, default 0.15
        electron mobility [m^2/V-s].

    nc : numeric, default 2.8e25
        density of states of the conduction band [m^-3]

    nv : numeric, default 1.6e25
        density of states of the valence band [m^-3]

    e_g : numeric, default 1.79444e-19
        bandgap of silicon [J].

    Returns
    -------
    dn : numeric
        excess carrier concentration [m^-3]

    """
    k = Boltzmann
    q = elementary_charge

    # unit conversions
    tau = tau * 1e-6  # convert microseconds to seconds
    temperature = convert_temperature(temperature, 'C', 'K')  # convert Celsius to Kelvin
    jsc = jsc * 0.001 * 10000  # convert mA/cm^2 to A/m^2
    wafer_thickness = wafer_thickness * 1e-6  # convert microns to meters
    s_rear = s_rear / 100  # convert cm/s to m/s

    i_applied = suns * jsc
    v_applied = convert_i_to_v(
        tau, na, i_applied, wafer_thickness, s_rear, temperature
    )

    diffusivity = e_mobility * k * temperature / q
    ni2 = nc * nv * np.exp(-e_g / (k * temperature))  # ni^2 = Nc*Nv*exp(-Eg/kT)

    arg = (wafer_thickness - xp) / (diffusivity * tau) ** 0.5

    exp_prefactor = ni2 / na * np.exp(q * v_applied / (k * temperature))

    cosh = np.cosh(arg)
    sinh = np.sinh(arg)

    numerator = (s_rear / diffusivity) * cosh + sinh * ((diffusivity * tau) ** (-0.5))
    denominator = cosh * ((diffusivity * tau) ** (-0.5)) + (s_rear / diffusivity) * sinh

    a_p = -exp_prefactor * numerator / denominator
    b_p = exp_prefactor

    dn = (((diffusivity * tau) ** (0.5)) / (wafer_thickness - xp)) * (
        a_p * (cosh - 1) + b_p * sinh
    )

    return dn

def convert_i_to_v(tau, na, current, wafer_thickness, srv, temperature=298.15, e_mobility=0.15,
                    xp=0.00000024, nc=2.8e25, nv=1.6e25, e_g=1.79444e-19):
    """
    Return voltage given lifetime and applied current, and cell parameters

    Parameters
    ----------
    tau : numeric
        Carrier lifetime [s].

    na : numeric
        Doping density [m^-3].

    current : numeric
        applied current [A].

    wafer_thickness : numeric
        Wafer thickness [m].

    srv : numeric
        Surface recombination velocity [m/s].

    temperature : numeric, default 298.15
        Cell temperature [K]
    
    e_mobility : numeric, default 0.15
        electron mobility [m^2/V-s].

    xp : numeric, default 0.00000024
        width of the depletion region [m]. Treated as fixed width, as it is very small compared
        to the bulk, so injection-dependent variations will have very small effects.
    
    nc : numeric, default 2.8e25
        density of states of the conduction band [m^-3]

    nv : numeric, default 1.6e25
        density of states of the valence band [m^-3]

    e_g : numeric, default 1.79444e-19
        bandgap of silicon [J].

    Returns
    -------
    voltage : numeric
        cell voltage [V]
    """
    k = Boltzmann
    q = elementary_charge

    diffusivity = e_mobility * k * temperature / q
    diffusion_length = np.sqrt(diffusivity * tau)
    ni2 = nc * nv * np.exp(-e_g / (k * temperature))  # ni^2 = Nc*Nv*exp(-Eg/kT)
    arg = (wafer_thickness - xp) / diffusion_length

    j0 = j0_gray(ni2, diffusivity, na, diffusion_length, arg, srv)

    if current > 0:
        voltage = (k * temperature / q) * np.log(current / j0)
    else:
        voltage = 0
    return voltage

def j0_gray(ni2, diffusivity, na, diffusion_length, arg, srv):
    """
    Returns j0 (saturation current density in quasi-neutral regions of a solar cell)
    as shown in eq. 3.128 in [1]_.

    Parameters
    ----------
    ni2 : numeric
        intrinsic carrier concentration [m^-3].

    diffusivity : numeric
        carrier diffusivity [m/s].

    na : numeric
        doping density [m^-3].

    diffusion_length : numeric
        carrier diffusion length [m].

    arg : numeric
        (W-xn)/Lp term, (wafer_thickness-depletion region thickness)/diffusion_length [unitless].

    srv : numeric
        surface recombination velocity [m/s].

    Returns
    -------
    numeric
        j0 [A]

    References
    ----------
    .. [1] J. L. Gray, “The Physics of the Solar Cell,”
    in Handbook of Photovoltaic Science and Engineering,
    A. Luque and S. Hegedus, Eds. Chichester, UK: John Wiley & Sons, Ltd,
    2011, pp. 82–129. doi: 10.1002/9780470974704.ch3.
    """
    q = elementary_charge

    prefactor = q * ni2 * diffusivity / (na * diffusion_length)
    numerator = (diffusivity / diffusion_length) * np.sinh(arg) + srv * np.cosh(arg)
    denominator = (diffusivity / diffusion_length) * np.cosh(arg) + srv * np.sinh(
        arg
    )
    return prefactor * (numerator / denominator)

def calc_voc_from_tau(tau, wafer_thickness, srv_rear, jsc, temperature, na=7.2e21):
    '''
    Return solar cell open-circuit voltage (Voc), given lifetime and other device parameters
    
    Parameters
    ----------
    
    tau : numeric
        Carrier lifetime [us].
        
    wafer_thickness : numeric
        Wafer thickness [um].
        
    srv_rear : numeric 
        Rear surface recombination velocity [cm/s].
        
    jsc : numeric
        Short-circuit current density [mA/cm^2].
        
    temperature : numeric 
        Temperature [C].
        
    na : numeric, default 7.2e21
        Doping density [m^-3]. Default value corresponds to ~2 Ω-cm boron-doped c-Si.
        
    Returns
    -------
    numeric 
        Device Voc [V].
    '''
    # unit conversions
    tau = tau*1e-6 # convert microseconds to seconds
    wafer_thickness = wafer_thickness*1e-6  # convert microns to meters    
    srv_rear = srv_rear/100 # convert cm/s to m/s
    jsc = jsc*.001*10000 # convert mA/cm^2 to A/m^2
    temperature = convert_temperature(temperature, 'C', 'K') # convert Celsius to Kelvin
    
    return(convert_i_to_v(tau, na, jsc, wafer_thickness, srv_rear, temperature))

def calc_device_params(timesteps, cell_area=239):
    """
    Returns device parameters given a Dataframe of Jsc and Voc

    Parameters
    ----------

    timesteps : Dataframe
        Column names must include:
            - ``'Jsc'``
            - ``'Voc'``

    cell_area : numeric, default 239
        Cell area [cm^2]. 239 cm^2 is roughly the area of a 156x156mm pseudosquare "M0" wafer

    Returns
    -------
    timesteps : Dataframe
        Dataframe with new columns for Isc, FF, Pmp, and normalized Pmp

    """
    timesteps.loc[:, "Isc"] = timesteps.loc[:, "Jsc"] * (cell_area / 1000)
    timesteps.loc[:, "FF"] = ff_green(timesteps.loc[:, "Voc"])
    timesteps.loc[:, "Pmp"] = (
        timesteps.loc[:, "Voc"]
        * timesteps.loc[:, "FF"]
        * timesteps.loc[:, "Jsc"]
        * (cell_area / 1000)
    )
    timesteps.loc[:, "Pmp_norm"] = timesteps.loc[:, "Pmp"] / timesteps.loc[0, "Pmp"]

    return timesteps

def calc_energy_loss(timesteps):
    """
    Returns energy loss given a timeseries containing normalized changes in maximum power 

    Parameters
    ----------

    timesteps : Dataframe
        timesteps.index must be DatetimeIndex OR timesteps must include ``'Datetime'``
        column with dtype datetime

        Column names must include:
            - ``'Pmp_norm'``, a column of normalized (0-1) maximum power such as returned by
            Degradation.calc_device_params

    Returns
    -------
    energy_loss : float
        fractional energy loss over time
    """
    if isinstance(timesteps.index, pd.DatetimeIndex):
        start = timesteps.index[0]
        timedelta = [(d - start).total_seconds()/3600 for d in timesteps.index]
    else:
        start = timesteps['Datetime'].iloc[0]
        timedelta = [(d - start).total_seconds()/3600 for d in timesteps['Datetime']]
    
    pmp_norm = timesteps['Pmp_norm']
    energy_loss = 1-(simpson(pmp_norm, timedelta)/simpson(np.ones(len(pmp_norm)), timedelta))

    return energy_loss

def calc_regeneration_time(timesteps, x = 80, rtol = 1e-05):
    """
    Returns time to x% regeneration, determined by the percentage of defects in State C.

    Parameters
    ----------
    timesteps : Dataframe
        timesteps.index must be DatetimeIndex OR timesteps must include ``'Datetime'`` column
        with dtype datetime
        Column names must include:
            - ``'NC'``, the percentage of defects in state C
        
    x : numeric, default 80
        percentage regeneration to look for. Note that 100% State C will take a very long time,
        whereas in most cases >99% of power is regenerated after NC = ~80%

    rel_tol : float
        The relative tolerance parameter

    Returns
    -------
    regen_time : timedelta
        The time taken to reach x% regeneration
    """

    if isinstance(timesteps.index, pd.DatetimeIndex):
        start = timesteps.index[0]
        stop_row = timesteps[np.isclose(timesteps['NC'], x, rtol)].iloc[0]
        stop = stop_row.name

        regen_time = stop-start

    else:
        start = timesteps['Datetime'].iloc[0]
        stop_row = timesteps[np.isclose(timesteps['NC'], x, rtol)].iloc[0]
        stop = stop_row['Datetime']

        regen_time = stop-start

    return regen_time

def calc_pmp_loss_from_tau_loss(tau_0, tau_deg, cell_area, wafer_thickness, s_rear,
                                generation = None, depth = None):
    """
    Function to estimate power loss from bulk lifetime loss

    Parameters
    ----------
    tau_0 : numeric
        Initial bulk lifetime [us]

    tau_deg : numeric
        Degraded bulk lifetime [us]

    cell_area : numeric
        Cell area [cm^2]

    wafer_thickness : numeric
        Wafer thickness [um]

    s_rear : numeric
        Rear surface recombination velocity [cm/s]

    Returns
    -------
    pmp_loss, pmp_0, pmp_deg : tuple of numeric
        Power loss [%], Initial power [W], and Degraded power [W]
    """

    if generation is None or depth is None:
        absolute_path = os.path.dirname(__file__)
        relative_path = 'data/PVL_GenProfile.xlsx'
        path = os.path.join(absolute_path, relative_path)
        
        generation_df = pd.read_excel(path, header = 0,
                                    engine="openpyxl")
        generation = generation_df['Generation (cm-3s-1)']
        depth = generation_df['Depth (um)']

    jsc_0 = collection.calculate_jsc_from_tau_cp(
        tau_0,
        wafer_thickness=wafer_thickness,
        d_base=27,
        s_rear=s_rear,
        generation=generation,
        depth=depth,
    )
    jsc_deg = collection.calculate_jsc_from_tau_cp(
        tau_deg,
        wafer_thickness=wafer_thickness,
        d_base=27,
        s_rear=s_rear,
        generation=generation,
        depth=depth,
    )

    voc_0 = calc_voc_from_tau(
        tau_0, wafer_thickness, s_rear, jsc_0, temperature=25, na=7.2e21
    )
    voc_deg = calc_voc_from_tau(
        tau_deg, wafer_thickness, s_rear, jsc_deg, temperature=25, na=7.2e21
    )

    ff_0 = ff_green(voc_0)
    ff_deg = ff_green(voc_deg)

    pmp_0 = jsc_0 / 1000 * cell_area * voc_0 * ff_0
    pmp_deg = jsc_deg / 1000 * cell_area * voc_deg * ff_deg

    pmp_loss = (pmp_0 - pmp_deg) / pmp_0

    return pmp_loss, pmp_0, pmp_deg

def calc_ndd(tau_0, tau_deg):
    """
    Calculates normalized defect density given starting and ending lifetimes

    Parameters
    ----------
    tau_0 : numeric
        Initial bulk lifetime [us]

    tau_deg : numeric
        Degraded bulk lifetime [us]

    Returns
    -------
    ndd : numeric
        normalized defect density
    """
    ndd = (1/tau_deg) - (1/tau_0)
    return ndd

def ff_green(voltage, temperature=298.15):
    """
    Calculates the empirical expression for fill factor of Si cells from open-circuit voltage.

    See [1]_, equation 4.

    Parameters
    ----------
    voltage : numeric
        Open-circuit voltage of the solar cell [V].
    temperature : numeric, default 298.15
        Temperature of the solar cell [K].

    Returns
    -------
    numeric
        Fill factor of the solar cell

    References
    ----------
    .. [1] M. A. Green, “Solar cell fill factors: General graph and empirical expressions”,
    Solid-State Electronics, vol. 24, pp. 788 - 789, 1981.
    https://doi.org/10.1016/0038-1101(81)90062-9
    """
    k = Boltzmann
    q = elementary_charge

    v = voltage * q / (k * temperature)
    return (v - np.log(v + 0.72)) / (v + 1)
