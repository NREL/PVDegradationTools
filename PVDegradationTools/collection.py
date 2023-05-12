"""Collection of functions related to calculating current collection in solar cells
"""

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.constants import elementary_charge
import photovoltaic as pv

def collection_probability(x, thickness, s, l, d):
    """
    Returns the collection probability (unit 0 to 1) at a distance x (cm) from the junction.
    See [1]_.
    
    Parameters
    ----------
    x : array-like
        array of x positions from a junction to a surface (typically [cm]).
    
    thickness : numeric 
        Layer thickness [cm].
        
    s : numeric
        Surface recombination velocity [cm/s].
        
    l : numeric
        Minority carrier diffusion length [cm].
    
    d : numeric
        Minority carrier diffusivity [cm^2/Vs].
    
    Returns
    -------
    cp : array-like
        Collection probability along x.
    
    References
    ----------
    .. [1] https://www.pveducation.org/pvcdrom/solar-cell-operation/collection-probability
    """
    
    cosh_xl = np.cosh(x/l)
    cosh_wl = np.cosh(thickness/l)
    sinh_xl = np.sinh(x/l)
    sinh_wl = np.sinh(thickness/l)
    
    num = s*l/d*cosh_wl + sinh_wl
    den = s*l/d*sinh_wl + cosh_wl
    
    cp = cosh_xl - (num/den)*sinh_xl
    
    return cp

def calculate_jsc_from_tau_cp(tau, wafer_thickness, d_base, s_rear, generation, depth,
                              w_emitter = .36, l_emitter = 0.5, d_emitter = 4, s_emitter = 4.86e5,
                              xp = 0.00000024):
    '''
    Returns cell Jsc given lifetime and cell parameters

    Jsc is calculated via integrating collection probability and optical generation profiles
    through the wafer depth. Includes contribution of the emitter, depletion region (assumes
    CP = 1 in the depletion region), and base.
    
    Parameters
    ----------
    tau : numeric
        Carrier lifetime [us].
        
    wafer_thickness : numeric
        Wafer thickness [um].
        
    d_base : numeric
        Minority carrier diffusivity [cm^2/Vs]. 
        
    s_rear : numeric
        Rear surface recombination velocity [cm/s].
        
    generation : array-like
        array of generation current G(z), e.g. from PVlighthouse OPAL2 [cm^3/s] [1]_.
        
    depth : array-like
        array of wafer depth 0-z [um].

    w_emitter : numeric, default 0.36
        Emitter thickness [um].

    l_emitter : numeric, default 0.5
        Minority carrier diffusion length of the emitter [um].
    
    d_emitter : numeric, default 4
        Minority carrier diffusivity in the emitter [cm^2/s].
    
    s_emitter : numeric, default 4.86e5
        Front surface recombination velocity [cm/s].
    
    xp : numeric, default 0.00000024
        width of the depletion region [m]. Treated as fixed width, as it is very small compared
        to the bulk, so injection-dependent variations will have very small effects.
    
    Returns
    -------
    jsc : numeric
        Short circuit current of the solar cell [mA/cm^2].
        
    Notes
    -----
    Emitter parameters w, l, d, and s are 

    Pros of this approach
        can account for e.g. lighttrapping via the generation profile
        
    Cons of this approach
        requires a generation profile
    
    Future work:
        include default generation and depth profile.
   
    References
    ----------
    .. [1] K.R. McIntosh and S.C. Baker-Finch, "OPAL 2: Rapid optical simulation of silicon solar
    cells," Proceedings of the 38th IEEE Photovoltaic Specialists Conference,Austin, in-press, 2012.
    '''
    q = elementary_charge

    # Unit conversions:
    w_depletion = xp*100 #depletion region width, 240 nm in cm

    #w_emitter = .36e-4 #emitter thickness, .36 um in cm
    w_emitter = w_emitter*1e-4 #um to cm
    #l_emitter = 0.5e-4 #diffusion length of the emitter, 0.5 um in cm
    l_emitter = l_emitter*1e-4
    #d_emitter = 4 #diffusivity of the emitter, 4 cm^2/s
    #s_emitter = 4.86e5 #front srv, 4.86e5 cm/s
    #s_emitter = 150 #front srv, 4.86e5 cm/s

    #x_j = w_emitter + w_depletion #junction depth in cm
    w_base = wafer_thickness*1e-4 - w_emitter #width of the base in cm
    #d_base = 27 #diffusivity in cm^2/s
    tau = tau*1e-6 #us to s
    s_base = s_rear #cm/s
    depth = depth*1e-4 #um to cm

    # 1. calc diffusion length
    l_base = np.sqrt(tau*d_base) #diffusion length in cm

    # 2. set up depth profile and cell regions
    index_emitter = np.searchsorted(depth, w_emitter)
    emitter_depth = depth[0:index_emitter]
    emitter_depth = np.flip((-emitter_depth+np.max(emitter_depth)).to_numpy())
    index_depletion = np.searchsorted(depth, w_emitter+w_depletion)
    depletion_depth = depth[index_emitter:index_depletion]
    base_depth = depth[index_depletion:]

    # 3. calc c.p. for emitter, depletion region, and base
    cp_emitter = collection_probability(emitter_depth, thickness = w_emitter, s = s_emitter,
                                        l = l_emitter, d = d_emitter)
    cp_emitter = np.flip(cp_emitter)
    emitter_depth = np.flip((-emitter_depth+np.max(emitter_depth)))

    cp_depletion = np.ones(len(depletion_depth))

    cp_base = collection_probability(base_depth, w_base, s_base, l_base, d_base)

    depth_array = np.concatenate(np.array([emitter_depth, depletion_depth, base_depth],
                                          dtype = object), axis = 0)
    collection_array = np.concatenate(np.array([cp_emitter, cp_depletion, cp_base], dtype = object),
                                      axis = 0)

    # 4. Reinterpolate to match the generation depth profile
    f = interp1d(depth_array, collection_array, kind='cubic')
    collection_array_interp = f(depth)

    # 5. integrate
    jsc = simpson(collection_array_interp*generation, depth)*q

    return jsc*1000

def calculate_jsc_from_tau_iqe(tau, wafer_thickness, d_base, s_rear, spectrum, absorption,
                               wavelengths, w_emitter = .36, l_emitter = 0.5, d_emitter = 4,
                               s_emitter = 4.86e5, xp = 0.00000024):
    """
    Returns cell Jsc given lifetime and cell parameters
    
    Calculates Jsc via calculating cell internal quantum efficiency (IQE) and absorption for Si
    
    Parameters
    ----------
    tau : numeric
        Carrier lifetime [us].
        
    wafer_thickness : numeric
        Wafer thickness [um].
        
    d_base : numeric
        Minority carrier diffusivity [cm^2/Vs]. 
        
    s_rear : numeric
        Rear surface recombination velocity [cm/s].
    
    spectrum : array-like
        photon flux density solar spectrum.
    
    absorption : array-like
        absorption coefficient of Si, e.g. loaded from 'photovoltaic' library [1]_.
        
    wavelengths : array-like
        wavelength series (nm) for absorption coefficient and solar spectrum.

    w_emitter : numeric, default 0.36
        Emitter thickness [um].

    l_emitter : numeric, default 0.5
        Diffusion length of the emitter [um].
    
    d_emitter : numeric, default 4
        Minority carrier diffusivity in the emitter [cm^2/s].
    
    s_emitter : numeric, default 4.86e5
        Front surface recombination velocity [cm/s].
    
    xp : numeric, default 0.00000024
        width of the depletion region [m]. Treated as fixed width, as it is very small compared
        to the bulk, so injection-dependent variations will have very small effects.

    Returns
    -------
    jsc : numeric
        Short circuit current of the solar cell [mA/cm^2]
    
    Notes
    -----
    Pros of this approach:
        requires only fundamental inputs: Si absorption and spectrum 
        
    Cons:
        does not account for any anti-reflection or light trapping
        
    To do:
        accept user inputs for spectrum and absorption, otherwise use photovoltaic library
        allow user to customize hard-coded assuptions for cell properties below.
        
    References
    ----------
    .. [1] “photovoltaic.” https://github.com/pvedu/photovoltaic.

    """
    # Unit conversions:
    w_depletion = xp*100 #depletion region width, 240 nm in cm

    #w_emitter = .36e-4 #emitter thickness, .36 um in cm
    w_emitter = w_emitter*1e-4
    #l_emitter = 0.5e-4 #diffusion length of the emitter, 0.5 um in cm
    l_emitter = l_emitter*1e-4
    #d_emitter = 4 #diffusivity of the emitter, 4 cm^2/s
    #s_emitter = 4.86e5 #front srv, 4.86e5 cm/s
    #s_emitter = 150 #front srv, 4.86e5 cm/s

    #x_j = w_emitter + w_depletion #junction depth in cm
    w_base = wafer_thickness*1e-4 - w_emitter #width of the base in cm
    #d_base = 27 #diffusivity in cm^2/s
    tau = tau*1e-6 #us to s
    s_base = s_rear #cm/s

    # 1. calculate diffusion length
    l_base = np.sqrt(tau*d_base) #diffusion length in cm

    # 2. calculate iqe
    _, _, _, qet = pv.cell.iqe(absorption, w_depletion, s_emitter, l_emitter, d_emitter,
                                     w_emitter, s_base, w_base, l_base, d_base)

    # 3. integrate iqe with spectrum
    jsc = simpson(qet*spectrum, wavelengths)/10
    return jsc

def generation_current(generation, depth):
    """
    Returns cell generation current given generation profile
        
    Parameters
    ----------
    generation : array-like
        array of generation current G(z), e.g. from PVlighthouse OPAL2 [cm^3/s] [1]_.
        
    depth : array-like
        array of wafer depth 0-z [um].

    Returns
    -------
    j_gen : numeric
        generation current [mA/cm^2]
    """
    q = elementary_charge

    j_gen = simpson(generation, depth)*q/10
    return j_gen
