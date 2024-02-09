.. _pv-variables-terms:

PV Variables and Terms
======================

pvdeg aims to use the following variables and parameters in its codebase and documentation whenever possible. Please use these conventions when contributing.


Stressor Parameters
-------------------
.. list-table::
    :widths: 20 50 15 
    :header-rows: 1

    * - Input Variable
      - Description
      - Units
    * - T_k
      - temperature
      - [K]
    * - T
      - temperature
      - [C]
    * - G
      - Irradiance
      - [W/m^2]
    * - G_UV
      - UV Irradiance
      - [W/m^2`]
    * - G_340
      - UV Irradiance
      - W/m^2/nm
    * - G_pyr
      - W/m^2
      - UV Irradiance
    * - G_550
      - UV Irradiance
      - Photons*m^-2*s^-1
    * - TOW
      - Time of Wetness
      - [h/{year}]
    * - RH
      - Relative Humidity
      - [%]
    * - FF_0
      - Initial Fill Factor
      - [%]
    * - L
      - Lambda (Wavelength)
      - [mm]
    * - Q
      - Quantum Yield
      -
    * - BPT_K
      - Black Pannel Temperature
      - [K]
    


Modeling Constants
------------------
.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - Variable
      - Definition
    * - R_0
      - Frequency factor, prefactor, 
    * - R_D	
      - Rate of degradation
    * - E_a
      - Activation Energy
    * - t_fail
      - embrittlement time
    * - A 
      - prefactor
    * - FF 
      - Fill Factor
    * - B
      - Beta
    * - E 
      - Epsilon
    * - v_ab
      - LeTID prefactor, attempt frequency from state A to B
    * - v_bc
      - LeTID prefactor, attempt frequency from state A to B
    * - v_ba
      - LeTID prefactor, attempt frequency from state A to B
    * - v_cb
      - LeTID prefactor, attempt frequency from state A to B
    * - E_(a, ab)
      - LeTID Activation energy from state A to B
    * - E_(a, bc)
      - LeTID Activation energy from state B to C
    * - E_(a, ba)
      - LeTID Activation energy from state B to A
    * - E_(a, cb) 
      - LeTID Activation energy from state C to B
    * - x_ab
      - LeTID Excess Carrier Density Exponent
    * - x_bc
      - LeTID Excess Carrier Density Exponent
    * - x_ba
      - LeTID Excess Carrier Density Exponent
    * - A_T
      - Temperature prefactor
    * - A_UV
      - UV prefactor
    * - A_RH
      - RH prefactor
    * - LE
      - Life Expectation
    * - da
      - E of lambda
