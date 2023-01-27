"""
Contains classes for calculating Stress Factors, Degradation, and installation Standards for PV Modules.

"""

import numpy as np
from numba import jit
import pandas as pd
from scipy.constants import convert_temperature
import pvlib

class StressFactors:
    """
    EnergyCalcs class contains the Vant Hoff acceleration factor and Arrhenius
    Equations Acceleration Factor

    """

    def k(avg_wvp):
        """
        This function generates a constant k, relating the average moisture ingress rate through a specific edge seal, Helioseal 101.
        Is an emperical estimation the rate of water ingress of water through edge seal material.
        This function was determined from numerical calculations from several locations and thus produces typical responses.
        This simplification works because the environmental temperature is not as important as local water vapor pressure.
        For the same environmental water concentration, a higher temperature results in lower absorption in the edge seal but lower diffusivity through the edge seal. In practice, these effects nearly cancel out makeing absolute humidity the primary parameter determining moisture ingress through edge seals.
        See: Kempe, Nobles, Postak Calderon,"Moisture ingress prediction in polyisobutylene‐based edge seal with molecular sieve desiccant", Progress in Photovoltaics, DOI: 10.1002/pip.2947

        Parameters
        -----------
        avg_wvp : float
            Time averaged water vapor pressure for an environment in kPa. 
            When looking at outdoor data, one should average over 1 year

        Returns
        -------
        k : float [cm/h^0.5]
            Ingress rate of water through edge seal. 
            Specifically it is the ratio of the breakthrough distance X/t^0.5.
            With this constant, one can determine an approximate estimate of the ingress distance for a particular climate without more complicated numerical methods and detailed environmental analysis.

        """

        k = .0013 * (avg_wvp)**.4933

        return k

    def edge_seal_width(k):
        """
        Determine the width of edge seal required for a 25 year water ingress.

        Parameters
        ----------
        k: float 
            Ingress rate of water through edge seal. [cm/h^0.5]
            Specifically it is the ratio of the breakthrough distance X/t^0.5.

        Returns
        ----------
        width : float 
            Width of edge seal required for a 25 year water ingress. [cm]
        """

        width = k * (25 * 365.25 * 24)**.5

        return width


    # Numba Machine Language Level

    @jit(nopython=True, error_model='python')
    def dew_yield(elevation, dew_point, dry_bulb, wind_speed, n):
        """
        Estimates the dew yield in [mm/day].  Calculation taken from:
        Beysens, "Estimating dew yield worldwide from a few meteo data", Atmospheric Research 167 (2016) 146-155

        Parameters
        -----------
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
            This is a quasi emperical scale from 0 to 8 used in meterology which corresponds to 0-sky completely clear, to 8-sky completely cloudy. Does not account for cloud type or thickness.

        Returns
        -------
        dew_yield : float
            Amount of dew yield in [mm/day]

        """
        wind_speed_cut_off = 4.4
        dew_yield = (1/12) * (.37 * (1 + (0.204323 * h) - (0.0238893 * h**2) -
                             (18.0132 - (1.04963 * h**2) + (0.21891 * h**2)) * (10**(-3) * dew_point)) *
                             ((((dew_point + 273.15) / 285)**4)*(1 - (n/8))) +
                             (0.06 * (dew_point - dry_bulb)) *
                             (1 + 100 * (1 - np.exp(- (wind_speed / wind_speed_cut_off)**20))))

        return dew_yield

    def water_vapor_pressure(dew_pt_temp):
        """
        I'm wanting to discontinue use of the function and replace it with _psat(temp) because these are teh same function.

        """
        water_vapor_pressure = (np.exp((3.2575315268E-13 * dew_pt_temp**6) -
                                     (1.5680734584E-10 * dew_pt_temp**5) +
                                     (2.2213041913E-08 * dew_pt_temp**4) +
                                     (2.3720766595E-7 * dew_pt_temp**3) -
                                     (4.0316963015E-04 * dew_pt_temp**2) +
                                     (7.9836323361E-02 * dew_pt_temp) -
                                     (5.6983551678E-1)))

        return water_vapor_pressure

    def _psat(temp):
        """
        Function calculated the water saturation temperature or dew point for a given water vapor pressure.
        Water vapor pressure model created from an emperical fit of ln(Psat) vs temperature using a 6th order polynomial fit in microsoft Excel. The fit produced  R^2=0.999813.
        Calculation created by Michael Kempe, unpublished data.

        Parameters
        -----------
        temp : float
            Temperature [°C]

        Returns
        -------
        _psat : float
            Water vapor pressure [kPa]

        """
        psat = np.exp((3.2575315268E-13 * temp**6) -
                       (1.5680734584E-10 * temp**5) +
                       (2.2213041913E-08 * temp**4) +
                       (2.3720766595E-7 * temp**3) -
                       (4.0316963015E-04 * temp**2) +
                       (7.9836323361E-02 * temp) -
                       (5.6983551678E-1))

        return psat

    def rh_surface_outside(rh_ambient, temp_ambient, temp_module):
        """
        Function calculates the Relative Humidity of a Solar Panel Surface at module temperature

        Parameters
        ----------
        rh_ambient : float
            The ambient outdoor environmnet relative humidity expressed as a fraction or as a percent.
        temp_ambient : float
            The ambient outdoor environmnet temperature [°C]
        temp_module : float
            The surface temperature of the solar panel module [°C]

        Returns
        --------
        rh_Surface : float
            The relative humidity of the surface of a solar module as a fraction or percent depending on input.

        """
        rh_Surface = rh_ambient * \
            (StressFactors._psat(temp_ambient) /
             StressFactors._psat(temp_module))

        return rh_Surface

        ###########
        # Front Encapsulant RH
        ###########

    def _diffusivity_numerator(rh_ambient, temp_ambient, temp_module, So=1.81390702, Eas=16.729, Ead=38.14):
        """
        Calculation is used in determining a weighted average Relative Humidity of the outside surface of a module.
        This funciton is used exclusively in the function _diffusivity_weighted_water and could be combined.
        
        The function returns values needed for the numerator of the Diffusivity weighted water
        content equation. This function will return a pandas series prior to summation of the
        numerator

        Parameters
        ----------
        rh_ambient : pandas series (float)
            The ambient outdoor environmnet relative humidity in (%)
            EXAMPLE: "50 = 50% NOT .5 = 50%"
        temp_ambient : pandas series (float)
            The ambient outdoor environmnet temperature in Celsius
        temp_module : pandas series (float)
            The surface temperature in Celsius of the solar panel module
        So : float
            Float, Encapsulant solubility prefactor in [g/cm3]
            So = 1.81390702(g/cm3) is the suggested value for EVA.
        Eas : float
            Encapsulant solubility activation energy in [kJ/mol]
            Eas = 16.729(kJ/mol) is the suggested value for EVA.
        Ead : float
            Encapsulant diffusivity activation energy in [kJ/mol]
            Ead = 38.14(kJ/mol) is the suggested value for EVA.

        Returns
        -------
        diff_numerator : pandas series (float)
            Nnumerator of the Sdw equation prior to summation

        """

        # Get the relative humidity of the surface
        rh_surface = StressFactors.rh_surface_outside(
            rh_ambient, temp_ambient, temp_module)

        # Generate a series of the numerator values "prior to summation"
        diff_numerator = So * np.exp(- (Eas / (0.00831446261815324 * (temp_module + 273.15))))\
                                * rh_surface * \
                                np.exp(- (Ead / (0.00831446261815324 * (temp_module + 273.15))))

        return diff_numerator

    def _diffusivity_denominator(temp_module, Ead=38.14):
        """
        Calculation is used in determining a weighted average Relative Humidity of the outside surface of a module.
        This funciton is used exclusively in the function _diffusivity_weighted_water and could be combined.
        
        The function returns values needed for the denominator of the Diffusivity
        weighted water content equation(diffuse_water). This function will return a pandas
        series prior to summation of the denominator

        Parameters
        ----------
        Ead : float
            Encapsulant diffusivity activation energy in [kJ/mol]
            38.14(kJ/mol) is the suggested value for EVA.
        temp_module : pandas series (float)
            The surface temperature in Celsius of the solar panel module

        Returns
        -------
        diff_denominator : pandas series (float)
            Denominator of the diffuse_water equation prior to summation

        """

        diff_denominator = np.exp(- (Ead /
                                   (0.00831446261815324 * (temp_module + 273.15))))
        return diff_denominator

       
    def _diffusivity_weighted_water(rh_ambient, temp_ambient, temp_module,
                                    So=1.81390702,  Eas=16.729, Ead=38.14):
        """
        Calculation is used in determining a weighted average water content at the surface of a module.
        It is used as a constant water content that is equivalent to the time varying one with respect to moisture ingress.

        The function calculates the Diffusivity weighted water content. 

        Parameters
        ----------
        rh_ambient : pandas series (float)
            The ambient outdoor environmnet relative humidity in (%)
            EXAMPLE: "50 = 50% NOT .5 = 50%"
        temp_ambient : pandas series (float)
            The ambient outdoor environmnet temperature in Celsius
        temp_module : pandas series (float)
            The surface temperature in Celsius of the solar panel module
        So : float
            Float, Encapsulant solubility prefactor in [g/cm3]
            So = 1.81390702(g/cm3) is the suggested value for EVA.
        Eas : float
            Encapsulant solubility activation energy in [kJ/mol]
            Eas = 16.729(kJ/mol) is the suggested value for EVA.
        Ead : float
            Encapsulant diffusivity activation energy in [kJ/mol]
            Ead = 38.14(kJ/mol) is the suggested value for EVA.

        Returns
        ------
        diffuse_water : float
            Diffusivity weighted water content

        """

        numerator = StressFactors._diffusivity_numerator(
            rh_ambient, temp_ambient, temp_module, So,  Eas, Ead)
        # get the summation of the numerator
        numerator = numerator.sum(axis=0, skipna=True)

        denominator = StressFactors._diffusivity_denominator(temp_module, Ead)
        # get the summation of the denominator
        denominator = denominator.sum(axis=0, skipna=True)

        diffuse_water = (numerator / denominator)/100

        return diffuse_water

    def rh_front_encap(rh_ambient, temp_ambient, temp_module, So=1.81390702, Eas=16.729):
        """
        Function returns a diffusivity weighted average Relative Humidity of the module surface.

        Parameters
        ----------
        rh_ambient : series (float)
            ambient Relative Humidity (%)
        temp_ambient : series (float)
            ambient outdoor temperature [°C]        
        temp_module : pandas series (float)
            The surface temperature in Celsius of the solar panel module
            "module temperature [°C]"
        So : float
            Encapsulant solubility prefactor in [g/cm3]
            So = 1.81390702(g/cm3) is the suggested value for EVA.
        Eas : float
            Encapsulant solubility activation energy in [kJ/mol]
            Eas = 16.729(kJ/mol) is the suggested value for EVA.


        Return
        ------
        RHfront_series : pandas series (float)
            Relative Humidity of Frontside Solar module Encapsulant

        """
        diffuse_water = StressFactors._diffusivity_weighted_water(rh_ambient=rh_ambient,
                                                                    temp_ambient=temp_ambient,
                                                                    temp_module=temp_module)

        RHfront_series = (diffuse_water / (So * np.exp(- (Eas / (0.00831446261815324 *
                                                       (temp_module + 273.15)))))) * 100

        return RHfront_series

        ###########
        # Back Encapsulant Relative Humidity
        ###########

    def _csat(temp_module, So=1.81390702, Eas=16.729):
        """
        Calculation is used in determining Relative Humidity of Backside Solar
        Module Encapsulant, and returns saturation of Water Concentration [g/cm³]

        Parameters
        -----------
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
            Saturation of Water Concentration (g/cm³)

        """

        # Saturation of water concentration
        Csat = So * \
            np.exp(- (Eas / (0.00831446261815324 * (273.15 + temp_module))))

        return Csat

    def _ceq(Csat, rh_SurfaceOutside):
        """
        Calculation is used in determining Relative Humidity of Backside Solar
        Module Encapsulant, and returns Equilibration water concentration (g/cm³)

        Parameters
        ------------
        Csat : pandas series (float)
            Saturation of Water Concentration (g/cm³)
        rh_SurfaceOutside : pandas series (float)
            The relative humidity of the surface of a solar module (%)

        Returns
        --------
        Ceq : pandas series (float)
            Equilibration water concentration (g/cm³)

        """

        Ceq = Csat * (rh_SurfaceOutside/100)

        return Ceq

    @jit(nopython=True)
    def Ce_numba(start, temp_module, rh_surface,
                    WVTRo=7970633554, EaWVTR=55.0255, So=1.81390702, l=0.5, Eas=16.729):
        """
        Calculation is used in determining Relative Humidity of Backside Solar
        Module Encapsulant. This function returns a numpy array of the Concentration of water in the
        encapsulant at every time step

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
            The relative humidity of the surface of a solar module (%)
            EXAMPLE: "50 = 50% NOT .5 = 50%"
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
        l : float
            Thickness of the backside encapsulant (mm).
            The suggested value for encapsulat is EVA l=0.5(mm)
        Eas : float
            Encapsulant solubility activation energy in [kJ/mol]
            Eas = 16.729(kJ/mol) is the suggested value for EVA.

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
                Ce = Ce_list[i-1]

            Ce = Ce + ((WVTRo/100/100/24 * np.exp(-((EaWVTR) / (0.00831446261815324 * (temp_module[i] + 273.15))))) /
                       (So * l/10 * np.exp(-((Eas) / (0.00831446261815324 * (temp_module[i] + 273.15))))) *
                       (rh_surface[i]/100 * So * np.exp(-((Eas) / (0.00831446261815324 * (temp_module[i] + 273.15)))) - Ce))

            Ce_list[i] = Ce

        return Ce_list

    def rh_back_encap(rh_ambient, temp_ambient, temp_module,
                    WVTRo=7970633554, EaWVTR=55.0255, So=1.81390702, l=0.5, Eas=16.729):
        """
        rh_back_encap()

        Function to calculate the Relative Humidity of Backside Solar Module Encapsulant
        and return a pandas series for each time step        

        Parameters
        -----------
        rh_ambient : pandas series (float)
            The ambient outdoor environmnet relative humidity in (%)
            EXAMPLE: "50 = 50% NOT .5 = 50%"
        temp_ambient : pandas series (float)
            The ambient outdoor environmnet temperature in Celsius
        temp_module : list (float)
            The surface temperature in Celsius of the solar panel module
            "module temperature [°C]"
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
        l : float
            Thickness of the backside encapsulant (mm).
            The suggested value for encapsulat is EVA l=0.5(mm)
        Eas : float
            Encapsulant solubility activation energy in [kJ/mol]
            Eas = 16.729(kJ/mol) is the suggested value for EVA.

        Returns
        --------  
        RHback_series : pandas series (float)
            Relative Humidity of Backside Solar Module Encapsulant     

        """

        rh_surface = StressFactors.rh_surface_outside(rh_ambient=rh_ambient,
                                                       temp_ambient=temp_ambient,
                                                       temp_module=temp_module)

        Csat = StressFactors._csat(
            temp_module=temp_module, So=So, Eas=Eas)
        Ceq = StressFactors._ceq(Csat=Csat, rh_SurfaceOutside=rh_surface)

        start = Ceq[0]

        # Need to convert these series to numpy arrays for numba function
        temp_module_numba = temp_module.to_numpy()
        rh_surface_numba = rh_surface.to_numpy()
        Ce_nparray = StressFactors.Ce_numba(start=start,
                                               temp_module=temp_module_numba,
                                               rh_surface=rh_surface_numba,
                                               WVTRo=WVTRo,
                                               EaWVTR=EaWVTR,
                                               So=So,
                                               l=l,
                                               Eas=Eas)

        #RHback_series = 100 * (Ce_nparray / (So * np.exp(-( (Eas) / 
        #                   (0.00831446261815324 * (temp_module + 273.15))  )) ))
        RHback_series = 100 * (Ce_nparray / Csat)

        return RHback_series

    def rh_backsheet_from_encap(rh_back_encap, rh_surface_outside):
        """
        Function to calculate the Relative Humidity of solar module backsheet as timeseries.
        Requires the RH of the backside encapsulant and the outside surface of the module.
        See StressFactors.rh_back_encap and StressFactors.rh_surface_outside

        Parameters
        ----------
        rh_back_encap : pandas series (float)
            Relative Humidity of Frontside Solar module Encapsulant. *See rh_back_encap()
        rh_surface_outside : pandas series (float)
            The relative humidity of the surface of a solar module. *See rh_surface_outside()

        Returns
        --------
        RHbacksheet_series : pandas series (float)
            Relative Humidity of Backside Backsheet of a Solar Module

        @return rh_Surface     -
\                             
        """

        RHbacksheet_series = (rh_back_encap + rh_surface_outside)/2

        return RHbacksheet_series

    def rh_backsheet(rh_ambient, temp_ambient, temp_module,
                    WVTRo=7970633554, EaWVTR=55.0255, So=1.81390702, l=0.5, Eas=16.729):
        """Function to calculate the Relative Humidity of solar module backsheet as timeseries.
        This function uses the same formula as StressFactors.rh_backsheet_from_encap but does
        not require you to independently solve for the backside ecapsulant and surface humidity
        before calculation. 

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
        l : float
            Thickness of the backside encapsulant (mm).
            The suggested value for encapsulat is EVA l=0.5(mm)
        Eas : float
            Encapsulant solubility activation energy in [kJ/mol]
            Eas = 16.729(kJ/mol) is the suggested value for EVA.

        Returns
        --------
        rh_backsheet : float series or array
            relative humidity of the PV backsheet as a time-series   
        """

        rh_back_encap = StressFactors.rh_back_encap(rh_ambient=rh_ambient,
                                                    temp_ambient=temp_ambient,
                                                    temp_module=temp_module, WVTRo=WVTRo,
                                                    EaWVTR=EaWVTR, So=So, l=l, Eas=Eas)
        rh_surface = StressFactors.rh_surface_outside(rh_ambient=rh_ambient,
                                                    temp_ambient=temp_ambient,
                                                    temp_module=temp_module)
        rh_backsheet = (rh_back_encap + rh_surface)/2
        return rh_backsheet

    def rh_module(rh_ambient, temp_ambient, temp_module,
                WVTRo=7970633554, EaWVTR=55.0255, So=1.81390702, l=0.5, Eas=16.729,
                pandas=True):
        """
        Generate the relative humidity for the following components
        - outside surface of the module
        - frontside encapsulant
        - backside encpasulant
        - backsheet
        To generate these individually, see StressFactors.rh_surface_outside, etc

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
        l : float
            Thickness of the backside encapsulant (mm).
            The suggested value for encapsulat is EVA l=0.5(mm)
        Eas : float
            Encapsulant solubility activation energy in [kJ/mol]
            Eas = 16.729(kJ/mol) is the suggested value for EVA.
        pandas: boolean, default=True
            If true, the calculation will return a dataframe containing named columns
            for each material. If false, it will instead return a tuple where each value is
            a time series.

        Returns
        --------
        rh_surface_outside : float series
            relative humidity immediately outside the surface of a module
        rh_front_encap : float series
            relative humidity of the module front encapsulant
        rh_back_encap : float series
            relative humidity of the module backside encapsulant
        rh_backsheet : float series or array
            relative humidity of the PV backsheet as a time-series
        """

        rh_surface_outside = StressFactors.rh_surface_outside(rh_ambient=rh_ambient,
                                                            temp_ambient=temp_ambient,
                                                            temp_module=temp_module)

        rh_front_encap = StressFactors.rh_front_encap(rh_ambient, temp_ambient, temp_module,
                                                    So=So, Eas=Eas)

        rh_back_encap = StressFactors.rh_back_encap(rh_ambient=rh_ambient,
                                                    temp_ambient=temp_ambient,
                                                    temp_module=temp_module, WVTRo=WVTRo,
                                                    EaWVTR=EaWVTR, So=So, l=l, Eas=Eas)

        rh_backsheet = StressFactors.rh_backsheet_from_encap(rh_back_encap=rh_back_encap,
                                                            rh_surface_outside=rh_surface_outside)

        if not pandas:
            return (rh_surface_outside, rh_front_encap, rh_back_encap, rh_backsheet)
        
        else:
            data = {'surface_outside': rh_surface_outside,
                    'front_encap': rh_front_encap,
                    'back_encap': rh_back_encap,
                    'backsheet': rh_backsheet}
            results = pd.DataFrame(data=data)
            return results

class Degradation:

    def _deg_rate_env(poa_global, temp_cell, temp_chamber, x, Tf):
        """
        Helper function. Find the rate of degradation kenetics using the Fischer model.
        Degradation kentics model interpolated 50 coatings with respect to
        color shift, cracking, gloss loss, fluorescense loss,
        retroreflectance loss, adhesive transfer, and shrinkage.

        (ADD IEEE reference)

        Parameters
        ------------
        poa_global : float
            (Global) Plan of Array irradiance (W/m²)
        temp_cell : float
            Solar module cell temperature [°C]
        temp_chamber : float
            Reference temperature [°C] "Chamber Temperature"
        x : float
            Fit parameter
        Tf : float
            Multiplier for the increase in degradation
                                          for every 10[°C] temperature increase

        Returns
        --------
        degradationrate : float
            rate of Degradation (NEED TO ADD METRIC)

        """
        return poa_global**(x) * Tf ** ((temp_cell - temp_chamber)/10)

    def _deg_rate_chamber(I_chamber, x):
        """
        Helper function. Find the rate of degradation kenetics of a simulated chamber. Mike Kempe's
        calculation of the rate of degradation inside a accelerated degradation chamber.

        (ADD IEEE reference)

        Parameters
        ----------
        I_chamber : float
            Irradiance of Controlled Condition W/m²
        x : float
            Fit parameter

        Returns
        --------
        chamberdegradationrate : float
            Degradation rate of chamber
        """
        chamberdegradationrate = I_chamber ** (x)

        return chamberdegradationrate

    def _acceleration_factor(numerator, denominator):
        """
        Helper Function. Find the acceleration factor

        (ADD IEEE reference)

        Parameters
        ----------
        numerator : float
            Typically the numerator is the chamber settings
        denominator : float
            Typically the TMY data summation

        Returns
        -------
        chamberAccelerationFactor : float
            Acceleration Factor of chamber (NEED TO ADD METRIC)
        """

        chamberAccelerationFactor = (numerator / denominator)

        return chamberAccelerationFactor

    def vantHoff_deg(I_chamber, poa_global, temp_cell, temp_chamber, x=0.5, Tf=1.41):
        """

        Van 't Hoff Irradiance Degradation

        Parameters
        -----------
        I_chamber : float
            Irradiance of Controlled Condition W/m²
        poa_global : float series
            Global Plane of Array Irradiance W/m²
        temp_cell : pandas series
            Solar module temperature or Cell temperature [°C]
        temp_chamber : float
            Reference temperature [°C] "Chamber Temperature"
        x : float
            fit parameter
        Tf : float
            Multiplier for the increase in degradation for every 10[°C] temperature increase

        Returns
        -------
        accelerationFactor : float or series
            Degradation acceleration factor

        """
        rateOfDegEnv = Degradation._deg_rate_env(poa_global=poa_global,
                                                 temp_cell=temp_cell,
                                                 temp_chamber=temp_chamber,
                                                 x=x,
                                                 Tf=Tf)
        #sumOfDegEnv = rateOfDegEnv.sum(axis = 0, skipna = True)
        avgOfDegEnv = rateOfDegEnv.mean()

        rateOfDegChamber = Degradation._deg_rate_chamber(I_chamber, x)

        accelerationFactor = Degradation._acceleration_factor(
            rateOfDegChamber, avgOfDegEnv)

        return accelerationFactor

    def _to_eq_vantHoff(temp_cell, Tf=1.41):
        """
        Function to obtain the Vant Hoff temperature equivalent [°C]

        Parameters
        ----------
        Tf : float
            Multiplier for the increase in degradation for every 10[°C] temperature increase
        temp_cell : pandas series
            Solar module temperature or Cell temperature [°C]

        Returns
        -------
        Toeq : float
            Vant Hoff temperature equivalent [°C]

        """
        toSum = Tf ** (temp_cell / 10)
        summation = toSum.sum(axis=0, skipna=True)

        Toeq = (10 / np.log(Tf)) * np.log(summation / len(temp_cell))

        return Toeq


    def IwaVantHoff(poa_global, temp_cell, Teq=None, x=0.5, Tf=1.41):
        """
        IWa : Environment Characterization (W/m²)
        *for one year of degredation the controlled environmnet lamp settings will
            need to be set to IWa

        Parameters
        -----------
        poa_global : float series
            Global Plane of Array Irradiance W/m²
        temp_cell : float series
            Solar module temperature or Cell temperature [°C]
        Teq : series
            VantHoff equivalent temperature [°C]
        x : float
            Fit parameter
        Tf : float
            Multiplier for the increase in degradation for every 10[°C] temperature increase

        Returns
        --------
        Iwa : float
            Environment Characterization (W/m²)

        """
        if Teq is None:
            Teq = Degradation._to_eq_vantHoff(temp_cell, Tf)
        toSum = (poa_global ** x) * (Tf ** ((temp_cell - Teq)/10))
        summation = toSum.sum(axis=0, skipna=True)

        Iwa = (summation / len(poa_global)) ** (1 / x)

        return Iwa

    def _arrhenius_denominator(poa_global, rh_outdoor, temp_cell, Ea, x, n):
        """
        Helper function. Calculates the rate of degredation of the Environmnet

        Parameters
        ----------
        poa_global : float series
            (Global) Plan of Array irradiance (W/m²)
        x : float
            Fit parameter
        rh_outdoor : pandas series
            Relative Humidity of material of interest. Acceptable relative
            humiditys can be calculated from these functions: rh_backsheet(),
            rh_back_encap(); rh_front_encap();  rh_surface_outside()
        n : float
            Fit parameter for relative humidity
        temp_cell : pandas series
            Solar module temperature or Cell temperature [°C]
        Ea : float
            Degredation Activation Energy (kJ/mol)

        Returns
        -------
        environmentDegradationRate : pandas series
            Degradation rate of environment
        """

        environmentDegradationRate = poa_global**(x) * rh_outdoor**(
            n) * np.exp(- (Ea / (0.00831446261815324 * (temp_cell + 273.15))))

        return environmentDegradationRate

    def _arrhenius_numerator(I_chamber, rh_chamber,  temp_chamber, Ea, x, n):
        """
        Helper function. Find the rate of degradation of a simulated chamber.

        Parameters
        ----------
        I_chamber : float
            Irradiance of Controlled Condition W/m²
        Rhchamber : float
            Relative Humidity of Controlled Condition (%)
            EXAMPLE: "50 = 50% NOT .5 = 50%"
        temp_chamber : float
            Reference temperature [°C] "Chamber Temperature"
        Ea : float
            Degredation Activation Energy (kJ/mol)
        x : float
            Fit parameter
        n : float
            Fit parameter for relative humidity

        Returns
        --------
        arrheniusNumerator : float
            Degradation rate of the chamber
        """

        arrheniusNumerator = (I_chamber ** (x) * rh_chamber ** (n) *
                              np.exp(- (Ea / (0.00831446261815324 *
                                              (temp_chamber+273.15)))))
        return arrheniusNumerator

    def arrhenius_deg(I_chamber, rh_chamber, temp_chamber, rh_outdoor, poa_global, temp_cell,
                        Ea, x=0.5, n=1):
        """
        Calculate the Acceleration Factor between the rate of degredation of a
        modeled environmnet versus a modeled controlled environmnet

        Example: "If the AF=25 then 1 year of Controlled Environment exposure
                    is equal to 25 years in the field"

        Parameters
        ----------
        I_chamber : float
            Irradiance of Controlled Condition W/m²
        rh_chamber : float
            Relative Humidity of Controlled Condition (%).
            EXAMPLE: "50 = 50% NOT .5 = 50%"
        temp_chamber : float
            Reference temperature [°C] "Chamber Temperature"
        rh_outdoor : float series
            Relative Humidity of material of interest
            Acceptable relative humiditys can be calculated
            from these functions: rh_backsheet(), rh_back_encap(), rh_front_encap(),
            rh_surface_outside()
        poa_global : pandas series
            Global Plane of Array Irradiance W/m²
        temp_cell : pandas series
            Solar module temperature or Cell temperature [°C]
        Ea : float
            Degredation Activation Energy (kJ/mol)
        x : float
            Fit parameter
        n : float
            Fit parameter for relative humidity

        Returns
        --------
        accelerationFactor : pandas series
            Degradation acceleration factor

        """
        arrheniusDenominator = Degradation._arrhenius_denominator(poa_global=poa_global,
                                                                 rh_outdoor=rh_outdoor,
                                                                 temp_cell=temp_cell,
                                                                 Ea=Ea,
                                                                 x=x,
                                                                 n=n)

        AvgOfDenominator = arrheniusDenominator.mean()

        arrheniusNumerator = Degradation._arrhenius_numerator(I_chamber=I_chamber, 
                                                             rh_chamber=rh_chamber,
                                                             temp_chamber=temp_chamber, Ea=Ea, x=x, n=n)

        accelerationFactor = Degradation._acceleration_factor(
            arrheniusNumerator, AvgOfDenominator)

        return accelerationFactor

    def _T_eq_arrhenius(temp_cell, Ea):
        """
        Get the Temperature equivalent required for the settings of the controlled environment
        Calculation is used in determining Arrhenius Environmental Characterization

        Parameters
        -----------
        temp_cell : pandas series
            Solar module temperature or Cell temperature [°C]
        Ea : float
            Degredation Activation Energy (kJ/mol)

        Returns
        -------
        Teq : float
            Temperature equivalent (Celsius) required
            for the settings of the controlled environment

        """

        summationFrame = np.exp(- (Ea /
                                   (0.00831446261815324 * (temp_cell + 273.15))))
        sumForTeq = summationFrame.sum(axis=0, skipna=True)
        Teq = -((Ea) / (0.00831446261815324 * np.log(sumForTeq / len(temp_cell))))
        # Convert to celsius
        Teq = Teq - 273.15

        return Teq

    def _RH_wa_arrhenius(rh_outdoor, temp_cell, Ea, Teq=None, n=1):
        """
        NOTE

        Get the Relative Humidity Weighted Average.
        Calculation is used in determining Arrhenius Environmental Characterization

        Parameters
        -----------
        rh_outdoor : pandas series
            Relative Humidity of material of interest. Acceptable relative
            humiditys can be calculated from the below functions:
            rh_backsheet(), rh_back_encap(), rh_front_encap(), rh_surface_outside()
        temp_cell : pandas series
            solar module temperature or Cell temperature [°C]
        Ea : float
            Degredation Activation Energy (kJ/mol)
        Teq : series
            Equivalent Arrhenius temperature [°C]
        n : float
            Fit parameter for relative humidity

        Returns
        --------
        RHwa : float
            Relative Humidity Weighted Average (%)

        """

        if Teq is None:
            Teq = Degradation._T_eq_arrhenius(temp_cell, Ea)

        summationFrame = (rh_outdoor ** n) * np.exp(- (Ea /
                                                      (0.00831446261815324 * (temp_cell + 273.15))))
        sumForRHwa = summationFrame.sum(axis=0, skipna=True)
        RHwa = (sumForRHwa / (len(summationFrame) * np.exp(- (Ea /
                                                (0.00831446261815324 * (Teq + 273.15)))))) ** (1/n)

        return RHwa

    def IwaArrhenius(poa_global, rh_outdoor, temp_cell, Ea,
                     RHwa=None, Teq=None, x=0.5, n=1):
        """
        TODO:   CHECK
                STANDARDIZE

        Function to calculate IWa, the Environment Characterization (W/m²)
        *for one year of degredation the controlled environmnet lamp settings will
            need to be set at IWa

        Parameters
        ----------
        poa_global : float
            (Global) Plan of Array irradiance (W/m²)
        rh_outdoor : pandas series
            Relative Humidity of material of interest
            Acceptable relative humiditys can be calculated
            from these functions: rh_backsheet(), rh_back_encap(), rh_front_encap()
                                  rh_surface_outside()
        temp_cell : pandas series
            Solar module temperature or Cell temperature [°C]
        Ea : float
            Degradation Activation Energy (kJ/mol)
        RHwa : float
            Relative Humidity Weighted Average (%)
        Teq : float
            Temperature equivalent (Celsius) required
            for the settings of the controlled environment
        x : float
            Fit parameter
        n : float
            Fit parameter for relative humidity

        Returns
        --------
        Iwa : float
            Environment Characterization (W/m²)

        """
        if Teq is None:
            Teq = Degradation._T_eq_arrhenius(temp_cell, Ea)

        if RHwa is None:
            RHwa = Degradation._RH_wa_arrhenius(rh_outdoor, temp_cell, Ea)

        numerator = poa_global**(x) * rh_outdoor**(n) * \
            np.exp(- (Ea / (0.00831446261815324 * (temp_cell + 273.15))))
        sumOfNumerator = numerator.sum(axis=0, skipna=True)

        denominator = (len(numerator)) * ((RHwa)**n) * \
            (np.exp(- (Ea / (0.00831446261815324 * (Teq + 273.15)))))

        IWa = (sumOfNumerator / denominator)**(1/x)

        return IWa


############
# Misc. Functions for Energy Calcs
############

    def _rh_Above85(rh):
        """
        Helper function. Determines if the relative humidity is above 85%.

        Parameters
        ----------
        rh : float
            Relative Humidity %

        Returns
        --------
        rhabove85 : boolean
            True if the relative humidity is above 85% or False if the relative
            humidity is below 85%

        """

        if rh > 85:
            rhabove85 = True

        else:
            rhabove85 = False

        return rhabove85

    def _hoursRH_Above85(df):
        """
        Helper Function. Count the number of hours relative humidity is above 85%.

        Parameters
        ----------
        df : dataframe
            DataFrame, dataframe containing Relative Humidity %

        Returns
        -------
        numhoursabove85 : int
            Number of hours relative humidity is above 85%

        """
        booleanDf = df.apply(lambda x: Degradation._rh_Above85(x))
        numhoursabove85 = booleanDf.sum()

        return numhoursabove85

    def _whToGJ(wh):
        """
        NOTE: unused, remove?

        Helper Function to convert Wh/m² to GJ/m²

        Parameters
        -----------
        wh : float
            Input Value in Wh/m²

        Returns
        -------
        gj : float
            Value in GJ/m²

        """

        gj = 0.0000036 * wh

        return gj

    def _gJtoMJ(gJ):
        """
        NOTE: unused, remove?

        Helper Function to convert GJ/m² to MJ/y

        Parameters
        -----------
        gJ : float
            Value in GJ/m^-2

        Returns
        -------
        MJ : float
            Value in MJ/m^-2

        """
        MJ = gJ * 1000

        return MJ

    def degradation(spectra, rh_module, temp_module, wavelengths,
                    Ea=40.0, n=1.0, x=0.5, C2=0.07, C=1.0):
        '''
        Compute degredation as double integral of Arrhenius (Activation
        Energy, RH, Temperature) and spectral (wavelength, irradiance)
        functions over wavelength and time.

        Parameters
        ----------
        spectra : pd.Series type=Float
            front or rear irradiance at each wavelength in "wavelengths"
        rh_module : pd.Series type=Float
            module RH, time indexed
        temp_module : pd.Series type=Float
            module temperature, time indexed
        wavelengths : int-array
            integer array (or list) of wavelengths tested w/ uniform delta
            in nanometers [nm]
        Ea : float
            Arrhenius activation energy. The default is 40. [kJ/mol]
        n : float
            Fit paramter for RH sensitivity. The default is 1.
        x : float
            Fit parameter for irradiance sensitivity. Typically
            0.6 +- 0.22
        C2 : float
            Fit parameter for sensitivity to wavelength exponential.
            Typically 0.07
        C : float
            Fit parameter for the Degradation equaiton
            Typically 1.0

        Returns
        -------
        degradation : float
            Total degredation factor over time and wavelength.

        '''
        # --- TO DO ---
        # unpack input-dataframe
        # spectra = df['spectra']
        # temp_module = df['temp_module']
        # rh_module = df['rh_module']

        # Constants
        R = 0.0083145  # Gas Constant in [kJ/mol*K]

        wav_bin = list(np.diff(wavelengths))
        wav_bin.append(wav_bin[-1])  # Adding a bin for the last wavelength

        # Integral over Wavelength
        try:
            irr = pd.DataFrame(spectra.tolist(), index=spectra.index)
            irr.columns = wavelengths
        except:
            # TODO: Fix this except it works on some cases, veto it by cases
            print("Removing brackets from spectral irradiance data")
            #irr = data['spectra'].str.strip('[]').str.split(',', expand=True).astype(float)
            irr = spectra.str.strip('[]').str.split(
                ',', expand=True).astype(float)
            irr.columns = wavelengths

        sensitivitywavelengths = np.exp(-C2*wavelengths)
        irr = irr*sensitivitywavelengths
        irr *= np.array(wav_bin)
        irr = irr**x
        data = pd.DataFrame(index=spectra.index)
        data['G_integral'] = irr.sum(axis=1)

        EApR = -Ea/R
        C4 = np.exp(EApR/temp_module)

        RHn = rh_module**n
        data['Arr_integrand'] = C4*RHn

        data['dD'] = data['G_integral']*data['Arr_integrand']

        degradation = C*data['dD'].sum(axis=0)

        return degradation

    def _avg_daily_temp_change(time_range, temp_cell):
        """
        Helper function. Get the average of a year for the daily maximum temperature change.

        For every 24hrs this function will find the delta between the maximum
        temperature and minimun temperature.  It will then take the deltas for
        every day of the year and return the average delta.

        Parameters
        ------------
        time_range : timestamp series
            Local time of specific site by the hour
            year-month-day hr:min:sec . (Example) 2002-01-01 01:00:00
        temp_cell : float series
            Photovoltaic module cell temperature(Celsius) for every hour of a year

        Returns
        -------
        avg_daily_temp_change : float
            Average Daily Temerature Change for 1-year (Celsius)
        avg_max_temp_cell : float
            Average of Daily Maximum Temperature for 1-year (Celsius)

        """
        
        if time_range.dtype == 'object':
            time_range = pd.to_datetime(time_range)
        
        # Setup frame for vector processing
        timeAndTemp_df = pd.DataFrame(columns=['Cell Temperature'])
        timeAndTemp_df['Cell Temperature'] = temp_cell
        timeAndTemp_df.index = time_range
        timeAndTemp_df['month'] = timeAndTemp_df.index.month
        timeAndTemp_df['day'] = timeAndTemp_df.index.day

        # Group by month and day to determine the max and min cell Temperature [°C] for each day
        dailyMaxCellTemp_series = timeAndTemp_df.groupby(
            ['month', 'day'])['Cell Temperature'].max()
        dailyMinCellTemp_series = timeAndTemp_df.groupby(
            ['month', 'day'])['Cell Temperature'].min()
        temp_cell_change = pd.DataFrame(
            {'Max': dailyMaxCellTemp_series, 'Min': dailyMinCellTemp_series})
        temp_cell_change['TempChange'] = temp_cell_change['Max'] - \
            temp_cell_change['Min']

        # Find the average temperature change for every day of one year [°C]
        avg_daily_temp_change = temp_cell_change['TempChange'].mean()
        # Find daily maximum cell temperature average
        avg_max_temp_cell = dailyMaxCellTemp_series.mean()

        return avg_daily_temp_change, avg_max_temp_cell

    def _times_over_reversal_number(temp_cell, reversal_temp):
        """
        Helper function. Get the number of times a temperature increases or decreases over a
        specific temperature gradient.

        Parameters
        ------------
        temp_cell : float series
            Photovoltaic module cell temperature(Celsius)
        reversal_temp : float
            Temperature threshold to cross above and below

        Returns
        --------
        num_changes_temp_hist : int
            Number of times the temperature threshold is crossed

        """
        # Find the number of times the temperature crosses over 54.8(°C)

        temp_df = pd.DataFrame()
        temp_df['CellTemp'] = temp_cell
        temp_df['COMPARE'] = temp_cell
        temp_df['COMPARE'] = temp_df.COMPARE.shift(-1)

        #reversal_temp = 54.8

        temp_df['cross'] = (
            ((temp_df.CellTemp >= reversal_temp) & (temp_df.COMPARE < reversal_temp)) |
            ((temp_df.COMPARE > reversal_temp) & (temp_df.CellTemp <= reversal_temp)) |
            (temp_df.CellTemp == reversal_temp))

        num_changes_temp_hist = temp_df.cross.sum()

        return num_changes_temp_hist

    def solder_fatigue(time_range, temp_cell, reversal_temp=54.8, n=1.9, b=0.33, C1=405.6, Q=0.12):
        """
        Get the Thermomechanical Fatigue of flat plate photovoltaic module solder joints.
        Damage will be returned as the rate of solder fatigue for one year. Based on:

            Bosco, N., Silverman, T. and Kurtz, S. (2020). Climate specific thermomechanical
            fatigue of flat plate photovoltaic module solder joints. [online] Available
            at: https://www.sciencedirect.com/science/article/pii/S0026271416300609
            [Accessed 12 Feb. 2020].

        This function uses the default values for 60-min input intervals from Table 4 of the above
        paper. For other use cases, please refer to the paper for recommended values of C1 and
        the reversal temperature.
        
        Parameters
        ------------
        time_range : timestamp series
            Local time of specific site by the hour year-month-day hr:min:sec
            (Example) 2002-01-01 01:00:00
        temp_cell : float series
            Photovoltaic module cell temperature(Celsius) for every hour of a year
        reversal_temp : float
            Temperature threshold to cross above and below.
            See the paper for other use cases
        n : float
            fit parameter for daily max temperature amplitude
        b : float
            fit parameter for reversal temperature
        C1 : float
            scaling constant, see the paper for details on appropriate values
        Q : float
            activation energy [eV]

        Returns
        --------
        damage : float series
            Solder fatigue damage for a time interval depending on time_range (kPa)

        """

        # TODO Make this function have more utility.
        # People want to run all the scenarios from the bosco paper.
        # Currently have everything hard coded for hourly calculation
        # i.e. 405.6, 1.9, .33, .12

        # Boltzmann Constant
        k = .00008617333262145
        
        temp_amplitude, temp_max_avg = Degradation._avg_daily_temp_change(time_range, temp_cell)
        
        temp_max_avg = convert_temperature(temp_max_avg, 'Celsius', 'Kelvin')
        
        num_changes_temp_hist = Degradation._times_over_reversal_number(temp_cell, reversal_temp)

        damage = C1*(temp_amplitude**n)*(num_changes_temp_hist**b)*np.exp(-(Q/(k*temp_max_avg)))
        
        # Convert pascals to kilopascals
        damage = damage/1000
        
        return damage


class BOLIDLeTID:
    """
    Class for Field extrapolation of BOLID and LeTID
    """

    ###########
    # field Degradation Profile
    ###########

    def fieldDegradationProfile(T, x0=[1E16, 0, 0], t=np.linspace(0, 1000, 10000),
                                v_AB=4E3, v_BA=1E13, v_BC=1.25E10,
                                v_CB=1E9, Ea_AB=0.475, Ea_BA=1.32,
                                Ea_BC=0.98, Ea_CB=1.25):
        '''
        This function calculates and plots the % of Defects on each state (A, B,C)
        as a function of time and temperature, using the kinetic parameters 
        (activation energies adn attempt fequencies). 
        Values from Repins, Solar Energy 2020
        Equations from Hallam, Energy Proc 2016

        Parameters
        ----------
        T : float
            Temperature in Kelvin Degrees
        x0 : array
            1D array with the 3 initial values for number of defects in each state,
            in the format [N_A, N_B, N_C]
        t : array
            1D array of the times to be simulated. Can be defined as a 
            numpy linspace between inital time and end time and number of sample points,
            for example t = np.linspace(0,1000,10000)
        v_AB : float
            Attempt frequency, in Hz for states A-->B
        v_BA : float
            Attempt frequency, in Hz for states B-->A
        v_BC : float
            Attempt frequency, in Hz for states B-->C
        v_CB : float
            Attempt frequency, in Hz for states C-->B
        Ea_AB : float
            Activation energy between states A-->B, in eV.
        Ea_BA : float
            Activation energy between states B-->A, in eV.
        Ea_BC : float
            Activation energy between states B-->C, in eV.
        Ea_CB : float
            Activation energy between states C-->B, in eV.

        Returns
        --------        
        '''
        from scipy.integrate import odeint
        import matplotlib.pyplot as plt

        # PARAMETERS
        kb = 8.617333262145E-5  # eV * K-1 Boltmanzz Constnat

        # Arrhenius Equation
        # k = reaction rate for the transition from state i to state j
        # where i,j = A,B,C, but i != j. Also, no direct transition between
        # state A and C (k_AC = k_CA = 0)
        # k_ij = v_ij * np.exp(-Ea_ij/(kb*T))
        k_AB = v_AB * np.exp(-Ea_AB/(kb*T))
        k_AC = 0
        k_CA = 0
        k_BA = v_BA * np.exp(-Ea_BA/(kb*T))
        k_BC = v_BC * np.exp(-Ea_BC/(kb*T))
        k_CB = v_CB * np.exp(-Ea_CB/(kb*T))

        # function that returns dN_A/dt, dN_B, and dN_C
        def model(x, t, k_AB, k_BA, k_BC, k_CB):
            N_A = x[0]
            N_B = x[1]
            N_C = x[2]
            dN_Adt = k_BA * N_B - k_AB * N_A
            dN_Bdt = k_AB * N_A + k_CB * N_C - (k_BA + k_BC) * N_B
            dN_Cdt = k_BC * N_B - k_CB * N_C
            return [dN_Adt, dN_Bdt, dN_Cdt]

        # Calling the Derivate model with odeint
        # args pebble.)
        x = odeint(model, x0, t, args=(k_AB, k_BA, k_BC, k_CB))

        N_A = x[:, 0]
        N_B = x[:, 1]
        N_C = x[:, 2]

        # CAlculating Percentage of Defects in Each State
        T_def = x.sum(axis=1)  # total of defects
        P_A = N_A*100/T_def
        P_B = N_B*100/T_def
        P_C = N_C*100/T_def

        # Plotting # of Defects
        plt.semilogy(t, N_A, label='A')
        plt.semilogy(t, N_B, label='B')
        plt.semilogy(t, N_C, label='C')
        plt.legend()
        plt.xlabel('Exposure length (t)')
        plt.ylabel('Number of Defects in Each State')
        plt.show()

        # Plotting % of Defects
        plt.plot(t, P_A, label='A')
        plt.plot(t, P_B, label='B')
        plt.plot(t, P_C, label='C')
        plt.legend()
        plt.xlabel('Exposure length (t)')
        plt.ylabel('% Defects in Each State')
        plt.show()


class Standards:

    def ideal_installation_distance(df_tmy, metadata, level=0, x0 = 6, tilt=None, 
                                    azimuth=180, skymodel='isotropic'):
        '''
    
        Preliminary calculation for module gap according to IEC TS 63126. The
        steps of the calculation are:
        1. Takes a weather file
        2. Calculates the module temperature with PVLIB, for an open rack
           polymer-back which is assumed to have infinite gap
        3. Calculates the module temperature with PVLIB, for an insulated
            back module which is assumed to be flush mount with a roof (gap=0)
        4. Calculates the 98 percentile for step 2
        5. Calculates the 98 percentile for step 3
        5. Use both percentiles to calculate the effective gap "x" for the lower limit to use a
           Level 1 or Level 0 module (in IEC TS 63216), I.e. T98=80 or T98=70 respectively.

        Reference: M. Kempe, PVSC Proceedings 2023 (forthcoming)
        
        Parameters
        ----------
        df_tmy : DataFrame
            Dataframe with the weather data. Must have
            'air_temperature' and 'wind_speed'
        metadata : Dictionary
            must have 'latitude' and 'longitude'
        level : int, optional
            Options 0, or 1. Level 1 or Level 0 module (in IEC TS 63216) define 
            the testing regime for the module; the boundaries are defined 
            internally, to use a level 0 module is the boundary is less than 
            70, and for Level 1 is less than 80. Above 80 Level 2 testing 
            regime is required.
        x0 : float, optional
            Thermal decay constant, [Kempe, PVSC Proceedings 2023]
        tilt : float, optional
            Tilt of the PV array. If None, uses latitude. 
        azimuth : float, optional
            Azimuth of the PV array. The default is 180, facing south
        skymodel : str
            Tells PVlib which sky model to use. Default 'isotropic'.

        Returns
        -------
        x : float
            Recommended installation distance per IEC TS 63126.
            effective gap "x" for the lower limit to use a Level 1 or Level 0 module (in IEC TS 63216)

        '''

        if level == 0:
            T98 = 70
        if level == 1:
            T98 = 80

        # 1. Calculate Sun Position & POA Irradiance

        # Make Location
        # make a Location object corresponding to this TMY
        location = pvlib.location.Location(latitude=metadata['latitude'],
                                           longitude=metadata['longitude'])

        # TODO: change for handling HSAT tracking passed or requested
        if tilt is None:
            tilt = float(metadata['latitude'])

        # Calculate Sun Position
        # Note: TMY datasets are right-labeled hourly intervals, e.g. the
        # 10AM to 11AM interval is labeled 11.  We should calculate solar position in
        # the middle of the interval (10:30), so we subtract 30 minutes:
        times = df_tmy.index - pd.Timedelta('30min')
        solar_position = location.get_solarposition(times)
        # but remember to shift the index back to line up with the TMY data:
        solar_position.index += pd.Timedelta('30min')


        # Calculate POA_Global
        df_poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,  # tilted 20 degrees from horizontal
        surface_azimuth=azimuth,  # facing South
        dni=df_tmy['dni'],
        ghi=df_tmy['ghi'],
        dhi=df_tmy['dhi'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        model=skymodel)

        # Access the library of values for the SAPM (King's) Model
        all_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']

        #2. Calculate the module temperature with PVLIB, at infinity gap
        parameters = all_parameters['open_rack_glass_polymer']
        # note the "splat" operator "**" which expands the dictionary "parameters"
        # into a comma separated list of keyword arguments
        T1 = pvlib.temperature.sapm_cell(
                           poa_global=df_poa['poa_global'], temp_air=df_tmy['air_temperature'], wind_speed=df_tmy['wind_speed'],**parameters)


        #3. Calculate the module temperature with PVLIB, at 0 gap
        parameters = all_parameters['insulated_back_glass_polymer']
        T0 = pvlib.temperature.sapm_cell(
                           poa_global=df_poa['poa_global'], temp_air=df_tmy['air_temperature'], wind_speed=df_tmy['wind_speed'],**parameters)

        # Make the DataFrame
        results = pd.DataFrame({'timestamp':T1.index, 'T1':T1.values})
        results['T0'] = T0.values

        # Calculate the Quantiles
        T1p = results['T1'].quantile(q=0.98, interpolation='linear')
        T0p = results['T0'].quantile(q=0.98, interpolation='linear')

        x=-x0 * np.log(1-(T0p-T98)/(T0p-T1p))

        return x

    def calculate_T98Temperature(df_tmy, metadata, Tval=0.98, tilt=None, 
                                    azimuth=180, skymodel='isotropic',
                                    moduleandmountingtype = 'open_rack_glass_polymer'):
        '''
    
        Preliminary calculation for module gap according to IEC 63126. The
        steps of the calculation are:
        1. Takes a weather file
        2. Calculates the module temperature with PVLIB, for an open rack
           polymer-back which is assumed to have infinite gap
        3. Calculates the module temperature with PVLIB, for an insulated
            back module which is assumed to be flush mount with a roof (gap=0)
        4. Calculates the 98 percentile for step 2
        5. Calculates the 98 percentile for step 3
        5. Use both percentiles to calculate the effective gap "x" for the lower limit to use a
           Level 1 or Level 0 module (in IEC 63216), I.e. T98=80 or T98=70 respectively.

        Reference: M. Kempe, PVSC Proceedings 2023 (forthcoming)
        
        Parameters
        ----------
        df_tmy : DataFrame
            Dataframe with the weather data. Must have
            'air_temperature' and 'wind_speed'
        metadata : Dictionary
            must have 'latitude' and 'longitude'
        Tval : int
            Confidence interval value. Defalt T98 = 0.98
        tilt : float, optional
            Tilt of the PV array. If None, uses latitude. 
        azimuth : float, optional
            Azimuth of the PV array. The default is 180, facing south
        skymodel : str
            Tells PVlib which sky model to use. Default 'isotropic'.
        moduleandmountingtype : str
            To request sapm model parameters. Options: 
            'open_rack_glass_polymer' (default), 'open_rack_glass_glass',
            'close_mount_glass_glass', 'insulated_back_glass_polymer'

        Returns
        -------
        x : float
            Recommended installation distance per IEC 63126.
            effective gap "x" for the lower limit to use a Level 1 or Level 0 module (in IEC 63216)

        '''

        # 1. Calculate Sun Position & POA Irradiance

        # Make Location
        # make a Location object corresponding to this TMY
        location = pvlib.location.Location(latitude=metadata['latitude'],
                                           longitude=metadata['longitude'])

        # TODO: change for handling HSAT tracking passed or requested
        if tilt is None:
            tilt = float(metadata['latitude'])

        # Calculate Sun Position
        # Note: TMY/NSRDB datasets are right-labeled hourly intervals, e.g. the
        # 10AM to 11AM interval is labeled 11.  We should calculate solar position in
        # the middle of the interval (10:30), so we subtract 30 minutes:
        times = df_tmy.index - pd.Timedelta('30min')
        solar_position = location.get_solarposition(times)
        # but remember to shift the index back to line up with the TMY data:
        solar_position.index += pd.Timedelta('30min')


        # Calculate POA_Global
        df_poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,  # tilted 20 degrees from horizontal
        surface_azimuth=azimuth,  # facing South
        dni=df_tmy['dni'],
        ghi=df_tmy['ghi'],
        dhi=df_tmy['dhi'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        model=skymodel)

        # Access the library of values for the SAPM (King's) Model
        all_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']

        #2. Calculate the module temperature with PVLIB, at infinity gap
        parameters = all_parameters[moduleandmountingtype]
        # note the "splat" operator "**" which expands the dictionary "parameters"
        # into a comma separated list of keyword arguments
        T1 = pvlib.temperature.sapm_cell(
                           poa_global=df_poa['poa_global'], 
                           temp_air=df_tmy['air_temperature'], 
                           wind_speed=df_tmy['wind_speed'],**parameters)


        # Make the DataFrame
        results = pd.DataFrame({'timestamp':T1.index, 'T1':T1.values})

        # Calculate the Quantiles
        T1p = results['T1'].quantile(q=Tval, interpolation='linear')

        return T1p
