"""
Contains classes for calculating Stress Factors, Degradation, and installation Standards for PV Modules.

"""

import numpy as np
from numba import jit
import pandas as pd
from datetime import date
from datetime import datetime as dt
from scipy.constants import convert_temperature
from scipy.integrate import simpson
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import os
from PVDegradationTools import utilities as utils
import PVDegradationTools as PVD
import json

class StressFactors:
    """
    EnergyCalcs class contains the Vant Hoff acceleration factor and Arrhenius
    Equations Acceleration Factor

    """

    def k(avg_wvp):
        """
        This function generates a constant k, relating the average moisture ingress rate through a
        specific edge seal, Helioseal 101. Is an emperical estimation the rate of water ingress of
        water through edge seal material. This function was determined from numerical calculations
        from several locations and thus produces typical responses. This simplification works
        because the environmental temperature is not as important as local water vapor pressure.
        For the same environmental water concentration, a higher temperature results in lower
        absorption in the edge seal but lower diffusivity through the edge seal. In practice, these
        effects nearly cancel out makeing absolute humidity the primary parameter determining
        moisture ingress through edge seals.
        
        See: Kempe, Nobles, Postak Calderon,"Moisture ingress prediction in polyisobutylene‐based
        edge seal with molecular sieve desiccant", Progress in Photovoltaics, DOI: 10.1002/pip.2947

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
            With this constant, one can determine an approximate estimate of the ingress distance
            for a particular climate without more complicated numerical methods and detailed
            environmental analysis.

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

    def edge_seal_from_dew_pt(dew_pt_temp, all_results=False):
        """
        Compute the edge seal width required for 25 year water ingress directly from
        dew pt tempterature.

        Parameters
        ----------
        dew_pt_temp : float, or float series
            Dew Point Temperature
        all_results : boolean
            If true, returns all calculation steps: psat, avg_psat, k, edge seal width
            If false, returns only edge seal width

        Returns
        ----------
        edge_seal_width: float
            Width of edge seal [mm] required for 25 year water ingress

        Optional Returns
        ----------
        psat : series
            Hourly saturation point
        avg_psat : float
            Average saturation point over sample times
        k : float
            Ingress rate of water vapor
        """
        
        psat = StressFactors.psat(dew_pt_temp)
        avg_psat = psat.mean()

        k = .0013 * (avg_psat)**.4933

        edge_seal_width = StressFactors.edge_seal_width(k)

        if all_results:
            return {'psat':psat,
                    'avg_psat':avg_psat,
                    'k':k,
                    'edge_seal_width':edge_seal_width}

        return edge_seal_width

        

    # Numba Machine Language Level

    @jit(nopython=True, error_model='python')
    def dew_yield(elevation, dew_point, dry_bulb, wind_speed, n):
        """
        Estimates the dew yield in [mm/day].  Calculation taken from:
        Beysens, "Estimating dew yield worldwide from a few meteo data", Atmospheric Research 167
        (2016) 146-155

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
            This is a quasi emperical scale from 0 to 8 used in meterology which corresponds to
            0-sky completely clear, to 8-sky completely cloudy. Does not account for cloud type
            or thickness.

        Returns
        -------
        dew_yield : float
            Amount of dew yield in [mm/day]

        """
        wind_speed_cut_off = 4.4
        dew_yield = (1/12) * (.37 * (1 + (0.204323 * elevation) - (0.0238893 * elevation**2) -
                             (18.0132 - (1.04963 * elevation**2) + (0.21891 * elevation**2)) * (10**(-3) * dew_point)) *
                             ((((dew_point + 273.15) / 285)**4)*(1 - (n/8))) +
                             (0.06 * (dew_point - dry_bulb)) *
                             (1 + 100 * (1 - np.exp(- (wind_speed / wind_speed_cut_off)**20))))

        return dew_yield

    def psat(temp):
        """
        Function calculated the water saturation temperature or dew point for a given water vapor
        pressure. Water vapor pressure model created from an emperical fit of ln(Psat) vs
        temperature using a 6th order polynomial fit in microsoft Excel. The fit produced
        R^2=0.999813.
        Calculation created by Michael Kempe, unpublished data.

        #TODO:  verify this is consistant with psat in main branch (main is most up to date)
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
            (StressFactors.psat(temp_ambient) /
             StressFactors.psat(temp_module))

        return rh_Surface

        ###########
        # Front Encapsulant RH
        ###########
       
    def _diffusivity_weighted_water(rh_ambient, temp_ambient, temp_module,
                                    So=1.81390702,  Eas=16.729, Ead=38.14):
        """
        Calculation is used in determining a weighted average water content at the surface of a module.
        It is used as a constant water content that is equivalent to the time varying one with respect 
        to moisture ingress.

        The function calculates the Diffusivity weighted water content. 

        Parameters
        ----------
        rh_ambient : pandas series (float)
            The ambient outdoor environmnet relative humidity in [%]
            EXAMPLE: "50 = 50% NOT .5 = 50%"
        temp_ambient : pandas series (float)
            The ambient outdoor environmnet temperature in Celsius
        temp_module : pandas series (float)
            The surface temperature of the solar panel [°C]
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
            Diffusivity weighted water content [g/cm3]

        """

        # Get the relative humidity of the surface
        rh_surface = StressFactors.rh_surface_outside(
            rh_ambient, temp_ambient, temp_module)

        # Generate a series of the numerator values "prior to summation"
        numerator = So * np.exp(- (Eas / (0.00831446261815324 * (temp_module + 273.15))))\
                                * rh_surface * \
                                np.exp(- (Ead / (0.00831446261815324 * (temp_module + 273.15))))

        # get the summation of the numerator
        numerator = numerator.sum(axis=0, skipna=True)

        denominator = np.exp(- (Ead / (0.00831446261815324 * (temp_module + 273.15))))
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
            ambient Relative Humidity [%]
        temp_ambient : series (float)
            ambient outdoor temperature [°C]        
        temp_module : pandas series (float)
            The surface temperature in Celsius of the solar module
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
            Saturation of Water Concentration [g/cm³]

        """

        # Saturation of water concentration
        Csat = So * \
            np.exp(- (Eas / (0.00831446261815324 * (273.15 + temp_module))))

        return Csat

    def _ceq(Csat, rh_SurfaceOutside):
        """
        Calculation is used in determining Relative Humidity of Backside Solar
        Module Encapsulant, and returns Equilibration water concentration [g/cm³]

        Parameters
        ------------
        Csat : pandas series (float)
            Saturation of Water Concentration [g/cm³]
        rh_SurfaceOutside : pandas series (float)
            The relative humidity of the surface of a solar module [%]

        Returns
        --------
        Ceq : pandas series (float)
            Equilibration water concentration [g/cm³]

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
            The relative humidity of the surface of a solar module [%]
            EXAMPLE: "50 = 50% NOT .5 = 50%"
        WVTRo : float
            Water Vapor Transfer Rate prefactor (g/m2/day).
            The suggested value for EVA is WVTRo = 7970633554(g/m2/day).
        EaWVTR : float
            Water Vapor Transfer Rate activation energy [kJ/mol] .
            It is suggested to use 0.15(mm) thick PET as a default
            for the backsheet and set EaWVTR=55.0255(kJ/mol)
        So : float
            Encapsulant solubility prefactor in [g/cm3]
            So = 1.81390702(g/cm3) is the suggested value for EVA.
        l : float
            Thickness of the backside encapsulant [mm].
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
            The ambient outdoor environmnet relative humidity in [%]
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
            Water Vapor Transfer Rate activation energy [kJ/mol] .
            It is suggested to use 0.15(mm) thick PET as a default
            for the backsheet and set EaWVTR=55.0255(kJ/mol)
        So : float
            Encapsulant solubility prefactor in [g/cm3]
            So = 1.81390702(g/cm3) is the suggested value for EVA.
        l : float
            Thickness of the backside encapsulant [mm].
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
            The ambient outdoor environmnet relative humidity in [%]
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
            Water Vapor Transfer Rate activation energy [kJ/mol] .
            It is suggested to use 0.15(mm) thick PET as a default
            for the backsheet and set EaWVTR=55.0255(kJ/mol)
        So : float
            Encapsulant solubility prefactor in [g/cm3]
            So = 1.81390702(g/cm3) is the suggested value for EVA.
        l : float
            Thickness of the backside encapsulant [mm].
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
            The ambient outdoor environmnet relative humidity in [%]
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
            Water Vapor Transfer Rate activation energy [kJ/mol] .
            It is suggested to use 0.15(mm) thick PET as a default
            for the backsheet and set EaWVTR=55.0255(kJ/mol)
        So : float
            Encapsulant solubility prefactor in [g/cm3]
            So = 1.81390702(g/cm3) is the suggested value for EVA.
        l : float
            Thickness of the backside encapsulant [mm].
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
            Degredation Activation Energy [kJ/mol]

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
            Relative Humidity of Controlled Condition [%]
            EXAMPLE: "50 = 50% NOT .5 = 50%"
        temp_chamber : float
            Reference temperature [°C] "Chamber Temperature"
        Ea : float
            Degredation Activation Energy [kJ/mol]
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
            Relative Humidity of Controlled Condition [%].
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
            Degredation Activation Energy [kJ/mol]
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
            Degredation Activation Energy [kJ/mol]

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
            Degredation Activation Energy [kJ/mol]
        Teq : series
            Equivalent Arrhenius temperature [°C]
        n : float
            Fit parameter for relative humidity

        Returns
        --------
        RHwa : float
            Relative Humidity Weighted Average [%]

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
            Degradation Activation Energy [kJ/mol]
        RHwa : float
            Relative Humidity Weighted Average [%]
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


class Scenario:
    """
    The scenario object contains all necessary parameters and criteria for a given scenario.
    Generally speaking, this will be information such as:
    Scenario Name, Path, Geographic Location, Module Type, Racking Type
    """

    def __init__(self, name=None, path=None, gids=None, modules=[], pipeline=[],
                 hpc=False, file=None) -> None:
        """
        Initialize the degradation scenario object.

        Parameters:
        -----------
        name : (str)
            custom name for deg. scenario. If none given, will use date of initialization (DDMMYY)
        path : (str, pathObj)
            File path to operate within and store results. If none given, new folder "name" will be
            created in the working directory.
        gids : (str, pathObj)
            Spatial area to perform calculation for. This can be Country or Country and State.
        modules : (list, str)
            List of module names to include in calculations.
        pipeline : (list, str)
            List of function names to run in job pipeline
        file : (path)
            Full file path to a pre-generated Scenario object. If specified, all other parameters
            will be ignored and taken from the .json file.
        """
        
        if file is not None:
            with open(file,'r') as f:
                data = json.load()
            name = data['name']
            path = data['path']
            modules = data['modules']
            gids = data['gids']
            pipeline = data['pipeline']
        
        self.name = name
        self.path = path
        self.modules = modules
        self.gids = gids
        self.pipeline = pipeline

        filedate = dt.strftime(date.today(), "%d%m%y")

        if name is None:
            name = filedate
        self.name = name

        if path is None:
            self.path = os.path.join(os.getcwd(),f'pvd_job_{self.name}')
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        os.chdir(self.path)
    
    def addLocation(self, weather_fp=None, region=None, region_col='state',
                    lat_long=None, gids=None):
        """
        Add a location to the scenario. Generates "gids.csv" and saves the file path within
        Scenario dictionary. This can be done in three ways: Pass (region, region_col) for gid list,
        pass (gid) for a single location, pass (lat, long) for a single location.

        Parameters:
        -----------
        weather_fp : (str, path_obj)
            File path to the source dataframe for weather and spatial data. Default should be NSRDB
        region : (str)
            Region or state to iterate over
        region_col : (str)
            Region column name within h5 file (example "State")
        lat : (tuple - float)
            latitute and longitude of a single location
        """
        
        if self.gids is not None:
            print('Scenario already has designated project points.\nNothing has been added.')
            print(self.gids)
            return

        if not weather_fp:
            weather_fp = r'/datasets/NSRDB/current/nsrdb_tmy-2021.h5'
        
        file_name = f'gids_{self.name}'
        gids_path = utils.write_gids(weather_fp,
                                     region=region,
                                     region_col=region_col,
                                     lat_long=lat_long,
                                     gids=gids,
                                     out_fn=file_name)
       
        self.gids = gids_path
        print(f'Location Added - {self.gids}')
    
    def addModule(self,
                  module_name,
                  racking='open_rack_glass_polymer', #move ?? split RACKING_CONSTRUCTION
                  material='EVA'):
        """
        Add a module to the Scenario. Multiple modules can be added. Each module will be tested in
        the given scenario.

        Parameters:
        -----------
        module_name : (str)
            unique name for the module. adding multiple modules of the same name will replace the
            existing entry.
        racking : (str)
            temperature model racking type as per PVLIB (see pvlib.temperature). Allowed entries:
            'open_rack_glass_glass', 'open_rack_glass_polymer',
            'close_mount_glass_glass', 'insulated_back_glass_polymer'
        material : (str)
            Name of the material desired. For a complete list, see data/materials.json.
            To add a custom material, see PVDegradationTools.addMaterial (ex: EVA, Tedlar)
        """

        # fetch material parameters (Eas, Ead, So, etc)
        try:
            mat_params = utils._read_material(name=material)
        except:
            print('Material Not Found - No module added to scenario.')
            print('If you need to add a custom material, use .add_material()')
            return

        # remove module if found in instance list
        for i in range(self.modules.__len__()):
            if self.modules[i]['module_name'] == module_name:
                print(f'WARNING - Module already found by name "{module_name}"')
                print('Module will be replaced with new instance.')
                self.modules.pop(i)
        
        # generate temperature model params
        # TODO: move to temperature based functions
        # temp_params = TEMPERATURE_MODEL_PARAMETERS[model][racking]
            
        # add the module and parameters
        self.modules.append({'module_name':module_name,
                             'material_params':mat_params})
        print(f'Module "{module_name}" added.')

    def add_material(self,name, alias, Ead, Eas, So, Do=None, Eap=None, Po=None, fickian=True):
        """
        add a new material type to master list
        """
        utils._add_material(name=name, alias=alias,
                            Ead=Ead, Eas=Eas, So=So,
                            Do=Do, Eap=Eap, Po=Po, fickian=fickian)
        print('Material has been added.')
        print('To add the material as a module in your current scene, run .addModule()')

    def viewScenario(self):
        '''
        Print all scenario information currently stored
        '''

        import pprint
        pp = pprint.PrettyPrinter(indent=4,sort_dicts=False)
        print(f'Name : {self.name}')
        print(f'pipeline: {self.pipeline}')
        print(f'gid file : {self.gids}')
        print('test modules :')
        for mod in self.modules:
            pp.pprint(mod)
        return

    def addFunction(self, func_name=None, func_params=None):
        """
        Add a PVD function to the scenario pipeline

        TODO: list public functions if no func_name given or bad func_name given

        Parameters:
        -----------
        func_name : (str)
            The name of the requested PVD function. Do not include the class.
        func_params : (dict)
            The required parameters to run the requested PVD function
        
        Returns:
        --------
        func_name : (str)
            the name of the PVD function requested
        """

        _func, reqs = PVD.Scenario._verify_function(func_name)
        
        if _func == None:
            print(f'FAILED: Requested function "{func_name}" not found')
            print('Function has not been added to pipeline.')
            return None
        
        if not all( x in func_params for x in reqs):
            print(f'FAILED: Requestion function {func_name} did not receive enough parameters')
            print(f'Requestion function: \n {_func} \n ---')
            print(f'Required Parameters: \n {reqs} \n ---')
            print('Function has not been added to pipeline.')
            return None

        # add the function and arguments to pipeline
        job_dict = {'job':func_name,
                    'params':func_params}                
        
        self.pipeline.append(job_dict)
        return func_name
    
    def runJob(self, job=None):
        '''
        Run a named function on the scenario object
        
        TODO: overhaul with futures/slurm
              capture results
              standardize result format for all of PVD

        Parameters:
        -----------
        job : (str, default=None)
        '''
        if self.hpc:
            # do something else
            pass
        
        for job in self.pipeline:
            args = job['parameters']
            _func = PVD.Scenario._verify_function(job['job'],args)[0]
            result = _func(**args)

    def exportScenario(self, file_path=None):
        '''
        Export the scenario dictionaries to a json configuration file
        
        TODO exporting functions as name string within pipeline. cannot .json dump <PVD.func>
             Need to make sure name is verified > stored > export > import > re-verify > converted.
             This could get messy. Need to streamline the process or make it bullet proof
        
        Parameters:
        -----------
        file_path : (str, default = None)
            Desired file path to save the scenario.json file
        '''
        
        if not file_path:
            file_path = self.path
        file_name = f'config_{self.name}.json'
        out_file = os.path.join(file_path,file_name) 

        scene_dict = {'name': self.name,
                      'path': self.path,
                      'pipeline': self.pipeline,
                      'gid_file': self.gids,
                      'test_modules': self.modules}
        
        with open(out_file, 'w') as f:
            json.dump(scene_dict, f, indent=4)
        print(f'{file_name} exported')
    
    def importScenario(self, file_path=None):
        """
        Import scenario dictionaries from an existing 'scenario.json' file
        """
        
        with open(file_path,'r') as f:
            data = json.load()
        name = data['name']
        path = data['path']
        modules = data['modules']
        gids = data['gids']
        pipeline = data['pipeline']

        self.name = name
        self.path = path
        self.modules = modules
        self.gids = gids
        self.pipeline = pipeline
    
    def _verify_function(func_name):
        """
        Check all classes in PVD for a function of the name "func_name". Returns a callable function
        and list of all function parameters with no default values.
        
        Parameters:
        -----------
        func_name : (str)
            Name of the desired function. Only returns for 1:1 matches
        
        Returns:
        --------
        _func : (func)
            callable instance of named function internal to PVD
        reqs : (list(str))
            list of minimum required paramters to run the requested funciton
        """
        from inspect import signature
        
        # find the function in PVD
        class_list = [c for c in dir(PVD) if not c.startswith('_')]
        func_list = []
        for c in class_list:
            _class = getattr(PVD,c)
            if func_name in dir(_class):
                _func = getattr(_class,func_name)
        if _func == None:
            return (None,None)

        # check if necessary parameters given
        reqs_all = signature(_func).parameters
        reqs = []
        for param in reqs_all:
            if reqs_all[param].default == reqs_all[param].empty:
                reqs.append(param)
        
        return(_func, reqs)
