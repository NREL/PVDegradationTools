"""
Contains energy algorithms for processing.

@author: Derek Holsapple NREL 2020
"""

import numpy as np
from numba import jit
import pandas as pd
from scipy.constants import convert_temperature
import pvlib

class EnergyCalcs:
    """
    EnergyCalcs class contains the Vant Hoff acceleration factor and Arrhenius
    Equations Acceleration Factor

    """

    def k(avg_wvp):
        """
        Determine the rate of water ingress of water through edge seal material

        Parameters
        -----------
        avg_wvp : float
            Average of the Yearly water vapor
            pressure for 1 year

        Returns
        -------
        k : float
            Ingress rate of water through edge seal

        """

        k = .0013 * (avg_wvp)**.4933

        return k

    def edge_seal_width(k):
        """
        Determine the width of edge seal required for a 25 year water ingress

        Parameters
        ----------
        k: float
            Rate of water ingress

        Returns
        ----------
        width : float
            Width of edge seal required for a 25 year water ingress (mm)
        """

        width = k * (25 * 365.25 * 24)**.5

        return width


############
# Dew Yield
############
    # Numba Machine Language Level

    @jit(nopython=True, error_model='python')
    def dew_yield(h, tD, tA, wind_speed, n):
        """
        Find the dew yield in (mm·d−1).  Calculation taken from journal
        "Estimating dew yield worldwide from a few meteo data"
            -D. Beysens

        (ADD IEEE reference)

        Parameters
        -----------
        h : int
            Site elevation in kilometers
        tD : float
            Dewpoint temperature in Celsius
        tA : float
            Air temperature "dry bulb temperature"
        wind_speed : float
            Air or windspeed measure in m*s^-1  or m/s
        n : float
            Total sky cover(okta)

        Returns
        -------
        dew_yield : float
            Amount of dew yield in (mm·d−1)

        """
        wind_speed_cut_off = 4.4
        dew_yield = (1/12) * (.37 * (1 + (0.204323 * h) - (0.0238893 * h**2) -
                             (18.0132 - (1.04963 * h**2) + (0.21891 * h**2)) * (10**(-3) * tD)) *
                             ((((tD + 273.15) / 285)**4)*(1 - (n/8))) +
                             (0.06 * (tD - tA)) *
                             (1 + 100 * (1 - np.exp(- (wind_speed / wind_speed_cut_off)**20))))

        return dew_yield

############
# Water Vapor Pressure
############

    def water_vapor_pressure(dew_pt_temp):
        """
        Find the average water vapor pressure (kPa) based on the Dew Point
        Temperature model created from Mike Kempe on 10/07/19 from Miami,FL excel sheet.

        Parameters
        ----------
        dew_pt_temp : float
            Dew Point Temperature

        Returns
        --------
        watervaporpressure : float
            Water vapor pressure in kPa

        """
        water_vapor_pressure = (np.exp((3.257532E-13 * dew_pt_temp**6) -
                                     (1.568073E-10 * dew_pt_temp**6) +
                                     (2.221304E-08 * dew_pt_temp**4) +
                                     (2.372077E-7 * dew_pt_temp**3) -
                                     (4.031696E-04 * dew_pt_temp**2) +
                                     (7.983632E-02 * dew_pt_temp) -
                                     (5.698355E-1)))

        return water_vapor_pressure

############
# Solder Fatigue
############

    def _avg_daily_temp_change(local_time, cell_temp):
        """
        Helper function. Get the average of a year for the daily maximum temperature change.

        For every 24hrs this function will find the delta between the maximum
        temperature and minimun temperature.  It will then take the deltas for
        every day of the year and return the average delta.

        Parameters
        ------------
        local_time : timestamp series
            Local time of specific site by the hour
            year-month-day hr:min:sec . (Example) 2002-01-01 01:00:00
        cell_temp : float series
            Photovoltaic module cell temperature(Celsius) for every hour of a year

        Returns
        -------
        avg_daily_temp_change : float
            Average Daily Temerature Change for 1-year (Celsius)
        avg_max_cell_temp : float
            Average of Daily Maximum Temperature for 1-year (Celsius)

        """
        # Setup frame for vector processing
        timeAndTemp_df = pd.DataFrame(columns=['Cell Temperature'])
        timeAndTemp_df['Cell Temperature'] = cell_temp
        timeAndTemp_df.index = local_time
        timeAndTemp_df['month'] = timeAndTemp_df.index.month
        timeAndTemp_df['day'] = timeAndTemp_df.index.day

        # Group by month and day to determine the max and min cell Temperature (C) for each day
        dailyMaxCellTemp_series = timeAndTemp_df.groupby(
            ['month', 'day'])['Cell Temperature'].max()
        dailyMinCellTemp_series = timeAndTemp_df.groupby(
            ['month', 'day'])['Cell Temperature'].min()
        cell_temp_change = pd.DataFrame(
            {'Max': dailyMaxCellTemp_series, 'Min': dailyMinCellTemp_series})
        cell_temp_change['TempChange'] = cell_temp_change['Max'] - \
            cell_temp_change['Min']

        # Find the average temperature change for every day of one year (C)
        avg_daily_temp_change = cell_temp_change['TempChange'].mean()
        # Find daily maximum cell temperature average
        avg_max_cell_temp = dailyMaxCellTemp_series.mean()

        return avg_daily_temp_change, avg_max_cell_temp

    def _times_over_reversal_number(cell_temp, reversal_temp):
        """
        Helper function. Get the number of times a temperature increases or decreases over a
        specific temperature gradient.

        Parameters
        ------------
        cell_temp : float series
            Photovoltaic module cell temperature(Celsius)
        reversal_temp : float
            Temperature threshold to cross above and below

        Returns
        --------
        num_changes_temp_hist : int
            Number of times the temperature threshold is crossed

        """
        # Find the number of times the temperature crosses over 54.8(C)

        temp_df = pd.DataFrame()
        temp_df['CellTemp'] = cell_temp
        temp_df['COMPARE'] = cell_temp
        temp_df['COMPARE'] = temp_df.COMPARE.shift(-1)

        #reversal_temp = 54.8

        temp_df['cross'] = (
            ((temp_df.CellTemp >= reversal_temp) & (temp_df.COMPARE < reversal_temp)) |
            ((temp_df.COMPARE > reversal_temp) & (temp_df.CellTemp <= reversal_temp)) |
            (temp_df.CellTemp == reversal_temp))

        num_changes_temp_hist = temp_df.cross.sum()

        return num_changes_temp_hist

    def solderFatigue(local_time, cell_temp, reversal_temp):
        """
        Get the Thermomechanical Fatigue of flat plate photovoltaic module solder joints.
        Damage will be returned as the rate of solder fatigue for one year. Based on:

            Bosco, N., Silverman, T. and Kurtz, S. (2020). Climate specific thermomechanical
            fatigue of flat plate photovoltaic module solder joints. [online] Available
            at: https://www.sciencedirect.com/science/article/pii/S0026271416300609
            [Accessed 12 Feb. 2020].

        Parameters
        ------------
        local_time : timestamp series
            Local time of specific site by the hour year-month-day hr:min:sec
            (Example) 2002-01-01 01:00:00
        cell_temp : float series
            Photovoltaic module cell temperature(Celsius) for every hour of a year
        reversal_temp : float
            Temperature threshold to cross above and below

        Returns
        --------
        damage : float series
            Solder fatigue damage for a time interval depending on local_time (kPa)

        """

        # TODO Make this function have more utility.
        # People want to run all the scenarios from the bosco paper.
        # Currently have everything hard coded for hourly calculation
        # i.e. 405.6, 1.9, .33, .12
        # Get the:
        #   1) Average of the Daily Maximum Cell Temperature (C)
        #   2) Average of the Daily Maximum Temperature change avg(daily max - daily min)
        #   3) Number of times the temperaqture crosses above or below the reversal Temperature
        MeanDailyMaxCellTempChange, dailyMaxCellTemp_Average = EnergyCalcs._avg_daily_temp_change(
            local_time, cell_temp)
        # Needs to be in Kelvin for equation specs
        dailyMaxCellTemp_Average = convert_temperature(
            dailyMaxCellTemp_Average, 'Celsius', 'Kelvin')
        num_changes_temp_hist = EnergyCalcs._times_over_reversal_number(
            cell_temp, reversal_temp)

        # k = Boltzmann's Constant
        damage = 405.6 * (MeanDailyMaxCellTempChange ** 1.9) * \
                         (num_changes_temp_hist**.33) * \
            np.exp(-(.12/(.00008617333262145*dailyMaxCellTemp_Average)))
        # Convert pascals to kilopascals
        damage = damage/1000
        return damage

    def _power(cell_temp, poa_global):
        """
        TODO:   check units
                check 2 different functions
        Helper function. Find the relative power produced from a solar module.

        Model derived from Mike Kempe Calculation on paper
        (ADD IEEE reference)

        Parameters
        ------------
        cell_temp : float
            Cell Temperature of a solar module (C)
        poa_global : float
            plane-of-array irradiance, global. (W/m^2) 

        Returns
        --------
        power : float
            Power produced from a module in KW/hours
        """
        # KW/hr

        # Why is there two definitions?
        power = 0.0002 * poa_global * (1 + (25 - cell_temp) * .004)
        power = poa_global * (1 + (25 - cell_temp) * .004)

        return power


################################################################################

    ############################################
    # Vant Hoff Degradation Function
    ############################################


    def _deg_rate_env(poa_global, t_outdoor, ref_temp, x, Tf):
        """
        Helper function. Find the rate of degradation kenetics using the Fischer model.
        Degradation kentics model interpolated 50 coatings with respect to
        color shift, cracking, gloss loss, fluorescense loss,
        retroreflectance loss, adhesive transfer, and shrinkage.

        (ADD IEEE reference)

        Parameters
        ------------
        poa_global : float
            (Global) Plan of Array irradiance (W/m^2)
        t_outdoor : float
            Solar module cell temperature (C)
        ref_temp : float
            Reference temperature (C) "Chamber Temperature"
        x : float
            Fit parameter
        Tf : float
            Multiplier for the increase in degradation
                                          for every 10(C) temperature increase

        Returns
        --------
        degradationrate : float
            rate of Degradation (NEED TO ADD METRIC)

        """
        return poa_global**(x) * Tf ** ((t_outdoor - ref_temp)/10)

    def _deg_rate_chamber(I_chamber, x):
        """
        Helper function. Find the rate of degradation kenetics of a simulated chamber. Mike Kempe's
        calculation of the rate of degradation inside a accelerated degradation chamber.

        (ADD IEEE reference)

        Parameters
        ----------
        I_chamber : float
            Irradiance of Controlled Condition W/m^2
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

    def vantHoff_deg(I_chamber, poa_global, t_outdoor, ref_temp, x=0.64, Tf=1.41):
        """
        NOTE

        Vant Hoff Irradiance Degradation

        Parameters
        -----------
        I_chamber : float
            Irradiance of Controlled Condition W/m^2
        poa_global : float or series
            Global Plane of Array Irradiance W/m^2
        t_outdoor : pandas series
            Solar module temperature or Cell temperature (C)
        ref_temp : float
            Reference temperature (C) "Chamber Temperature"
        x : float
            fit parameter
        Tf : float
            Multiplier for the increase in degradation for every 10(C) temperature increase

        Returns
        -------
        accelerationFactor : float or series
            Degradation acceleration factor

        """
        rateOfDegEnv = EnergyCalcs._deg_rate_env(poa_global=poa_global,
                                                 t_outdoor=t_outdoor,
                                                 ref_temp=ref_temp,
                                                 x=x,
                                                 Tf=Tf)
        #sumOfDegEnv = rateOfDegEnv.sum(axis = 0, skipna = True)
        avgOfDegEnv = rateOfDegEnv.mean()

        rateOfDegChamber = EnergyCalcs._deg_rate_chamber(I_chamber, x)

        accelerationFactor = EnergyCalcs._acceleration_factor(
            rateOfDegChamber, avgOfDegEnv)

        return accelerationFactor


##############################################################################################
    ############################################
    # Vant Hoff Environmental Characterization
    ############################################

    def _to_eq_vantHoff(t_outdoor, Tf=1.41):
        """
        Function to obtain the Vant Hoff temperature equivalent (C)

        Parameters
        ----------
        Tf : float
            Multiplier for the increase in degradation for every 10(C) temperature increase
        t_outdoor : pandas series
            Solar module temperature or Cell temperature (C)

        Returns
        -------
        Toeq : float
            Vant Hoff temperature equivalent (C)

        """
        toSum = Tf ** (t_outdoor / 10)
        summation = toSum.sum(axis=0, skipna=True)

        Toeq = (10 / np.log(Tf)) * np.log(summation / len(t_outdoor))

        return Toeq


    def IwaVantHoff(poa_global, t_outdoor, Teq=None, x=0.64, Tf=1.41):
        """
        NOTE

        IWa : Environment Characterization (W/m^2)
        *for one year of degredation the controlled environmnet lamp settings will
            need to be set to IWa

        Parameters
        -----------
        poa_global : float or series
            Global Plane of Array Irradiance W/m^2
        t_outdoor : pandas series
            Solar module temperature or Cell temperature (C)
        Teq : series
            VantHoff equivalent temperature (C)
        x : float
            Fit parameter
        Tf : float
            Multiplier for the increase in degradation for every 10(C) temperature increase

        Returns
        --------
        Iwa : float
            Environment Characterization (W/m^2)

        """
        if Teq is None:
            Teq = EnergyCalcs._to_eq_vantHoff(t_outdoor, Tf)
        toSum = (poa_global ** x) * (Tf ** ((t_outdoor - Teq)/10))
        summation = toSum.sum(axis=0, skipna=True)

        Iwa = (summation / len(poa_global)) ** (1 / x)

        return Iwa


##############################################################################################
    ############################################
    # Arrhenius Degradation Function
    ############################################


    def _arrhenius_denominator(poa_global, rh_outdoor, t_outdoor, Ea, x, n):
        """
        Helper function. Calculates the rate of degredation of the Environmnet

        Parameters
        ----------
        poa_global : float
            (Global) Plan of Array irradiance (W/m^2)
        x : float
            Fit parameter
        rh_outdoor : pandas series
            Relative Humidity of material of interest. Acceptable relative
            humiditys can be calculated from these functions: RHbacksheet(),
            RHbackEncap(); RHfrontEncap();  RHsurfaceOutside()
        n : float
            Fit parameter for relative humidity
        t_outdoor : pandas series
            Solar module temperature or Cell temperature (C)
        Ea : float
            Degredation Activation Energy (kJ/mol)

        Returns
        -------
        environmentDegradationRate : pandas series
            Degradation rate of environment
        """

        environmentDegradationRate = poa_global**(x) * rh_outdoor**(
            n) * np.exp(- (Ea / (0.00831446261815324 * (t_outdoor + 273.15))))

        return environmentDegradationRate

    def _arrhenius_numerator(I_chamber, rh_chamber,  T_chamber, Ea, x, n):
        """
        Helper function. Find the rate of degradation of a simulated chamber.

        Parameters
        ----------
        I_chamber : float
            Irradiance of Controlled Condition W/m^2
        Rhchamber : float
            Relative Humidity of Controlled Condition (%)
            EXAMPLE: "50 = 50% NOT .5 = 50%"
        T_chamber : float
            Reference temperature (C) "Chamber Temperature"
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
                                              (T_chamber+273.15)))))
        return arrheniusNumerator

    def arrheniusCalc(I_chamber, rh_chamber, rh_outdoor, poa_global, T_chamber, t_outdoor,
                        Ea, x=0.64, n=1):
        """
        NOTE

        Calculate the Acceleration Factor between the rate of degredation of a
        modeled environmnet versus a modeled controlled environmnet

        Example: "If the AF=25 then 1 year of Controlled Environment exposure
                    is equal to 25 years in the field"

        Parameters
        ----------
        I_chamber : float
            Irradiance of Controlled Condition W/m^2
        rh_chamber : float
            Relative Humidity of Controlled Condition (%).
            EXAMPLE: "50 = 50% NOT .5 = 50%"
        rh_outdoor : pandas series
            Relative Humidity of material of interest
            Acceptable relative humiditys can be calculated
            from these functions: RHbacksheet(), RHbackEncap(), RHfrontEncap(),
            RHsurfaceOutside()
        poa_global : pandas series
            Global Plane of Array Irradiance W/m^2
        T_chamber : float
            Reference temperature (C) "Chamber Temperature"
        t_outdoor : pandas series
            Solar module temperature or Cell temperature (C)
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
        arrheniusDenominator = EnergyCalcs._arrhenius_denominator(poa_global=poa_global,
                                                                 rh_outdoor=rh_outdoor,
                                                                 t_outdoor=t_outdoor,
                                                                 Ea=Ea,
                                                                 x=x,
                                                                 n=n)

        AvgOfDenominator = arrheniusDenominator.mean()

        arrheniusNumerator = EnergyCalcs._arrhenius_numerator(I_chamber=I_chamber, 
                                                             rh_chamber=rh_chamber,
                                                             T_chamber=T_chamber, Ea=Ea, x=x, n=n)

        accelerationFactor = EnergyCalcs._acceleration_factor(
            arrheniusNumerator, AvgOfDenominator)

        return accelerationFactor


###############################################################################
    ############################################
    # Arrhenius Environmental Characterization
    ############################################

    def _T_eq_arrhenius(t_outdoor, Ea):
        """
        Get the Temperature equivalent required for the settings of the controlled environment
        Calculation is used in determining Arrhenius Environmental Characterization

        Parameters
        -----------
        t_outdoor : pandas series
            Solar module temperature or Cell temperature (C)
        Ea : float
            Degredation Activation Energy (kJ/mol)

        Returns
        -------
        Teq : float
            Temperature equivalent (Celsius) required
            for the settings of the controlled environment

        """

        summationFrame = np.exp(- (Ea /
                                   (0.00831446261815324 * (t_outdoor + 273.15))))
        sumForTeq = summationFrame.sum(axis=0, skipna=True)
        Teq = -((Ea) / (0.00831446261815324 * np.log(sumForTeq / len(t_outdoor))))
        # Convert to celsius
        Teq = Teq - 273.15

        return Teq

    def _RH_wa_arrhenius(rh_outdoor, t_outdoor, Ea, Teq=None, n=1):
        """
        NOTE

        Get the Relative Humidity Weighted Average.
        Calculation is used in determining Arrhenius Environmental Characterization

        Parameters
        -----------
        rh_outdoor : pandas series
            Relative Humidity of material of interest. Acceptable relative
            humiditys can be calculated from the below functions:
            RHbacksheet(), RHbackEncap(), RHfrontEncap(), RHsurfaceOutside()
        t_outdoor : pandas series
            solar module temperature or Cell temperature (C)
        Ea : float
            Degredation Activation Energy (kJ/mol)
        Teq : series
            Equivalent Arrhenius temperature (C)
        n : float
            Fit parameter for relative humidity

        Returns
        --------
        RHwa : float
            Relative Humidity Weighted Average (%)

        """

        if Teq is None:
            Teq = EnergyCalcs._T_eq_arrhenius(t_outdoor, Ea)

        summationFrame = (rh_outdoor ** n) * np.exp(- (Ea /
                                                      (0.00831446261815324 * (t_outdoor + 273.15))))
        sumForRHwa = summationFrame.sum(axis=0, skipna=True)
        RHwa = (sumForRHwa / (len(summationFrame) * np.exp(- (Ea /
                                                (0.00831446261815324 * (Teq + 273.15)))))) ** (1/n)

        return RHwa

    def RHwaArrhenius(rh_outdoor, t_outdoor, Ea, Teq=None, n=1):
        """
        NOTE

        Get the Relative Humidity Weighted Average.
        Calculation is used in determining Arrhenius Environmental Characterization

        Parameters
        -----------
        rh_outdoor : pandas series
            Relative Humidity of material of interest. Acceptable relative
            humiditys can be calculated from the below functions:
            RHbacksheet(), RHbackEncap(), RHfrontEncap(), RHsurfaceOutside()
        t_outdoor : pandas series
            solar module temperature or Cell temperature (C)
        Teq : float
            Temperature equivalent (Celsius) required
            for the settings of the controlled environment
        Ea : float
            Degredation Activation Energy (kJ/mol)
        n : float
            Fit parameter for relative humidity

        Returns
        --------
        RHwa : float
            Relative Humidity Weighted Average (%)

        """

        if Teq is None:
            Teq = EnergyCalcs._T_eq_arrhenius(t_outdoor, Ea)

        summationFrame = (rh_outdoor ** n) * np.exp(- (Ea /
                                                      (0.00831446261815324 * (t_outdoor + 273.15))))
        sumForRHwa = summationFrame.sum(axis=0, skipna=True)
        RHwa = (sumForRHwa / (len(summationFrame) * np.exp(- (Ea /
                                                (0.00831446261815324 * (Teq + 273.15)))))) ** (1/n)

        return RHwa

    def IwaArrhenius(poa_global, rh_outdoor, t_outdoor, Ea,
                     RHwa=None, Teq=None, x=0.64, n=1):
        """
        TODO:   CHECK
                STANDARDIZE

        Function to calculate IWa, the Environment Characterization (W/m^2)
        *for one year of degredation the controlled environmnet lamp settings will
            need to be set at IWa

        Parameters
        ----------
        poa_global : float
            (Global) Plan of Array irradiance (W/m^2)
        rh_outdoor : pandas series
            Relative Humidity of material of interest
            Acceptable relative humiditys can be calculated
            from these functions: RHbacksheet(), RHbackEncap(), RHfrontEncap()
                                  RHsurfaceOutside()
        t_outdoor : pandas series
            Solar module temperature or Cell temperature (C)
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
            Environment Characterization (W/m^2)

        """
        if Teq is None:
            Teq = EnergyCalcs._T_eq_arrhenius(t_outdoor, Ea)

        if RHwa is None:
            RHwa = EnergyCalcs._RH_wa_arrhenius(rh_outdoor, t_outdoor, Ea)

        numerator = poa_global**(x) * rh_outdoor**(n) * \
            np.exp(- (Ea / (0.00831446261815324 * (t_outdoor + 273.15))))
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
        booleanDf = df.apply(lambda x: EnergyCalcs._rh_Above85(x))
        numhoursabove85 = booleanDf.sum()

        return numhoursabove85

    def _whToGJ(wh):
        """
        Helper Function to convert Wh/m^2 to GJ/m^-2

        Parameters
        -----------
        wh : float
            Input Value in Wh/m^2

        Returns
        -------
        gj : float
            Value in GJ/m^-2

        """

        gj = 0.0000036 * wh

        return gj

    def _gJtoMJ(gJ):
        """
        Helper Function to convert GJ/m^-2 to MJ/y^-1

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


class relativeHumidity:
    """
    There are currently 4 selections for relative Humidity in Class relativeHumidity:

    1) RHsurfaceOutside : Relative Humidity of the Surface of a Solar Module
    2) RHfrontEncapsulant : Relative Humidity of the Frontside Encapsulant of a Solar Module
    3) RHbackEncapsulant : Relative Humidity of the backside Encapsulant of a Solar Module
    4) RHbacksheet : Relative

    """

    ###########
    # Surface RH
    ###########

    def Psat(temp):
        """
        Function to generate the point of saturation dependent on temperature
        Calculation created by Michael Kempe, implemented by Derek Holsapple

        3rd, 4th, 5th, and 6th order polynomial fits were explored.  The best fit
        was determined to be the 4th

        Parameters
        -----------
        temp : float
            Temperature in Celsius

        Returns
        -------
        Psat : float
            Point of saturation

        """

        Psat = np.exp(-0.000000002448137*temp**4
                      + 0.000001419572*temp**3
                      - 0.0003779559*temp**2
                      + 0.07796986*temp
                      - 0.5796729)

        return Psat

    def RHsurfaceOutside(rh_ambient, ambient_temp, surface_temp):
        """
        Function calculates the Relative Humidity of a Solar Panel Surface

        Parameters
        ----------
        rh_ambient : float
            The ambient outdoor environmnet relative humidity
        ambient_temp : float
            The ambient outdoor environmnet temperature in Celsius
        surface_temp : float
            The surface temperature in Celsius of the solar panel module

        Returns
        --------
        rh_Surface : float
            The relative humidity of the surface of a solar module

        """
        rh_Surface = rh_ambient * \
            (relativeHumidity.Psat(ambient_temp) /
             relativeHumidity.Psat(surface_temp))

        return rh_Surface

        ###########
        # Front Encapsulant RH
        ###########

    def SDwNumerator(rh_ambient, ambient_temp, surface_temp, So=1.81390702, Eas=16.729, Ead=38.14):
        """
        Calculation is used in determining Relative Humidity of Frontside Solar
        module Encapsulant

        The function returns values needed for the numerator of the Diffusivity weighted water
        content equation. This function will return a pandas series prior to summation of the
        numerator

        Parameters
        ----------
        rh_ambient : pandas series (float)
            The ambient outdoor environmnet relative humidity in (%)
            EXAMPLE: "50 = 50% NOT .5 = 50%"
        ambient_temp : pandas series (float)
            The ambient outdoor environmnet temperature in Celsius
        surface_temp : pandas series (float)
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
        SDwNumerator_series : pandas series (float)
            Nnumerator of the Sdw equation prior to summation

        """

        # Get the relative humidity of the surface
        rh_surface = relativeHumidity.RHsurfaceOutside(
            rh_ambient, ambient_temp, surface_temp)

        # Generate a series of the numerator values "prior to summation"
        SDwNumerator_series = So * np.exp(- (Eas / (0.00831446261815324 * (surface_temp + 273.15))))\
                                * rh_surface * \
                                np.exp(- (Ead / (0.00831446261815324 * (surface_temp + 273.15))))

        return SDwNumerator_series

    def SDwDenominator(surface_temp, Ead=38.14):
        """
        Calculation is used in determining Relative Humidity of Frontside Solar
        module Encapsulant

        The function returns values needed for the denominator of the Diffusivity
        weighted water content equation(SDw). This function will return a pandas
        series prior to summation of the denominator

        Parameters
        ----------
        Ead : float
            Encapsulant diffusivity activation energy in [kJ/mol]
            38.14(kJ/mol) is the suggested value for EVA.
        surface_temp : pandas series (float)
            The surface temperature in Celsius of the solar panel module

        Returns
        -------
        SDwDenominator : pandas series (float)
            Denominator of the SDw equation prior to summation

        """

        SDwDenominator = np.exp(- (Ead /
                                   (0.00831446261815324 * (surface_temp + 273.15))))
        return SDwDenominator

    def SDw(rh_ambient, ambient_temp, surface_temp, So=1.81390702,  Eas=16.729, Ead=38.14):
        """
        Calculation is used in determining Relative Humidity of Frontside Solar
        module Encapsulant

        The function calculates the Diffusivity weighted water content equation. 

        Parameters
        ----------
        rh_ambient : pandas series (float)
            The ambient outdoor environmnet relative humidity in (%)
            EXAMPLE: "50 = 50% NOT .5 = 50%"
        ambient_temp : pandas series (float)
            The ambient outdoor environmnet temperature in Celsius
        surface_temp : pandas series (float)
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
        SDw : float
            Diffusivity weighted water content

        """

        numerator = relativeHumidity.SDwNumerator(
            rh_ambient, ambient_temp, surface_temp, So,  Eas, Ead)
        # get the summation of the numerator
        numerator = numerator.sum(axis=0, skipna=True)

        denominator = relativeHumidity.SDwDenominator(surface_temp, Ead)
        # get the summation of the denominator
        denominator = denominator.sum(axis=0, skipna=True)

        SDw = (numerator / denominator)/100

        return SDw

    def RHfrontEncap(surface_temp, SDw, So=1.81390702, Eas=16.729):
        """
        Function returns Relative Humidity of Frontside Solar Module Encapsulant

        Parameters
        ----------
        surface_temp : pandas series (float)
            The surface temperature in Celsius of the solar panel module
            "module temperature (C)"
        SDw : float
            Diffusivity weighted water content. *See EnergyCalcs.SDw() function
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
        RHfront_series = (SDw / (So * np.exp(- (Eas / (0.00831446261815324 *
                                                       (surface_temp + 273.15)))))) * 100

        return RHfront_series

        ###########
        # Back Encapsulant Relative Humidity
        ###########

    def _Csat(surface_temp, So=1.81390702, Eas=16.729):
        """
        Calculation is used in determining Relative Humidity of Backside Solar
        Module Encapsulant, and returns saturation of Water Concentration (g/cm³)

        Parameters
        -----------
        surface_temp : pandas series (float)
            The surface temperature in Celsius of the solar panel module
            "module temperature (C)"
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
            np.exp(- (Eas / (0.00831446261815324 * (273.15 + surface_temp))))

        return Csat

    def _Ceq(Csat, rh_SurfaceOutside):
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

    # Returns a numpy array

    @jit(nopython=True)
    def Ce_numba(start, surface_temp, rh_surface,
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
            the _Ceq(Saturation of Water Concentration) as a point
            of acceptable equilibrium
        surface_temp : pandas series (float)
            The surface temperature in Celsius of the solar panel module
            "module temperature (C)"
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

        dataPoints = len(surface_temp)
        Ce_list = np.zeros(dataPoints)

        for i in range(0, len(rh_surface)):

            if i == 0:
                # Ce = Initial start of concentration of water
                Ce = start
            else:
                Ce = Ce_list[i-1]

            Ce = Ce + ((WVTRo/100/100/24 * np.exp(-((EaWVTR) / (0.00831446261815324 * (surface_temp[i] + 273.15))))) /
                       (So * l/10 * np.exp(-((Eas) / (0.00831446261815324 * (surface_temp[i] + 273.15))))) *
                       (rh_surface[i]/100 * So * np.exp(-((Eas) / (0.00831446261815324 * (surface_temp[i] + 273.15)))) - Ce))

            Ce_list[i] = Ce

        return Ce_list

    def RHbackEncap(rh_ambient,ambient_temp,surface_temp,
                    WVTRo=7970633554,EaWVTR=55.0255,So=1.81390702,l=0.5,Eas=16.729):
        """
        RHbackEncap()

        Function to calculate the Relative Humidity of Backside Solar Module Encapsulant
        and return a pandas series for each time step        

        Parameters
        -----------
        rh_ambient : pandas series (float)
            The ambient outdoor environmnet relative humidity in (%)
            EXAMPLE: "50 = 50% NOT .5 = 50%"
        ambient_temp : pandas series (float)
            The ambient outdoor environmnet temperature in Celsius
        surface_temp : list (float)
            The surface temperature in Celsius of the solar panel module
            "module temperature (C)"
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

        rh_surface = relativeHumidity.RHsurfaceOutside(rh_ambient=rh_ambient,
                                                       ambient_temp=ambient_temp,
                                                       surface_temp=surface_temp)

        Csat = relativeHumidity._Csat(
            surface_temp=surface_temp, So=So, Eas=Eas)
        Ceq = relativeHumidity._Ceq(Csat=Csat, rh_SurfaceOutside=rh_surface)

        start = Ceq[0]

        # Need to convert these series to numpy arrays for numba function
        surface_temp_numba = surface_temp.to_numpy()
        rh_surface_numba = rh_surface.to_numpy()
        Ce_nparray = relativeHumidity.Ce_numba(start=start,
                                               surface_temp=surface_temp_numba,
                                               rh_surface=rh_surface_numba,
                                               WVTRo=WVTRo,
                                               EaWVTR=EaWVTR,
                                               So=So,
                                               l=l,
                                               Eas=Eas)

        #RHback_series = 100 * (Ce_nparray / (So * np.exp(-( (Eas) / 
        #                   (0.00831446261815324 * (surface_temp + 273.15))  )) ))
        RHback_series = 100 * (Ce_nparray / Csat)

        return RHback_series

        ###########
        # Back Sheet Relative Humidity
        ###########

    def RHbacksheet(RHbackEncap, RHsurfaceOutside):
        """
        Function to calculate the Relative Humidity of Backside BackSheet of a Solar Module
        and return a pandas series for each time step

        Parameters
        ----------
        RHbackEncap : pandas series (float)
            Relative Humidity of Frontside Solar module Encapsulant. *See RHbackEncap()
        RHsurfaceOutside : pandas series (float)
            The relative humidity of the surface of a solar module. *See RHsurfaceOutside()

        Returns
        --------
        RHbacksheet_series : pandas series (float)
            Relative Humidity of Backside Backsheet of a Solar Module

        @return rh_Surface     -
\                             
        """

        RHbacksheet_series = (RHbackEncap + RHsurfaceOutside)/2

        return RHbacksheet_series


class Degradation:

    def degradation(spectra, module_rh, module_temp, wavelengths,
                    Ea=40.0, n=1.0, x=0.64, C2=0.07, C=1.0):
        '''
        Compute degredation as double integral of Arrhenius (Activation
        Energy, RH, Temperature) and spectral (wavelength, irradiance)
        functions over wavelength and time.

        Parameters
        ----------
        spectra : pd.Series type=Float
            front or rear irradiance at each wavelength in "wavelengths"
        module_rh : pd.Series type=Float
            module RH, time indexed
        module_temp : pd.Series type=Float
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
        # module_temp = df['module_temp']
        # module_rh = df['module_rh']

        # Constants
        R = 0.0083145  # Gas Constant in [kJ/mol*K]

        wav_bin = list(np.diff(wavelengths))
        wav_bin.append(wav_bin[-1])  # Adding a bin for the last wavelength

        # Integral over Wavelength
        try:
            irr = pd.DataFrame(spectra.tolist(), index=spectra.index)
        except:
            # TODO: Fix this except it works on some cases, veto it by cases
            print("USING THE EXCEPT in PVDegradations Degradation function")
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
        C4 = np.exp(EApR/module_temp)

        RHn = module_rh**n
        data['Arr_integrand'] = C4*RHn

        data['dD'] = data['G_integral']*data['Arr_integrand']

        degradation = C*data['dD'].sum(axis=0)

        return degradation


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

    def ideal_installation_distance(df_tmy, metadata, T=70, x0 = 6, tilt=20, azimuth=180,
                                  skymodel='isotropic'):
        '''
        #MIKE : Explanation HERE
        
        #MIKE REFERENCE for the paper... 
        
        1. Takes a weather file
        2. Calculates the module temperature with PVLIB, at infinity gap
        3. Calculates the module temperature with PVLIB, at 0
        4. Calculates the 98 percentile for 2 
        5. Calculates the 98 percentile for 3
        5. Use both percentiles to calculate x
        
        Parameters
        ----------
        df_tmy : DataFrame
            Dataframe with the weather data. Must have
            'air_temperature' and 'wind_speed'
        metadata : Dictionary
            must have 'latitude' and 'longitude'
        T : float, optional
            Temperature goal for the IEC 63126. The default is 70; 80 is also 
            another common value
        x0 : float, optional
            Decay constant, based on the characteristic distance. Comes from 
            paper REF #MIKE
        tilt : float, optional
            Tilt of the PV array. The default is 20.
        azimuth : float, optional
            Azimuth of the PV array. The default is 180, facing south
        skymodel : str
            Tells PVlib which sky model to use. Default 'isotropic'.
            
        Returns
        -------
        x : float
            Recommended installation distance per IEC 63126.
    
        '''    
        
        # 1. Calculate Sun Position & POA Irradiance
        
        # Make Location
        # make a Location object corresponding to this TMY
        location = pvlib.location.Location(latitude=metadata['latitude'],
                                           longitude=metadata['longitude'])
      
        
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
    
        x=-x0 * np.log(1-(T0p-T)/(T0p-T1p))

        return x