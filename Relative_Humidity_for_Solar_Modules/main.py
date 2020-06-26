"""
Contains energy algorithms for processing.

@author: Derek Holsapple
"""

import numpy as np
from numba import jit 
import pandas as pd   
from scipy.constants import convert_temperature    
   

class energyCalcs:
    """
    energyCalcs class contains the Vant Hoff acceleration factor and Arrhenius 
    Equations Acceleration Factor

    """ 
            
    def k( avgWVP ):
        """
        Determine the rate of water ingress of water through edge seal material
    
        Parameters
        -----------
        avgWVP : float 
            Average of the Yearly water vapor 
            pressure for 1 year

        Returns
        -------
        k : float
            Ingress rate of water through edge seal
        
        """  
        
        k = .0013 * (avgWVP)**.4933
        
        return k

    def edgeSealWidth( k ):
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
    @jit(nopython=True , error_model = 'python')  
    def dewYield( h , tD , tA , windSpeed , n ):
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
        windSpeed : float
            Air or windspeed measure in m*s^-1  or m/s
        n : float 
            Total sky cover(okta)
        
        Returns
        -------
        dewYield : float
            Amount of dew yield in (mm·d−1)  
            
        """
        windSpeedCutOff = 4.4 
        dewYield = ( 1/12 ) * (.37 * ( 1 + ( 0.204323 * h ) - (0.0238893 * \
                    h**2 ) - ( 18.0132 - ( 1.04963 * h**2 ) + ( 0.21891 * \
                    h**2 ) ) * (10**( -3 ) * tD ) ) * ( ( ( ( tD + 273.15)/ \
                    285)**4)*(1 - (n/8))) + (0.06 * (tD - tA ) ) * ( 1 + 100 * \
                    ( 1 - np.exp( - ( windSpeed / windSpeedCutOff)**20 ) ) ) ) 

        return dewYield
    
############
# Water Vapor Pressure
############        

    def waterVaporPressure( dewPtTemp ):
        """       
        Find the average water vapor pressure (kPa) based on the Dew Point 
        Temperature model created from Mike Kempe on 10/07/19 from Miami,FL excel sheet.  

        Parameters
        ----------        
        dewPtTemp : float
            Dew Point Temperature
        
        Returns
        --------
        watervaporpressure : float
            Water vapor pressure in kPa
            
        """    
        watervaporpressure = (np.exp(( 3.257532E-13 * dewPtTemp**6 ) - 
                       ( 1.568073E-10 * dewPtTemp**6 ) + 
                       ( 2.221304E-08 * dewPtTemp**4 ) + 
                       ( 2.372077E-7 * dewPtTemp**3) - 
                       ( 4.031696E-04 * dewPtTemp**2) + 
                       ( 7.983632E-02 * dewPtTemp ) - 
                       ( 5.698355E-1)))
    
        return watervaporpressure
    
############
# Solder Fatigue
############  
        
    def _avgDailyTempChange( localTime , cell_Temp ):
        """
        Helper function. Get the average of a year for the daily maximum temperature change.
        
        For every 24hrs this function will find the delta between the maximum
        temperature and minimun temperature.  It will then take the deltas for 
        every day of the year and return the average delta. 
    
        Parameters
        ------------
        localTime : timestamp series
            Local time of specific site by the hour
            year-month-day hr:min:sec . (Example) 2002-01-01 01:00:00
        cell_Temp : float series
            Photovoltaic module cell temperature(Celsius) for every hour of a year
        
        Returns
        -------
        avgDailyTempChange : float
            Average Daily Temerature Change for 1-year (Celsius)
        avgMaxCellTemp : float
            Average of Daily Maximum Temperature for 1-year (Celsius)
            
        """    
        #Setup frame for vector processing
        timeAndTemp_df = pd.DataFrame( columns = ['Cell Temperature'])
        timeAndTemp_df['Cell Temperature'] = cell_Temp
        timeAndTemp_df.index = localTime
        timeAndTemp_df['month'] = timeAndTemp_df.index.month
        timeAndTemp_df['day'] = timeAndTemp_df.index.day
        
        #Group by month and day to determine the max and min cell Temperature (C) for each day
        dailyMaxCellTemp_series = timeAndTemp_df.groupby(['month','day'])['Cell Temperature'].max()
        dailyMinCellTemp_series = timeAndTemp_df.groupby(['month','day'])['Cell Temperature'].min()
        cellTempChange = pd.DataFrame({ 'Max': dailyMaxCellTemp_series, 'Min': dailyMinCellTemp_series})
        cellTempChange['TempChange'] = cellTempChange['Max'] - cellTempChange['Min']
        
        #Find the average temperature change for every day of one year (C) 
        avgDailyTempChange = cellTempChange['TempChange'].mean()
        #Find daily maximum cell temperature average
        avgMaxCellTemp  = dailyMaxCellTemp_series.mean()
            
        return avgDailyTempChange , avgMaxCellTemp 



    def _timesOverReversalNumber( cell_Temp , reversalTemp):
        """
        Helper function. Get the number of times a temperature increases or decreases over a 
        specific temperature gradient.

        Parameters
        ------------
        cell_Temp : float series
            Photovoltaic module cell temperature(Celsius) 
        reversalTemp : float
            Temperature threshold to cross above and below

        Returns
        --------
        numChangesTempHist : int 
            Number of times the temperature threshold is crossed   
        
        """
        #Find the number of times the temperature crosses over 54.8(C)
        
        
        temp_df = pd.DataFrame()
        temp_df['CellTemp'] = cell_Temp
        temp_df['COMPARE'] = cell_Temp
        temp_df['COMPARE'] = temp_df.COMPARE.shift(-1)
        
        #reversalTemp = 54.8
        
        temp_df['cross'] = (
            ((temp_df.CellTemp >= reversalTemp) & (temp_df.COMPARE < reversalTemp)) |
            ((temp_df.COMPARE > reversalTemp) & (temp_df.CellTemp <= reversalTemp)) |
            (temp_df.CellTemp == reversalTemp))
        
        numChangesTempHist = temp_df.cross.sum()
        
        return numChangesTempHist
        
        
    def solderFatigue( localTime , cell_Temp , reversalTemp):
        """        
        Get the Thermomechanical Fatigue of flat plate photovoltaic module solder joints.
        Damage will be returned as the rate of solder fatigue for one year. Based on:
    
            Bosco, N., Silverman, T. and Kurtz, S. (2020). Climate specific thermomechanical 
            fatigue of flat plate photovoltaic module solder joints. [online] Available 
            at: https://www.sciencedirect.com/science/article/pii/S0026271416300609 
            [Accessed 12 Feb. 2020].
        
        Parameters
        ------------
        localTime : timestamp series
            Local time of specific site by the hour year-month-day hr:min:sec
            (Example) 2002-01-01 01:00:00
        cell_Temp : float series           
            Photovoltaic module cell temperature(Celsius) for every hour of a year
        reversalTemp : float
            Temperature threshold to cross above and below
        
        Returns
        --------
        damage : float series
            Solder fatigue damage for a time interval depending on localTime (kPa)      
        
        """ 
        
        #TODO Make this function have more utility.  People want to run all the scenarios from the bosco paper.  
        # Currently have everything hard coded for hourly calculation
        # i.e. 405.6, 1.9, .33, .12
        
        # Get the 1) Average of the Daily Maximum Cell Temperature (C)
        #         2) Average of the Daily Maximum Temperature change avg(daily max - daily min)
        #         3) Number of times the temperaqture crosses above or below the reversal Temperature
        MeanDailyMaxCellTempChange , dailyMaxCellTemp_Average = energyCalcs._avgDailyTempChange( localTime , cell_Temp )
        #Needs to be in Kelvin for equation specs
        dailyMaxCellTemp_Average = convert_temperature( dailyMaxCellTemp_Average , 'Celsius', 'Kelvin')
        numChangesTempHist = energyCalcs._timesOverReversalNumber( cell_Temp, reversalTemp )
              
        #k = Boltzmann's Constant
        damage = 405.6 * (MeanDailyMaxCellTempChange **1.9) * \
                         (numChangesTempHist**.33) * \
                         np.exp(-(.12/(.00008617333262145*dailyMaxCellTemp_Average)))
        #Convert pascals to kilopascals
        damage = damage/1000
        return damage
    
    def _power( cellTemp , globalPOA ):
        """
        Helper function. Find the relative power produced from a solar module.
    
        Model derived from Mike Kempe Calculation on paper
        (ADD IEEE reference)
        
        Parameters
        ------------
        cellTemp : float
            Cell Temperature of a solar module (C)

        Returns
        --------
        power : float
            Power produced from a module in KW/hours  
        """           
        #KW/hr
        
        # Why is there two definitions?
        power = 0.0002 * globalPOA * ( 1 + ( 25 - cellTemp ) * .004 )  
        power = globalPOA * ( 1 + ( 25 - cellTemp ) * .004 )
          
        return power
        
        
################################################################################

    ############################################
    #Vant Hoff Degradation Function
    ############################################       
    def _rateOfDegEnv( poa, x, cellTemp, refTemp, Tf):
        """
        Helper function. Find the rate of degradation kenetics using the Fischer model.  
        Degradation kentics model interpolated 50 coatings with respect to 
        color shift, cracking, gloss loss, fluorescense loss, 
        retroreflectance loss, adhesive transfer, and shrinkage.
        
        (ADD IEEE reference)
        
        Parameters
        ------------
        poa : float
            (Global) Plan of Array irradiance (W/m^2)
        x : float
            Fit parameter
        cellTemp : float
            Solar module cell temperature (C)
        refTemp : float
            Reference temperature (C) "Chamber Temperature"
        Tf : float
            Multiplier for the increase in degradation
                                          for every 10(C) temperature increase

        Returns
        --------
        degradationrate : float
            rate of Degradation (NEED TO ADD METRIC)

        """        
        return poa**(x) * Tf ** ( (cellTemp - refTemp)/10 )



    def _rateOfDegChamber( Ichamber , x ):
        """
        Helper function. Find the rate of degradation kenetics of a simulated chamber. Mike Kempe's 
        calculation of the rate of degradation inside a accelerated degradation chamber. 
        
        (ADD IEEE reference)

        Parameters
        ----------
        Ichamber : float
            Irradiance of Controlled Condition W/m^2
        x : float
            Fit parameter

        Returns
        --------
        chamberdegradationrate : float
            Degradation rate of chamber 
        """        
        chamberdegradationrate = Ichamber ** ( x )
        
        return chamberdegradationrate



    def _accelerationFactor( numerator , denominator ):
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
        
        chamberAccelerationFactor = ( numerator / denominator )
        
        return chamberAccelerationFactor
    
    
    
    def vantHoffDeg( x , Ichamber , poa , Toutdoor , Tf , refTemp):    
        """
        Vant Hoff Irradiance Degradation 
        
        Parameters
        -----------
        x : float
            fit parameter
        Ichamber : float
            Irradiance of Controlled Condition W/m^2
        globalPOA : float or series
            Global Plane of Array Irradiance W/m^2
        Toutdoor : pandas series
            Solar module temperature or Cell temperature (C)
        Tf : float
            Multiplier for the increase in degradation for every 10(C) temperature increase
        refTemp : float
            Reference temperature (C) "Chamber Temperature"                                          
                                          
        Returns
        -------
        accelerationFactor : float or series
            Degradation acceleration factor
            
        """  
        rateOfDegEnv = energyCalcs._rateOfDegEnv(poa,
                                                x , 
                                                Toutdoor ,
                                                refTemp ,
                                                Tf )        
        #sumOfDegEnv = rateOfDegEnv.sum(axis = 0, skipna = True)
        avgOfDegEnv = rateOfDegEnv.mean()
            
        rateOfDegChamber = energyCalcs._rateOfDegChamber( Ichamber , x )
        
        accelerationFactor = energyCalcs.accelerationFactor( rateOfDegChamber , avgOfDegEnv)
        
        return  accelerationFactor
    
    
    
##############################################################################################
    ############################################
    #Vant Hoff Environmental Characterization
    ############################################

    def ToeqVantHoff( Tf, Toutdoor ):
        """
        Function to obtain the Vant Hoff temperature equivalent (C)
        
        Parameters
        ----------
        Tf : float
            Multiplier for the increase in degradation for every 10(C) temperature increase  
        Toutdoor : pandas series
            Solar module temperature or Cell temperature (C)    
    
        Returns
        -------
        Toeq : float 
            Vant Hoff temperature equivalent (C)
                     
        """
        toSum = Tf ** ( Toutdoor / 10 )
        summation = toSum.sum(axis = 0, skipna = True)
    
        Toeq = (10 / np.log ( Tf ) ) * np.log ( summation / len(Toutdoor) )
        
        return Toeq
    
    
    
    def IwaVantHoff( globalPOA , x , Tf , Toutdoor , ToeqVantHoff):
        """       
        IWa : Environment Characterization (W/m^2)
        *for one year of degredation the controlled environmnet lamp settings will 
            need to be set to IWa
        
        Parameters
        -----------
        globalPOA : float or series
            Global Plane of Array Irradiance W/m^2    
        x : float
            Fit parameter  
        Tf : float
            Multiplier for the increase in degradation for every 10(C) temperature increase  
        Toutdoor : pandas series
            Solar module temperature or Cell temperature (C)      
        ToeqVantHoff : float 
            Vant Hoff temperature equivalent (C)
        
        Returns
        --------
        Iwa : float
            Environment Characterization (W/m^2)
            
        """
        toSum = (globalPOA ** x) * (Tf ** ( (Toutdoor - ToeqVantHoff)/10 ))
        summation = toSum.sum(axis = 0, skipna = True)
        
        Iwa = ( summation / len(globalPOA) ) ** ( 1 / x )
        
        return Iwa    


##############################################################################################
    ############################################
    #Arrhenius Degradation Function
    ############################################
        
        
    def _arrheniusDenominator( poa , x, rh_outdoor , n , Toutdoor , Ea):
        """
        Helper function. Calculates the rate of degredation of the Environmnet 

        Parameters
        ----------
        poa : float
            (Global) Plan of Array irradiance (W/m^2)
        x : float
            Fit parameter         
        rh_outdoor : pandas series
            Relative Humidity of material of interest. Acceptable relative 
            humiditys can be calculated from these functions: RHbacksheet(),
            RHbackEncap(); RHfront();  RHsurfaceOutside()
        n : float
            Fit parameter for relative humidity 
        Toutdoor : pandas series
            Solar module temperature or Cell temperature (C)   
        Ea : float
            Degredation Activation Energy (kJ/mol)        
        
        Returns
        -------
        environmentDegradationRate : pandas series
            Degradation rate of environment 
        """        
        
        environmentDegradationRate = poa**(x) * rh_outdoor**(n) * np.exp( - ( Ea/ ( 0.00831446261815324 * (Toutdoor + 273.15)  )))
        
        return environmentDegradationRate
    
    
    
    def _arrheniusNumerator( Ichamber , x , rhChamber, n ,  Ea , Tchamber ):
        """
        Helper function. Find the rate of degradation of a simulated chamber.  
        
        Parameters
        ----------
        Ichamber : float
            Irradiance of Controlled Condition W/m^2
        x : float 
            Fit parameter
        rhChamber : float 
            Relative Humidity of Controlled Condition (%)
            EXAMPLE: "50 = 50% NOT .5 = 50%" 
        n : float 
            Fit parameter for relative humidity                                      
        Ea : float
            Degredation Activation Energy (kJ/mol)
        Tchamber : float
            Reference temperature (C) "Chamber Temperature"
    
        Returns
        --------
        arrheniusNumerator : float
            Degradation rate of the chamber
        """        
        
        arrheniusNumerator = ( Ichamber ** ( x ) * rhChamber ** (n) * 
                              np.exp( - ( Ea/ ( 0.00831446261815324 * 
                                               (Tchamber+273.15)  ))))
        return arrheniusNumerator
    
    
    
    def arrheniusCalc( x , Ichamber , rhChamber , n , rh_outdoor , globalPOA , Tchamber , Toutdoor,  Ea):    
        """
        Calculate the Acceleration Factor between the rate of degredation of a 
        modeled environmnet versus a modeled controlled environmnet
        
        Example: "If the AF=25 then 1 year of Controlled Environment exposure 
                    is equal to 25 years in the field"

        Parameters
        ----------
        x : float 
            Fit parameter
        Ichamber : float 
            Irradiance of Controlled Condition W/m^2
        rhChamber : float 
            Relative Humidity of Controlled Condition (%). 
            EXAMPLE: "50 = 50% NOT .5 = 50%"
        n : float
            Fit parameter for relative humidity  
        rh_outdoor : pandas series 
            Relative Humidity of material of interest 
            Acceptable relative humiditys can be calculated 
            from these functions: RHbacksheet(), RHbackEncap(), RHfront(),
            RHsurfaceOutside()
        globalPOA : pandas series
            Global Plane of Array Irradiance W/m^2
        Tchamber : float
            Reference temperature (C) "Chamber Temperature"  
        Toutdoor : pandas series
            Solar module temperature or Cell temperature (C)
        Ea : float
            Degredation Activation Energy (kJ/mol) 
                                        
        Returns
        --------
        accelerationFactor : pandas series
            Degradation acceleration factor

        """  
        arrheniusDenominator = energyCalcs._arrheniusDenominator(globalPOA,
                                                        x , 
                                                        rh_outdoor,
                                                        n,
                                                        Toutdoor ,
                                                        Ea )        
        

        AvgOfDenominator = arrheniusDenominator.mean()
            
        arrheniusNumerator = energyCalcs._arrheniusNumerator( Ichamber , x , rhChamber, n ,  Ea , Tchamber )
        
        
        accelerationFactor = energyCalcs.accelerationFactor( arrheniusNumerator , AvgOfDenominator)
        
        return accelerationFactor
        


###############################################################################
    ############################################
    #Arrhenius Environmental Characterization
    ############################################

    def TeqArrhenius( Toutdoor , Ea ):
        """ 
        Get the Temperature equivalent required for the settings of the controlled environment
        Calculation is used in determining Arrhenius Environmental Characterization
    
        Parameters
        -----------
        Toutdoor : pandas series
            Solar module temperature or Cell temperature (C)
        Ea : float 
            Degredation Activation Energy (kJ/mol)    
        
        Returns
        -------
        Teq : float
            Temperature equivalent (Celsius) required 
            for the settings of the controlled environment
            
        """
        
        summationFrame = np.exp( - ( Ea/ ( 0.00831446261815324 * (Toutdoor + 273.15)  )))
        sumForTeq = summationFrame.sum(axis = 0, skipna = True)
        Teq = -( (Ea) /  ( 0.00831446261815324 * np.log ( sumForTeq / len(Toutdoor) ) ) )
        # Convert to celsius
        Teq = Teq - 273.15
        
        return Teq
    
    
    
    def RHwaArrhenius( rh_outdoor , n , Ea , Toutdoor, Teq ):
        """        
        Get the Relative Humidity Weighted Average.    
        Calculation is used in determining Arrhenius Environmental Characterization

        Parameters
        -----------
        rh_outdoor : pandas series
            Relative Humidity of material of interest. Acceptable relative 
            humiditys can be calculated from the below functions:
            RHbacksheet(), RHbackEncap(), RHfront(), RHsurfaceOutside()
        n : float
            Fit parameter for relative humidity       
        Ea : float 
            Degredation Activation Energy (kJ/mol)                   
        Toutdoor : pandas series
            solar module temperature or Cell temperature (C)
        Teq : float
            Temperature equivalent (Celsius) required 
            for the settings of the controlled environment
        
        Returns
        --------
        RHwa : float
            Relative Humidity Weighted Average (%)
            
        """
    
        summationFrame = (rh_outdoor ** n ) * np.exp( - ( Ea/ 
                         ( 0.00831446261815324 * (Toutdoor + 273.15)  )))
        sumForRHwa = summationFrame.sum(axis = 0, skipna = True)
        RHwa =  (sumForRHwa / ( len(summationFrame) * np.exp( - ( Ea/ 
                               ( 0.00831446261815324 * (Teq + 273.15)  ))))) ** (1/n)
    
        return RHwa
    
      
        
    def IwaArrhenius( poa , x ,  rh_outdoor , n , Toutdoor , Ea , RHwa, Teq):
        """       
        Function to calculate IWa, the Environment Characterization (W/m^2)
        *for one year of degredation the controlled environmnet lamp settings will 
            need to be set at IWa
        
        Parameters
        ----------
        poa : float
            (Global) Plan of Array irradiance (W/m^2)
        x : float
            Fit parameter         
        rh_outdoor : pandas series
            Relative Humidity of material of interest 
            Acceptable relative humiditys can be calculated 
            from these functions: RHbacksheet(), RHbackEncap(), RHfront()
                                  RHsurfaceOutside()
        n : float
            Fit parameter for relative humidity 
        Toutdoor : pandas series
            Solar module temperature or Cell temperature (C)   
        Ea : float 
            Degradation Activation Energy (kJ/mol)    
        RHwa : float
            Relative Humidity Weighted Average (%) 
        Teq : float
            Temperature equivalent (Celsius) required 
            for the settings of the controlled environment
        
        Returns
        --------
        Iwa : float
            Environment Characterization (W/m^2)
            
        """
        numerator = poa**(x) * rh_outdoor**(n) * np.exp( - ( Ea/ ( 0.00831446261815324 * (Toutdoor + 273.15) )))
        sumOfNumerator = numerator.sum(axis = 0, skipna = True)
    
        denominator = (len(numerator)) * ((RHwa)**n )  * (np.exp( - ( Ea/ ( 0.00831446261815324 * (Teq + 273.15)  ))))  
    
        IWa = ( sumOfNumerator / denominator )**(1/x)
    
        return IWa




############
# Misc. Functions for Energy Calcs
############ 

    
    def _rH_Above85( rH ):    
        """
        Helper function. Determines if the relative humidity is above 85%.  
        
        Parameters
        ----------
        rH : float
            Relative Humidity %
        
        Returns
        --------
        rHabove85 : boolean
            True if the relative humidity is above 85% or False if the relative 
            humidity is below 85%
            
        """         

        if rH > 85:
            rHabove85 = True
            
        else:
            rHabove85 = False
        
        return rHabove85
     
        
   
    def _hoursRH_Above85( df ):      
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
        booleanDf = df.apply(lambda x: energyCalcs._rH_Above85( x ) )
        numhoursabove85 = booleanDf.sum()
        
        return numhoursabove85
        
  

    def _whToGJ( wh ):
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
    
    

    def _gJtoMJ( gJ ):
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
        
    def Psat( temp ):
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
        
        Psat = np.exp( -0.000000002448137*temp**4  \
                       +0.000001419572*temp**3  \
                       -0.0003779559*temp**2  \
                       +0.07796986*temp  \
                       -0.5796729   )
        
        return Psat
        
        
    
    def RHsurfaceOutside(rH_ambient, ambient_temp, surface_temp ):
        """
        Function calculates the Relative Humidity of a Solar Panel Surface
    
        Parameters
        ----------
        rH_ambient : float 
            The ambient outdoor environmnet relative humidity 
        ambient_temp : float 
            The ambient outdoor environmnet temperature in Celsius
        surface_temp : float 
            The surface temperature in Celsius of the solar panel module 
        
        Returns
        --------
        rH_Surface : float
            The relative humidity of the surface of a solar module    
                
        """
        rH_Surface = rH_ambient*( relativeHumidity.Psat( ambient_temp ) / relativeHumidity.Psat( surface_temp )  )

        return rH_Surface
    
    
        ###########
        # Front Encapsulant RH       
        ###########
        
        
    def SDwNumerator( rH_ambient, ambient_temp, surface_temp, So=1.81390702, Eas=16.729, Ead=38.14):
        """
        Calculation is used in determining Relative Humidity of Frontside Solar
        module Encapsulant

        The function returns values needed for the numerator of the Diffusivity weighted water content equation. 
        This function will return a pandas series prior to summation of the numerator 

        Parameters
        ----------
        rH_ambient : pandas series (float)
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
        
        #Get the relative humidity of the surface
        RH_surface = relativeHumidity.RHsurfaceOutside(rH_ambient, ambient_temp, surface_temp )
        
        #Generate a series of the numerator values "prior to summation"
        SDwNumerator_series = So * np.exp( - ( Eas / (0.00831446261815324 * (surface_temp + 273.15) ))) * \
                        RH_surface * np.exp( - ( Ead / (0.00831446261815324 * (surface_temp + 273.15) )))
        
        return SDwNumerator_series


    def SDwDenominator( surface_temp, Ead=38.14):
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
        
        SDwDenominator =  np.exp( - ( Ead / (0.00831446261815324 * (surface_temp + 273.15) )))
        return SDwDenominator



    def SDw( rH_ambient, ambient_temp, surface_temp, So=1.81390702,  Eas=16.729 , Ead=38.14):
        """
        Calculation is used in determining Relative Humidity of Frontside Solar
        module Encapsulant

        The function calculates the Diffusivity weighted water content equation. 

        Parameters
        ----------
        rH_ambient : pandas series (float)
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
        
        numerator = relativeHumidity._SDwNumerator( rH_ambient, ambient_temp, surface_temp, So ,  Eas , Ead)
        #get the summation of the numerator
        numerator = numerator.sum(axis = 0, skipna = True)

        denominator = relativeHumidity.SDwDenominator( surface_temp, Ead)
        #get the summation of the denominator
        denominator = denominator.sum(axis = 0, skipna = True)

        SDw = (numerator / denominator)/100
        
        return SDw
    
    
    
    def RHfront(surface_temp, SDw , So = 1.81390702, Eas = 16.729):
        """
        Function returns Relative Humidity of Frontside Solar Module Encapsulant
        
        Parameters
        ----------
        surface_temp : pandas series (float)
            The surface temperature in Celsius of the solar panel module
            "module temperature (C)"
        SDw : float
            Diffusivity weighted water content. *See energyCalcs.SDw() function        
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
        RHfront_series = (SDw / ( So * np.exp( - ( Eas / (0.00831446261815324 * \
                         (surface_temp + 273.15) ))))) * 100
        
        return RHfront_series


        ###########
        # Back Encapsulant Relative Humidity       
        ###########

    def Csat(surface_temp, So = 1.81390702, Eas = 16.729):
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
        
        #Saturation of water concentration
        Csat = So * np.exp( - ( Eas / 0.00831446261815324 / (273.15 + surface_temp ) ) )
        
        return Csat


    def Ceq( Csat , rH_SurfaceOutside ):
        """       
        Calculation is used in determining Relative Humidity of Backside Solar
        Module Encapsulant, and returns Equilibration water concentration (g/cm³)
 
        Parameters
        ------------
        Csat : pandas series (float)
            Saturation of Water Concentration (g/cm³)  
        rH_SurfaceOutside : pandas series (float) 
            The relative humidity of the surface of a solar module (%)
        
        Returns
        --------
        Ceq : pandas series (float)
            Equilibration water concentration (g/cm³) 
            
        """
        
        Ceq = Csat * (rH_SurfaceOutside/100)
        
        return Ceq



    #Returns a numpy array
    @jit(nopython=True)
    def Ce_numba( start , surface_temp, RH_surface, WVTRo = 7970633554, EaWVTR=55.0255, So = 1.81390702, l=0.5, Eas = 16.729):
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
            the Ceq(Saturation of Water Concentration) as a point
            of acceptable equilibrium
        surface_temp : pandas series (float)
            The surface temperature in Celsius of the solar panel module
            "module temperature (C)"
        rH_Surface : list (float) 
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
        
        for i in range(0 , len(RH_surface)):
            
            if i == 0:
                #Ce = Initial start of concentration of water
                Ce = start
            else:
                Ce = Ce_list[i-1]
            
            Ce = Ce + ((WVTRo/100/100/24 * np.exp(-( (EaWVTR) / (0.00831446261815324 * (surface_temp[i] + 273.15))))) / \
                       ( So * l/10 * np.exp(-( (Eas) / (0.00831446261815324 * (surface_temp[i] + 273.15))  ))   ) * \
                       (RH_surface[i]/100 * So * np.exp(-( (Eas) / (0.00831446261815324 * (surface_temp[i] + 273.15))  )) - Ce ))
        
            Ce_list[i] = Ce
            
        return Ce_list
    
    
    
    def RHbackEncap(rH_ambient, ambient_temp, surface_temp, WVTRo = 7970633554, EaWVTR=55.0255, So = 1.81390702, l=0.5, Eas = 16.729):
        """
        RHbackEncap()
        
        Function to calculate the Relative Humidity of Backside Solar Module Encapsulant
        and return a pandas series for each time step        
        
        Parameters
        -----------
        rH_ambient : pandas series (float)
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
        
        RH_surface = relativeHumidity.RHsurfaceOutside(rH_ambient, ambient_temp, surface_temp )
        
        
        Csat = relativeHumidity.Csat( So , Eas, surface_temp )
        Ceq = relativeHumidity.Ceq( Csat , RH_surface )

        start = Ceq[0]
        
        #Need to convert these series to numpy arrays for numba function
        surface_temp_numba = surface_temp.to_numpy()
        RH_surface_numba = RH_surface.to_numpy()
        
        Ce_nparray = relativeHumidity.Ce_numba(start, surface_temp_numba, RH_surface_numba, WVTRo , EaWVTR, So, l, Eas)
        
        RHback_series = 100 * (Ce_nparray / (So * np.exp(-( (Eas) / (0.00831446261815324 * (surface_temp + 273.15))  )) ))
        
        return RHback_series


        ###########
        # Back Sheet Relative Humidity       
        ###########
        
    def RHbacksheet( RHbackEncap , RHsurfaceOutside ):
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
            
        @return rH_Surface     -
\                             
        """
        
        RHbacksheet_series = (RHbackEncap + RHsurfaceOutside)/2

        return RHbacksheet_series

    