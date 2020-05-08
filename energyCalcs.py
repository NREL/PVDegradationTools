"""
Contains energy algorithms for processing.

@author: Derek Holsapple
"""

import numpy as np
from numba import jit 
import pandas as pd   
from scipy.constants import convert_temperature    
   

class energyCalcs:

    def k( avgWVP ):
        '''
        
        Determine the rate of water ingress of water through edge seal material
    
        
        @param avgWVP              -float, Average of the Yearly water vapor 
                                                pressure for 1 year

        @return k                  -float , Ingress rate of water through edge seal
        '''  
        return .0013 * (avgWVP)**.4933

    def edgeSealWidth( k ):
        '''
        
        Determine the width of edge seal required for a 25 year water ingress
    
        @param k              -float, rate of water ingress

        @return width                  -float , width of edge seal required for
                                                    a 25 year water ingress (mm)
        '''  
        return k * (25 * 365.25 * 24)**.5
    
    
############
# Dew Yield
############
    # Numba Machine Language Level
    @jit(nopython=True , error_model = 'python')  
    def dewYield( h , tD , tA , windSpeed , n ):
        '''
        dewYield()
        
        Find the dew yield in (mm·d−1).  Calculation taken from journal
        "Estimating dew yield worldwide from a few meteo data"
            -D. Beysens

        (ADD IEEE reference)
        
        @param h          -int, site elevation in kilometers
        @param tD         -float, Dewpoint temperature in Celsius
        @param tA         -float, air temperature "dry bulb temperature"
        @param windSpeed  -float, air or windspeed measure in m*s^-1  or m/s
        @param n          -float, Total sky cover(okta)
        @return  dewYield -float, amount of dew yield in (mm·d−1)  
        '''
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
        '''
        waterVaporPressure()
        
        Find the average water vapor pressure (kPa) based on the Dew Point 
        Temperature model created from Mike Kempe on 10/07/19 from Miami,FL excel sheet.  
        
        @param dewPtTemp          -float, Dew Point Temperature
        @return                   -float, return water vapor pressur in kPa
        '''    
        return( np.exp(( 3.257532E-13 * dewPtTemp**6 ) - 
                       ( 1.568073E-10 * dewPtTemp**6 ) + 
                       ( 2.221304E-08 * dewPtTemp**4 ) + 
                       ( 2.372077E-7 * dewPtTemp**3) - 
                       ( 4.031696E-04 * dewPtTemp**2) + 
                       ( 7.983632E-02 * dewPtTemp ) - 
                       ( 5.698355E-1)))
    
   
############
# Solder Fatigue
############  
        
    def avgDailyTempChange( localTime , cell_Temp ):
        '''
        HELPER FUNCTION
        
        Get the average of a year for the daily maximum temperature change.
        
        For every 24hrs this function will find the delta between the maximum
        temperature and minimun temperature.  It will then take the deltas for 
        every day of the year and return the average delta. 
    
        
        @param localTime           -timestamp series, Local time of specific site by the hour
                                                year-month-day hr:min:sec
                                                (Example) 2002-01-01 01:00:00
        @param cell_Temp           -float series, Photovoltaic module cell 
                                               temperature(Celsius) for every hour of a year
                                               
        @return avgDailyTempChange -float , Average Daily Temerature Change for 1-year (Celsius)
        @return avgMaxCellTemp     -float , Average of Daily Maximum Temperature for 1-year (Celsius)
        '''    
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



    def timesOverReversalNumber( cell_Temp , reversalTemp):
        '''
        HELPER FUNCTION
        
        Get the number of times a temperature increases or decreases over a 
        specific temperature gradient.


        @param cell_Temp           -float series, Photovoltaic module cell 
                                               temperature(Celsius) 
        @param reversalTemp        -float, temperature threshold to cross above and below

        @param numChangesTempHist  -int , Number of times the temperature threshold is crossed                          
        '''
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
        '''
        HELPER FUNCTION
        
        Get the Thermomechanical Fatigue of flat plate photovoltaic module solder joints.
        Damage will be returned as the rate of solder fatigue for one year
    
        Bosco, N., Silverman, T. and Kurtz, S. (2020). Climate specific thermomechanical 
        fatigue of flat plate photovoltaic module solder joints. [online] Available 
        at: https://www.sciencedirect.com/science/article/pii/S0026271416300609 
        [Accessed 12 Feb. 2020].
        
        @param localTime           -timestamp series, Local time of specific site by the hour
                                                year-month-day hr:min:sec
                                                (Example) 2002-01-01 01:00:00
        @param cell_Temp           -float series, Photovoltaic module cell 
                                               temperature(Celsius) for every hour of a year
        @param reversalTemp        -float, temperature threshold to cross above and below
        
        @return damage           - float series, Solder fatigue damage for a 
                                                    time interval depending on localTime (kPa)          
        ''' 
#TODO Make this function have more utility.  People want to run all the scenarios from the bosco paper.  
        # Currently have everything hard coded for hourly calculation
        # i.e. 405.6, 1.9, .33, .12
        
        
        # Get the 1) Average of the Daily Maximum Cell Temperature (C)
        #         2) Average of the Daily Maximum Temperature change avg(daily max - daily min)
        #         3) Number of times the temperaqture crosses above or below the reversal Temperature
        MeanDailyMaxCellTempChange , dailyMaxCellTemp_Average = energyCalcs.avgDailyTempChange( localTime , cell_Temp )
        #Needs to be in Kelvin for equation specs
        dailyMaxCellTemp_Average = convert_temperature( dailyMaxCellTemp_Average , 'Celsius', 'Kelvin')
        numChangesTempHist = energyCalcs.timesOverReversalNumber( cell_Temp, reversalTemp )
              
        #k = Boltzmann's Constant
        damage = 405.6 * (MeanDailyMaxCellTempChange **1.9) * \
                         (numChangesTempHist**.33) * \
                         np.exp(-(.12/(.00008617333262145*dailyMaxCellTemp_Average)))
        #Convert pascals to kilopascals
        damage = damage/1000
        return damage
    
    def power( cellTemp , globalPOA ):
        '''
        HELPER FUNCTION
        
        Find the relative power produced from a solar module.
    
        Model derived from Mike Kempe Calculation on paper
        (ADD IEEE reference)
        
        @param cellTemp           -float, Cell Temperature of a solar module (C)

        @return power produced from a module in KW/hours  
        '''           
        #KW/hr
        0.0002 * globalPOA * ( 1 + ( 25 - cellTemp ) * .004 )  
          
          
        return ( globalPOA * ( 1 + ( 25 - cellTemp ) * .004 )  )
        
        
##############################################################################################
    ############################################
    #Vant Hoff Degradation Function
    ############################################       
    def rateOfDegEnv( poa, x, cellTemp, refTemp, Tf):
        '''
        HELPER FUNCTION
        
        Find the rate of degradation kenetics using the Fischer model.  
        Degradation kentics model interpolated 50 coatings with respect to 
        color shift, cracking, gloss loss, fluorescense loss, 
        retroreflectance loss, adhesive transfer, and shrinkage.
        
        (ADD IEEE reference)
        
        @param poa                 -float, (Global) Plan of Array irradiance (W/m^2)
        @param x                   -float, fit parameter
        @param cellTemp            -float, solar module cell temperature (C)
        @param refTemp             -float, reference temperature (C) "Chamber Temperature"
        @param Tf                  -float, multiplier for the increase in degradation
                                          for every 10(C) temperature increase
        @return  degradation rate (NEED TO ADD METRIC)  
        '''        
        return poa**(x) * Tf ** ( (cellTemp - refTemp)/10 )



    def rateOfDegChamber( Ichamber , x ):
        '''
        HELPER FUNCTION
        
        Find the rate of degradation kenetics of a simulated chamber. Mike Kempe's 
        calculation of the rate of degradation inside a accelerated degradation chamber. 
        
        (ADD IEEE reference)

        @param Ichamber      -float, Irradiance of Controlled Condition W/m^2
        @param x             -float, fit parameter

        @return  degradation rate of chamber 
        '''        
        return Ichamber ** ( x )



    def accelerationFactor( numerator , denominator ):
        '''
        HELPER FUNCTION
        
        Find the acceleration factor 
        
        (ADD IEEE reference)

        @param numerator      -float, typically the numerator is the chamber settings
        @param denominator    -float, typically the TMY data summation

        @return  degradation rate of chamber (NEED TO ADD METRIC)  
        '''        
        return ( numerator / denominator )
    
    
    
    def vantHoffDeg( x , Ichamber , poa , Toutdoor , Tf , refTemp):    
        '''
        Vant Hoff Irradiance Degradation 
        

        @param x                     -float, fit parameter
        @param Ichamber              -float, Irradiance of Controlled Condition W/m^2
        @param globalPOA             -float or series, Global Plane of Array Irradiance W/m^2
        @param Toutdoor              -pandas series, solar module temperature or Cell temperature (C)
        @param Tf                    -float, multiplier for the increase in degradation
                                          for every 10(C) temperature increase
        @param refTemp               -float, reference temperature (C) "Chamber Temperature"                                          
                                          
        @return  sumOfDegEnv         -float or series, Summation of Degradation Environment 
        @return  avgOfDegEnv         -float or series, Average rate of Degradation Environment
        @return  rateOfDegChamber    -float or series, Rate of Degradation from Simulated Chamber
        @return  accelerationFactor  -float or series, Degradation acceleration factor
        '''  
        rateOfDegEnv = energyCalcs.rateOfDegEnv(poa,
                                                x , 
                                                Toutdoor ,
                                                refTemp ,
                                                Tf )        
        #sumOfDegEnv = rateOfDegEnv.sum(axis = 0, skipna = True)
        avgOfDegEnv = rateOfDegEnv.mean()
            
        rateOfDegChamber = energyCalcs.rateOfDegChamber( Ichamber , x )
        
        accelerationFactor = energyCalcs.accelerationFactor( rateOfDegChamber , avgOfDegEnv)
        
        return  accelerationFactor
    
    
    
##############################################################################################
    ############################################
    #Vant Hoff Environmental Characterization
    ############################################

    def ToeqVantHoff( Tf, Toutdoor ):
        '''
        ToeqVantHoff()
        
        Get the Vant Hoff temperature equivalent (C)
        
        @param Tf           -float, multiplier for the increase in degradation
                                    for every 10(C) temperature increase  
        @param Toutdoor     -pandas series, solar module temperature or Cell temperature (C)    
    
        @return Toeq        -float, Vant Hoff temperature equivalent (C)                          
        '''
        toSum = Tf ** ( Toutdoor / 10 )
        summation = toSum.sum(axis = 0, skipna = True)
    
        Toeq = (10 / np.log ( Tf ) ) * np.log ( summation / len(Toutdoor) )
        
        return Toeq
    
    
    
    def IwaVantHoff( globalPOA , x , Tf , Toutdoor , ToeqVantHoff):
        '''
        IwaVantHoff()
        
        IWa : Environment Characterization (W/m^2)
        *for one year of degredation the controlled environmnet lamp settings will 
            need to be set to IWa
            
        @param globalPOA             -float or series, Global Plane of Array Irradiance W/m^2    
        @param x                     -float, fit parameter  
        @param Tf           -float, multiplier for the increase in degradation
                                    for every 10(C) temperature increase   
        @param Toutdoor     -pandas series, solar module temperature or Cell temperature (C)      
        @param ToeqVantHoff        -float, Vant Hoff temperature equivalent (C)
                               
        @return Iwa        -float, Environment Characterization (W/m^2)
        '''
        toSum = (globalPOA ** x) * (Tf ** ( (Toutdoor - ToeqVantHoff)/10 ))
        summation = toSum.sum(axis = 0, skipna = True)
        
        Iwa = ( summation / len(globalPOA) ) ** ( 1 / x )
        
        return Iwa    




##############################################################################################
    ############################################
    #Arrhenius Degradation Function
    ############################################
        
        
    def arrheniusDenominator( poa , x, rh_outdoor , n , Toutdoor , Ea):
        '''
        arrheniusDenominator()
        
        Calculate the rate of degredation of the Environmnet 

        @param poa                 -float, (Global) Plan of Array irradiance (W/m^2)
        @param x             -float, fit parameter         
        @rh_outdoor                -pandas series, Relative Humidity of material of interest 
                                            Acceptable relative humiditys can be calculated 
                                            from the below functions
                                                 RHbacksheet()
                                                 RHbackEncap()
                                                 RHfront()
                                                 RHsurfaceOutside()
        @param n             -float, fit parameter for relative humidity 
        @param Toutdoor      -pandas series, solar module temperature or Cell temperature (C)   
        @param Ea            -float, Degredation Activation Energy (kJ/mol)        
        
    
        @return  degradation rate of environment 
        '''        
        return poa**(x) * rh_outdoor**(n) * np.exp( - ( Ea/ ( 0.00831446261815324 * (Toutdoor + 273.15)  )))
    
    
    
    def arrheniusNumerator( Ichamber , x , rhChamber, n ,  Ea , Tchamber ):
        '''
        arrheniusNumerator()
        
        Find the rate of degradation of a simulated chamber.  
        
    
        @param Ichamber      -float, Irradiance of Controlled Condition W/m^2
        @param x             -float, fit parameter
        @param rhChamber     -float, Relative Humidity of Controlled Condition (%)
                                      -EXAMPLE: "50 = 50% NOT .5 = 50%" 
        @param n             -float, fit parameter for relative humidity                                      
        @param Ea            -float, Degredation Activation Energy (kJ/mol)
        @param Tchamber      -float, reference temperature (C) "Chamber Temperature"
    
        @return  degradation rate of chamber 
        '''        
        return Ichamber ** ( x ) * rhChamber ** (n) * np.exp( - ( Ea/ ( 0.00831446261815324 * (Tchamber+273.15)  )))
    
    
    
    def arrheniusCalc( x , Ichamber , rhChamber , n , rh_outdoor , globalPOA , Tchamber , Toutdoor,  Ea):    
        '''
        arrheniusCalc()
        
        Calculate the Acceleration Factor between the rate of degredation of a 
        modeled environmnet versus a modeled controlled environmnet
        
        Example: "If the AF=25 then 1 year of Controlled Environment exposure 
                    is equal to 25 years in the field"

        @param x                     -float, fit parameter
        @param Ichamber              -float, Irradiance of Controlled Condition W/m^2
        @param rhChamber             -float, Relative Humidity of Controlled Condition (%)
                                                        -EXAMPLE: "50 = 50% NOT .5 = 50%"
        @param n                     -float, fit parameter for relative humidity  
        @rh_outdoor                  -pandas series, Relative Humidity of material of interest 
                                                    Acceptable relative humiditys can be calculated 
                                                    from the below functions
                                                         RHbacksheet()
                                                         RHbackEncap()
                                                         RHfront()
                                                         RHsurfaceOutside()
        @param globalPOA             -pandas series, Global Plane of Array Irradiance W/m^2
        @param Tchamber              -float, reference temperature (C) "Chamber Temperature"  
        @param Toutdoor              -pandas series, solar module temperature or Cell temperature (C)
        @param Ea                    -float, Degredation Activation Energy (kJ/mol) 
                                        
                                          
        @return  sumOfDegEnv         -pandas series, Summation of Degradation Environment 
        @return  avgOfDegEnv         -pandas series, Average rate of Degradation Environment
        @return  rateOfDegChamber    -pandas series, Rate of Degradation from Simulated Chamber
        @return  accelerationFactor  -pandas series, Degradation acceleration factor
        '''  
        arrheniusDenominator = energyCalcs.arrheniusDenominator(globalPOA,
                                                        x , 
                                                        rh_outdoor,
                                                        n,
                                                        Toutdoor ,
                                                        Ea )        
        

        AvgOfDenominator = arrheniusDenominator.mean()
            
        arrheniusNumerator = energyCalcs.arrheniusNumerator( Ichamber , x , rhChamber, n ,  Ea , Tchamber )
        
        
        accelerationFactor = energyCalcs.accelerationFactor( arrheniusNumerator , AvgOfDenominator)
        
        return accelerationFactor
        




##############################################################################################
    ############################################
    #Arrhenius Environmental Characterization
    ############################################

    def TeqArrhenius( Toutdoor , Ea ):
        ''' 
        TeqArrhenius()
        
        Get the Temperature equivalent required for the settings of the controlled environment
    
        Calculation is used in determining Arrhenius Environmental Characterization
    
        @param Toutdoor              -pandas series, solar module temperature or Cell temperature (C)
        @param Ea                    -float, Degredation Activation Energy (kJ/mol)    
        
        @return Teq                  -float, Temperature equivalent (Celsius) required 
                                                for the settings of the controlled environment
        '''
        summationFrame = np.exp( - ( Ea/ ( 0.00831446261815324 * (Toutdoor + 273.15)  )))
        sumForTeq = summationFrame.sum(axis = 0, skipna = True)
        Teq = -( (Ea) /  ( 0.00831446261815324 * np.log ( sumForTeq / len(Toutdoor) ) ) )
        # Convert to celsius
        Teq = Teq - 273.15
        
        return Teq
    
    
    
    def RHwaArrhenius( rh_outdoor , n , Ea , Toutdoor, Teq ):
        ''' 
        RHwaArrhenius()
        
        Get the Relative Humidity Weighted Average
    
        Calculation is used in determining Arrhenius Environmental Characterization
    
        @rh_outdoor                -pandas series, Relative Humidity of material of interest 
                                            Acceptable relative humiditys can be calculated 
                                            from the below functions
                                                 RHbacksheet()
                                                 RHbackEncap()
                                                 RHfront()
                                                 RHsurfaceOutside()
        @param n             -float, fit parameter for relative humidity       
        @param Ea            -float, Degredation Activation Energy (kJ/mol)                   
        @param Toutdoor      -pandas series, solar module temperature or Cell temperature (C)
        @param Teq           -float, Temperature equivalent (Celsius) required 
                                       for the settings of the controlled environment
        
        @return RHwa         -float, Relative Humidity Weighted Average (%)
        '''
    
        summationFrame = (rh_outdoor ** n ) * np.exp( - ( Ea/ ( 0.00831446261815324 * (Toutdoor + 273.15)  )))
        sumForRHwa = summationFrame.sum(axis = 0, skipna = True)
        RHwa =  (sumForRHwa / ( len(summationFrame) * np.exp( - ( Ea/ ( 0.00831446261815324 * (Teq + 273.15)  ))))) ** (1/n)
    
        return RHwa
    
      
        
    def IwaArrhenius( poa , x ,  rh_outdoor , n , Toutdoor , Ea , RHwa, Teq):
        '''
        IwaArrhenius()
        
        IWa : Environment Characterization (W/m^2)
        *for one year of degredation the controlled environmnet lamp settings will 
            need to be set at IWa
        
    
        @param poa           -float, (Global) Plan of Array irradiance (W/m^2)
        @param x             -float, fit parameter         
        @rh_outdoor                -pandas series, Relative Humidity of material of interest 
                                            Acceptable relative humiditys can be calculated 
                                            from the below functions
                                                 RHbacksheet()
                                                 RHbackEncap()
                                                 RHfront()
                                                 RHsurfaceOutside()
        @param n             -float, fit parameter for relative humidity 
        @param Toutdoor      -pandas series, solar module temperature or Cell temperature (C)   
        @param Ea            -float, Degredation Activation Energy (kJ/mol)    
        @param RHwa          -float, Relative Humidity Weighted Average (%) 
        @param Teq           -float, Temperature equivalent (Celsius) required 
                                   for the settings of the controlled environment
        
        @return Iwa             -float, Environment Characterization (W/m^2)
        '''
        numerator = poa**(x) * rh_outdoor**(n) * np.exp( - ( Ea/ ( 0.00831446261815324 * (Toutdoor + 273.15) )))
        sumOfNumerator = numerator.sum(axis = 0, skipna = True)
    
        denominator = (len(numerator)) * ((RHwa)**n )  * (np.exp( - ( Ea/ ( 0.00831446261815324 * (Teq + 273.15)  ))))  
    
        IWa = ( sumOfNumerator / denominator )**(1/x)
    
        return IWa




############
# Misc. Functions for Energy Calcs
############ 

    
    def rH_Above85( rH ):    
        '''
        HELPER FUNCTION
        
        rH_Above85()
        
        Determine if the relative humidity is above 85%.  
        
        @param rH          -float, Relative Humidity %
        @return                   -Boolean, True if the relative humidity is 
                                            above 85% or False if the relative 
                                            humidity is below 85%
        '''         
        if rH > 85:
            return( True )
        else:
            return ( False )
     
        
   
    def hoursRH_Above85( df ):      
        '''
        HELPER FUNCTION
        
        hoursRH_Above85()
        
        Count the number of hours relative humidity is above 85%.  
        
        @param    df     -dataFrame, dataframe containing Relative Humidity %
        @return          -int, number of hours relative humidity is above 85%
        
        '''         
        booleanDf = df.apply(lambda x: energyCalcs.rH_Above85( x ) )
        return( booleanDf.sum() )
        
  

    def whToGJ( wh ):
        '''
        HELPER FUNCTION
        
        whToGJ()
        
        Convert Wh/m^2 to GJ/m^-2 
        
        @param wh          -float, Wh/m^2
        @return                   -float, GJ/m^-2
        
        '''    
        return( 0.0000036 * wh )
    
    

    def gJtoMJ( gJ ):
        '''
        HELPER FUNCTION
        
        gJtoMJ()
        
        Convert GJ/m^-2 to MJ/y^-1
        
        @param gJ          -float, Wh/m^2
        @return            -float, GJ/m^-2
        
        '''    
        return( gJ * 1000 )
  

    