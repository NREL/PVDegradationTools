# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:55:35 2020

@author: dholsapp
"""

"""
Contains energy algorithms for processing.

@author: Derek Holsapple
"""

import numpy as np
from numba import jit    
   

class relativeHumidity:

        
        ###########
        # Surface RH       
        ###########
        
    def Psat( temp ):
        '''
        Function to generate the point of saturation dependent on temperature
        Calculation created by Michael Kempe, implemented by Derek Holsapple
        
        3rd, 4th, 5th, and 6th order polynomial fits were explored.  The best fit 
        was determined to be the 4th
        
        @param temp    float, temperature in Celsius 
        
        @return Psat   float, point of saturation
        '''
        Psat = np.exp( -0.000000002448137*temp**4  \
                       +0.000001419572*temp**3  \
                       -0.0003779559*temp**2  \
                       +0.07796986*temp  \
                       -0.5796729   )
        return Psat
        
        
    
    def RHsurfaceOutside(rH_ambient, ambient_temp, surface_temp ):
        '''
        Function calculates the Relative Humidity of a Solar Panel Surface
    
        @param rH_ambient      float, The ambient outdoor environmnet relative humidity 
        @param ambient_temp    float, The ambient outdoor environmnet temperature in Celsius
        @param surface_temp    float, The surface temperature in Celsius of the solar panel module 
        
        @return rH_Surface     float, The relative humidity of the surface of a solar module                    
        '''
        rH_Surface = rH_ambient*( relativeHumidity.Psat( ambient_temp ) / relativeHumidity.Psat( surface_temp )  )

        return rH_Surface
    

    
        ###########
        # Front Encapsulant RH       
        ###########
        
        
        
        
    def SDwNumerator( So ,  Eas , Ead , rH_ambient, ambient_temp, surface_temp):
        '''
        SDwNumerator()

        Calculation is used in determining Relative Humidity of Frontside Solar
        module Encapsulant

        The function returns values needed for the numerator of the Diffusivity weighted water content equation. 
        This function will return a pandas series prior to summation of the numerator 


        @param So                   -float, Encapsulant solubility prefactor in [g/cm3] 
                                                1.81390702(g/cm3) is the suggested value for EVA.                           
        @param Eas                  -float, Encapsulant solubility activation energy in [kJ/mol] 
                                                16.729(kJ/mol) is the suggested value for EVA.  
        @param Ead                  -float, Encapsulant diffusivity activation energy in [kJ/mol] 
                                                38.14(kJ/mol) is the suggested value for EVA. 
        @param rH_ambient           -pandas series (float), The ambient outdoor 
                                            environmnet relative humidity in (%) 
                                            -EXAMPLE: "50 = 50% NOT .5 = 50%"
        @param ambient_temp         -pandas series (float), The ambient outdoor
                                                    environmnet temperature in Celsius                                    
        @param surface_temp         -pandas series (float), The surface temperature 
                                                    in Celsius of the solar panel module 
                        
                        
        @return SDwNumerator_series -pandas series (float), numerator of the Sdw equation
                                                    prior to summation
        ''' 
        
        #Get the relative humidity of the surface
        RH_surface = relativeHumidity.RHsurfaceOutside(rH_ambient, ambient_temp, surface_temp )
        
        #Generate a series of the numerator values "prior to summation"
        SDwNumerator_series = So * np.exp( - ( Eas / (0.00831446261815324 * (surface_temp + 273.15) ))) * \
                        RH_surface * np.exp( - ( Ead / (0.00831446261815324 * (surface_temp + 273.15) )))
        
        return SDwNumerator_series



    def SDwDenominator( Ead , surface_temp):
        '''
        SDwDenominator()

        Calculation is used in determining Relative Humidity of Frontside Solar
        module Encapsulant

        The function returns values needed for the denominator of the Diffusivity
        weighted water content equation(SDw). This function will return a pandas 
        series prior to summation of the denominator 
        

        @param Ead                  -float, Encapsulant diffusivity activation energy in [kJ/mol] 
                                                38.14(kJ/mol) is the suggested value for EVA.    
        @param surface_temp         -pandas series (float), The surface temperature 
                                                    in Celsius of the solar panel module 
                         
                            
        @return SDwDenominator      -pandas series (float), denominator of the SDw equation
                                                    prior to summation                                                    
        '''
        
        SDwDenominator =  np.exp( - ( Ead / (0.00831446261815324 * (surface_temp + 273.15) )))
        return SDwDenominator



    def SDw( So ,  Eas , Ead  , rH_ambient , ambient_temp , surface_temp):
        '''
        SDw()

        Calculation is used in determining Relative Humidity of Frontside Solar
        module Encapsulant

        The function calculates the Diffusivity weighted water content equation. 


        @param So                   -float, Encapsulant solubility prefactor in [g/cm3] 
                                                1.81390702(g/cm3) is the suggested value for EVA.                           
        @param Eas                  -float, Encapsulant solubility activation energy in [kJ/mol] 
                                                16.729(kJ/mol) is the suggested value for EVA.  
        @param Ead                  -float, Encapsulant diffusivity activation energy in [kJ/mol] 
                                                38.14(kJ/mol) is the suggested value for EVA. 
        @param rH_ambient           -pandas series (float), The ambient outdoor 
                                            environmnet relative humidity in (%) 
                                            -EXAMPLE: "50 = 50% NOT .5 = 50%"
        @param ambient_temp         -pandas series (float), The ambient outdoor
                                                    environmnet temperature in Celsius                                    
        @param surface_temp         -pandas series (float), The surface temperature 
                                                    in Celsius of the solar panel module 
                        
                        
        @return SDw                 -float, Diffusivity weighted water content       
        '''
        numerator = relativeHumidity.SDwNumerator( So ,  Eas , Ead , rH_ambient, ambient_temp, surface_temp)
        #get the summation of the numerator
        numerator = numerator.sum(axis = 0, skipna = True)

        denominator = relativeHumidity.SDwDenominator( Ead , surface_temp)
        #get the summation of the denominator
        denominator = denominator.sum(axis = 0, skipna = True)

        SDw = (numerator / denominator)/100
        
        return SDw
    
    
    
    def RHfront( SDw , So , Eas, surface_temp ):
        '''
        RHfront()
        
        Function returns Relative Humidity of Frontside Solar Module Encapsulant
        
        @param SDw                  -float, Diffusivity weighted water content 
                                                *See energyCalcs.SDw() function        
        @param So                   -float, Encapsulant solubility prefactor in [g/cm3] 
                                                1.81390702(g/cm3) is the suggested value for EVA.                           
        @param Eas                  -float, Encapsulant solubility activation energy in [kJ/mol] 
                                                16.729(kJ/mol) is the suggested value for EVA.         
        @param surface_temp         -pandas series (float), The surface temperature
                                                    in Celsius of the solar panel module
                                                    "module temperature (C)"
                                                    
        @return RHfront_series      -pandas series (float), Relative Humidity of 
                                                Frontside Solar module Encapsulant                                
        '''
        RHfront_series = (SDw / ( So * np.exp( - ( Eas / (0.00831446261815324 * \
                         (surface_temp + 273.15) ))))) * 100
        
        return RHfront_series



        ###########
        # Back Encapsulant Relative Humidity       
        ###########
    def Csat( So , Eas, surface_temp ):
        '''
        Csat()
        
        Calculation is used in determining Relative Humidity of Backside Solar
        Module Encapsulant
        
        Function returns Saturation of Water Concentration (g/cm³)
               
        @param So                   -float, Encapsulant solubility prefactor in [g/cm3] 
                                                1.81390702(g/cm3) is the suggested value for EVA.                           
        @param Eas                  -float, Encapsulant solubility activation energy in [kJ/mol] 
                                                16.729(kJ/mol) is the suggested value for EVA.         
        @param surface_temp         -pandas series (float), The surface temperature
                                                    in Celsius of the solar panel module
                                                    "module temperature (C)"
                                                    
        @return Csat      -pandas series (float), Saturation of Water Concentration (g/cm³)                             
        '''
        #Saturation of water concentration
        Csat = So * np.exp( - ( Eas / 0.00831446261815324 / (273.15 + surface_temp )   )    )
        return Csat


    def Ceq( Csat , rH_SurfaceOutside ):
        '''
        Ceq()
        
        Calculation is used in determining Relative Humidity of Backside Solar
        Module Encapsulant
        
        Function returns Equilibration water concentration (g/cm³)
 
               
        @param Csat                 -pandas series (float), Saturation of Water Concentration                            
        @param rH_SurfaceOutside    -pandas series (float), The relative humidity of the surface of a solar module (%)
                                                    
        @return Csat                -pandas series (float),Equilibration water concentration (g/cm³)                            
        '''
        Ceq = Csat * (rH_SurfaceOutside/100)
        return Ceq



    #Returns a numpy array
    @jit(nopython=True)
    def Ce_numba( start , WVTRo , EaWVTR , surface_temp , So , l , RH_surface , Eas ):
        '''
        Ce_numba()

        Calculation is used in determining Relative Humidity of Backside Solar
        Module Encapsulant
        
        This function returns a numpy array of the Concentration of water in the 
        encapsulant at every time step         
        
        Numba was used to isolate recursion requiring a for loop
        Numba Functions compile and run in machine code but can not use pandas (Very fast).

        @param start         -float, initial value of the Concentration of water in the encapsulant
                                        -currently takes the first value produced from
                                            the Ceq(Saturation of Water Concentration) as a point
                                            of acceptable equilibrium
        @param datapoints    -int, the number of datapoints to calculate from data
                                    EXAMPLES:  1 year is "8760hrs" of datapoints NOT "8759hrs"
        @param WVTRo         -float, Water Vapor Transfer Rate prefactor (g/m2/day). 
                                            The suggested value for EVA is  7970633554(g/m2/day).
        @param EaWVTR        -float, Water Vapor Transfer Rate activation energy (kJ/mol) . 
                                            It is suggested to use 0.15(mm) thick PET as a default 
                                            for the backsheet and set EaWVTR=55.0255(kJ/mol)        
        @param surface_temp  -list (float), The surface temperature
                                            in Celsius of the solar panel module
                                            "module temperature (C)" 
        @param So            -float, Encapsulant solubility prefactor in [g/cm3] 
                                            1.81390702(g/cm3) is the suggested value for EVA.  
        @param l             -float, Thickness of the backside encapsulant (mm). 
                                            The suggested value for encapsulat is EVA, l=0.5(mm)   
        @param rH_Surface    -list (float), The relative humidity of the surface of a solar module (%)                                             
                                            -EXAMPLE: "50 = 50% NOT .5 = 50%"
        @param Eas           -float, Encapsulant solubility activation energy in [kJ/mol] 
                                                16.729(kJ/mol) is the suggested value for EVA. 
                                                
                                                
        @retrun Ce_list      -numpy_array, Concentration of water in the encapsulant at every time step                                    
        '''
        
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
    
    
    
    def RHbackEncap( rH_ambient , ambient_temp , surface_temp , WVTRo , EaWVTR , So , l , Eas ):
        '''
        RHbackEncap()
        
        Function to calculate the Relative Humidity of Backside Solar Module Encapsulant
        and return a pandas series for each time step        
        
        @param rH_ambient    -pandas series (float), The ambient outdoor 
                                                environmnet relative humidity in (%) 
                                                -EXAMPLE: "50 = 50% NOT .5 = 50%"
        @param ambient_temp  -pandas series (float), The ambient outdoor
                                                    environmnet temperature in Celsius                                    
        @param surface_temp  -list (float), The surface temperature
                                            in Celsius of the solar panel module
                                            "module temperature (C)"                                     
        @param WVTRo         -float, Water Vapor Transfer Rate prefactor (g/m2/day). 
                                            The suggested value for EVA is  7970633554(g/m2/day).                                    
        @param EaWVTR        -float, Water Vapor Transfer Rate activation energy (kJ/mol) . 
                                            It is suggested to use 0.15(mm) thick PET as a default 
                                            for the backsheet and set EaWVTR=55.0255(kJ/mol)                                     
        @param So            -float, Encapsulant solubility prefactor in [g/cm3] 
                                            1.81390702(g/cm3) is the suggested value for EVA.  
        @param l             -float, Thickness of the backside encapsulant (mm). 
                                            The suggested value for encapsulat is EVA, l=0.5(mm)   
        @param Eas           -float, Encapsulant solubility activation energy in [kJ/mol] 
                                                16.729(kJ/mol) is the suggested value for EVA.                                     
       
        
        @return RHback_series      -pandas series (float), Relative Humidity of 
                                                Backside Solar Module Encapsulant                             
        '''
        
        RH_surface = relativeHumidity.RHsurfaceOutside(rH_ambient, ambient_temp, surface_temp )
        
        
        Csat = relativeHumidity.Csat( So , Eas, surface_temp )
        Ceq = relativeHumidity.Ceq( Csat , RH_surface )

        start = Ceq[0]
        
        #Need to convert these series to numpy arrays for numba function
        surface_temp_numba = surface_temp.to_numpy()
        RH_surface_numba = RH_surface.to_numpy()
        
        Ce_nparray = relativeHumidity.Ce_numba( start , WVTRo , EaWVTR , surface_temp_numba , So , l , RH_surface_numba , Eas )
        
        RHback_series = 100 * (Ce_nparray / (So * np.exp(-( (Eas) / (0.00831446261815324 * (surface_temp + 273.15))  )) ))
        
        return RHback_series


        ###########
        # Back Sheet Relative Humidity       
        ###########
        
    def RHbacksheet( RHbackEncap , RHsurfaceOutside ):
        '''
        RHbacksheet()
        
        Function to calculate the Relative Humidity of Backside BackSheet of a Solar Module 
        and return a pandas series for each time step

        @param RHbackEncap      -pandas series (float), Relative Humidity of 
                                                Frontside Solar module Encapsulant
                                                *See RHbackEncap()
        @return rH_Surface     -pandas series (float), The relative humidity of 
                                               the surface of a solar module
                                              *See RHsurfaceOutside()
                                              
                                              
        @return RHbacksheet_series   -pandas series (float), Relative Humidity of 
                                                Backside Backsheet of a Solar Module                                  
        '''
        return (RHbackEncap + RHsurfaceOutside)/2




    