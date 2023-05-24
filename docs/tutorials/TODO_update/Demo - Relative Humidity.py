#!/usr/bin/env python
# coding: utf-8

# # Demo - Relative Humidity
# 
# This Demo is designed to show the functionality of the Relative Humidity Class 
# and the energyCalcs class.
# 
# energyCalcs class contains the Vant Hoff acceleration factor and Arrhenius 
# Equations Acceleration Factor
# 
# To demonstrate we use a processed TMY dataset from "Saudi Arabia, Riyad" that has already calculated the 
# Module Temperature using the pvlib library.
# 
# There are currently 4 selections for relative Humidity
#    
#    1) RHsurfaceOutside : Relative Humidity of the Surface of a Solar Module 
#    
#    2) RHfrontEncapsulant : Relative Humidity of the Frontside Encapsulant of a Solar Module
#    
#    3) RHbackEncapsulant : Relative Humidity of the backside Encapsulant of a Solar Module 
#    
#    4) RHbacksheet : Relative 

# In[ ]:


import pandas as pd
import Relative_Humidity_for_Solar_Modules

#import data
locationData , processedData_df = pd.read_pickle( '722024TYA.pickle' )

#Get the Relative Humidity of outside environment (TMY raw data)
rH_ambient = processedData_df['Relative humidity(%)']

#Get the ambient temperature of outside environment (TMY raw data)
ambient_temp = processedData_df['Dry-bulb temperature(C)']

#Get the temperature of the module (Calulated with pvlib to obtain module temperature)
#We will use open_rack_cell_glassback for this demo
surface_temp = processedData_df['Module Temperature(roof_mount_cell_glassback)(C)']


# ### parameters

# In[ ]:


#So : Encapsulant solubility prefactor (g/cm3).  The suggested value for EVA is 1.81390702(g/cm3)
So = 1.81390702
#Eas : Encapsulant solubility activation energy in (kJ/mol).  The suggested value for EVA is 16.729(kJ/mol) 
Eas = 16.729
#Ead : Encapsulant diffusivity activation energy in (kJ/mol) The suggested value for EVA is 38.14(kJ/mol).
Ead = 38.14
#SDW: Diffusivity weighted water content 
SDw = relativeHumidity.SDw( rH_ambient , ambient_temp , surface_temp, So ,  Eas , Ead)
#WVTRo : Water Vapor Transfer Rate prefactor (g/m2/day). The suggested value for EVA is  7970633554(g/m2/day).
WVTRo = 7970633554
#EaWVTR : Water Vapor Transfer Rate activation energy (kJ/mol) .
# It is suggested to use 0.15(mm) thick PET as a default for the backsheet and set EaWVTR=55.0255(kJ/mol)
EaWVTR = 55.0255
#l : Thickness of the backside encapsulant (mm). The suggested value for encapsulat in EVA,  l=0.5(mm)
l = .5


# ### Relative Humidity of a Solar Module 

# In[ ]:


#Get the Relative Humidity of the outside surface of the Solar Module.
RHsurfaceOutside = pd.Series(name="RHsurfaceOutside" , data=                       relativeHumidity.RHsurfaceOutside(rH_ambient, ambient_temp, surface_temp ) )

#Get the Relative Humidity of the Frontside Encapsulant of a Solar Module.
RHfrontEncap = pd.Series(name="RHfront" , data=                       relativeHumidity.RHfront( surface_temp, SDw , So , Eas) )

#Get the Relative Humidity of the Backside Encapsulant of a Solar Module 
RHbackEncap = pd.Series(name="RHbackEncap" , data=                       relativeHumidity.RHbackEncap( rH_ambient , ambient_temp , surface_temp , WVTRo , EaWVTR , So , l , Eas ) )
  
#Get the Relative Humidity of the backside Back sheet of a Solar Module 
RHbacksheet = pd.Series(name="RHbacksheet" , data=                       relativeHumidity.RHbacksheet( RHbackEncap , RHsurfaceOutside ) )


# ### Vant Hoff Characterization

# In[ ]:


#PARAMETERS
#Tf = multiplier for the increase in degradation for every 10(C) temperature increase
Tf = 1.41
#x = fit parameter
x = .64

#Temperature equivalent for Vant Hoff Equation
VantHoff_Toeq = energyCalcs.ToeqVantHoff( Tf, surface_temp )

#IWa : Environment Characterization (W/m^2)
#*for one year of degredation the controlled environmnet lamp settings will 
#    need to be set to IWa
VantHoff_Iwa = energyCalcs.IwaVantHoff( processedData_df['POA Global(W/m^2)'] ,
                                                         x , 
                                                         Tf , 
                                                         surface_temp ,
                                                         VantHoff_Toeq)


# ### Vant Hoff Equation Acceleration Factor

# In[ ]:


#Ichamber = Irradiance of the chamber settings
Ichamber = 1000
#Reference temperature of the chamber (C)
refTemp = 60

#Get the Vant Hoff equation acceleration factor 
VantHoff_AF = energyCalcs.vantHoffDeg( x , 
            Ichamber , 
            processedData_df['POA Global(W/m^2)'] , 
            surface_temp , 
            Tf , 
            refTemp)


# ### Arrhenius Characterization

# In[ ]:


# Ea = Degredation Activation Energy (kJ/mol)
Ea = 28

#Arrhenius_Teq = Temperature equivalent
Arrhenius_Teq = energyCalcs.TeqArrhenius( surface_temp , Ea )

# n = fit parameter for relative humidity 
n=1

#RHwa : Relative Humidity Weighted Average
#Use the Relative humidity surface Outside 
Arrhenius_RHwa = energyCalcs.RHwaArrhenius( RHsurfaceOutside ,
                                            n , 
                                            Ea ,
                                            surface_temp, 
                                            Arrhenius_Teq )

Arrhenius_Iwa = energyCalcs.IwaArrhenius( processedData_df['POA Global(W/m^2)'],
                                                           x ,
                                                           RHsurfaceOutside ,
                                                           n ,
                                                           surface_temp ,
                                                           Ea ,
                                                           Arrhenius_RHwa,
                                                           Arrhenius_Teq)


# ### Arrhenius Equation Acceleration Factor
# 

# In[ ]:


#rhChamber = Relative Humidity of the controlled environment % "chamber"
rhChamber = 15
#Get the Arrehnius equation acceleration factor
Arrehnius_AF = energyCalcs.arrheniusCalc( x ,
                                         Ichamber ,
                                         rhChamber ,
                                         n ,
                                         RHsurfaceOutside ,
                                         processedData_df['POA Global(W/m^2)'] ,
                                         refTemp ,
                                         surface_temp,
                                         Ea)

