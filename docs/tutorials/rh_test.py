# -*- coding: utf-8 -*-
"""
author: Matthew Brown
date:   Tue Aug 24 11:33:10 2021
"""

import pandas as pd
#import Relative_Humidity_for_Solar_Modules
import PVDegradationTools as PVD
import os

os.chdir('C:/Users/mbrown2/Documents/GitHub/PVDegradationTools/docs/tutorials')
path = os.path.join('..','..','PVDegradationTools','data','722024TYA.pickle')
#import data
locationData , processedData_df = pd.read_pickle( path )

#Get the Relative Humidity of outside environment (TMY raw data)
rH_ambient = processedData_df['Relative humidity(%)']

#Get the ambient temperature of outside environment (TMY raw data)
ambient_temp = processedData_df['Dry-bulb temperature(C)']

#Get the temperature of the module (Calulated with pvlib to obtain module temperature)
#We will use open_rack_cell_glassback for this demo
surface_temp = processedData_df['Module Temperature(roof_mount_cell_glassback)(C)']


dni = processedData_df['Direct normal irradiance(W/m^2)']

## Variables and Constants
#So : Encapsulant solubility prefactor (g/cm3).  The suggested value for EVA is 1.81390702(g/cm3)
So = 1.81390702
#Eas : Encapsulant solubility activation energy in (kJ/mol).  The suggested value for EVA is 16.729(kJ/mol) 
Eas = 16.729
#Ead : Encapsulant diffusivity activation energy in (kJ/mol) The suggested value for EVA is 38.14(kJ/mol).
Ead = 38.14
#SDW: Diffusivity weighted water content 
SDw = PVD.relativeHumidity.SDw( rH_ambient , ambient_temp , surface_temp, So ,  Eas , Ead)
#WVTRo : Water Vapor Transfer Rate prefactor (g/m2/day). The suggested value for EVA is  7970633554(g/m2/day).
WVTRo = 7970633554
#EaWVTR : Water Vapor Transfer Rate activation energy (kJ/mol) .
# It is suggested to use 0.15(mm) thick PET as a default for the backsheet and set EaWVTR=55.0255(kJ/mol)
EaWVTR = 55.0255
#l : Thickness of the backside encapsulant (mm). The suggested value for encapsulat in EVA,  l=0.5(mm)
l = .5

# ======= RH Calcs =======
#Get the Relative Humidity of the outside surface of the Solar Module.
RHsurfaceOutside = pd.Series(name="RHsurfaceOutside" , data= \
                      PVD.relativeHumidity.RHsurfaceOutside(rH_ambient, ambient_temp, surface_temp ) )

#Get the Relative Humidity of the Frontside Encapsulant of a Solar Module.
RHfrontEncap = pd.Series(name="RHfront" , data= \
                      PVD.relativeHumidity.RHfront( surface_temp, SDw , So , Eas) )
# In[1]:



# In[2]:
#Get the Relative Humidity of the Backside Encapsulant of a Solar Module 
RHbackEncap = pd.Series(name="RHbackEncap" , data= \
                      PVD.relativeHumidity.RHbackEncap( rH_ambient , ambient_temp , surface_temp , WVTRo , EaWVTR , So , l , Eas ) )
  
#Get the Relative Humidity of the backside Back sheet of a Solar Module 
RHbacksheet = pd.Series(name="RHbacksheet" , data= \
                      PVD.relativeHumidity.RHbacksheet( RHbackEncap , RHsurfaceOutside ) )

    
# ===== Results ======
table = {'DNI':dni,
         'RH_ambient':rH_ambient,
         'RHsurfOut':RHsurfaceOutside,
         'RHfrontEncap':RHfrontEncap,
         'RHbackEncap':RHbackEncap,
         'RHbackSheet':RHbacksheet}
demo = pd.DataFrame(table)
demo.iloc[:24]