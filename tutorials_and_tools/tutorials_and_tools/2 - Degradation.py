#!/usr/bin/env python
# coding: utf-8

# # 2 - Degradation
# Degradation and Solder Fatigue
#
# **Requirements**:
# - compatible weather file (PSM3, TMY3, EPW)
# - Accelerated testing chamber parameters
#     - chamber irradiance [W/m^2]
#     - chamber temperature [C]
#     - chamber humidity [%]
# - Activation energies for test material [kJ/mol]
#
# **Objectives**:
# 1. Read in the weather data
# 2. Generate basic modeling data
# 3. Calculate VantHoff degradation acceleration factor
# 4. Calculate Arrhenius degradation acceleration factor
# 5. Quick Method
# 5. Solder Fatigue

# In[1]:


# if running on google colab, uncomment the next line and execute this cell to install the dependencies and prevent "ModuleNotFoundError" in later cells:
# !pip install pvdeg==0.3.3


# In[2]:


import os
import pandas as pd

import pvdeg
from pvdeg import DATA_DIR


# In[3]:


# This information helps with debugging and getting support :)
import sys
import platform

print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)
print("pvdeg version ", pvdeg.__version__)


# ## 1. Read In the Weather File
#
# This is usually the first step. Use a PSM3, TMY3, or EPW file. For this demo, use the provided PSM3 weather file.

# In[2]:


PSM_FILE = os.path.join(DATA_DIR, "psm3_demo.csv")
WEATHER, META = pvdeg.weather.read(PSM_FILE, "psm")


# ## 2. Generate Basic Modeling Data
#
# For this tutorial we will need solar position, POA, PV cell and module temperature. Let's gernate those individually with their respective functions.

# In[3]:


sol_pos = pvdeg.spectral.solar_position(weather_df=WEATHER, meta=META)

poa_df = pvdeg.spectral.poa_irradiance(
    weather_df=WEATHER, meta=META, sol_position=sol_pos
)

temp_cell = pvdeg.temperature.cell(weather_df=WEATHER, meta=META, poa=poa_df)

temp_module = pvdeg.temperature.module(weather_df=WEATHER, meta=META, poa=poa_df)


# ## 3. VantHoff Degradation
#
# Van 't Hoff Irradiance Degradation
#
# For one year of degredation the controlled environmnet lamp settings will need to be set to IWa.
#
# As with most `pvdeg` functions, the following functions will always require two arguments (weather_df and meta)

# In[4]:


# chamber irradiance (W/m^2)
I_chamber = 1000
# chamber temperature (C)
temp_chamber = 60

# calculate the VantHoff Acceleration factor
vantHoff_deg = pvdeg.degradation.vantHoff_deg(
    weather_df=WEATHER,
    meta=META,
    I_chamber=I_chamber,
    temp_chamber=temp_chamber,
    poa=poa_df,
    temp_cell=temp_cell,
)

# calculate the VantHoff weighted irradiance
irr_weighted_avg_v = pvdeg.degradation.IwaVantHoff(
    weather_df=WEATHER, meta=META, poa=poa_df, temp_cell=temp_cell
)


# ## 4. Arrhenius
# Calculate the Acceleration Factor between the rate of degredation of a modeled environmnet versus a modeled controlled environmnet
#
# Example: "If the AF=25 then 1 year of Controlled Environment exposure is equal to 25 years in the field"
#
# Equation:
# $$ AF = N * \frac{ I_{chamber}^x * RH_{chamber}^n * e^{\frac{- E_a}{k T_{chamber}}} }{ \Sigma (I_{POA}^x * RH_{outdoor}^n * e^{\frac{-E_a}{k T_outdoor}}) }$$
#
# Function to calculate IWa, the Environment Characterization (W/m²). For one year of degredation the controlled environmnet lamp settings will need to be set at IWa.
#
# Equation:
# $$ I_{WA} = [ \frac{ \Sigma (I_{outdoor}^x * RH_{outdoor}^n e^{\frac{-E_a}{k T_{outdood}}}) }{ N * RH_{WA}^n * e^{- \frac{E_a}{k T_eq}} } ]^{\frac{1}{x}} $$

# In[6]:


# relative humidity within chamber (%)
rh_chamber = 15
# arrhenius activation energy (kj/mol)
Ea = 40

rh_surface = pvdeg.humidity.surface_relative(
    rh_ambient=WEATHER["relative_humidity"],
    temp_ambient=WEATHER["temp_air"],
    temp_module=temp_module,
)

arrhenius_deg = pvdeg.degradation.arrhenius_deg(
    weather_df=WEATHER,
    meta=META,
    rh_outdoor=rh_surface,
    I_chamber=I_chamber,
    rh_chamber=rh_chamber,
    temp_chamber=temp_chamber,
    poa=poa_df,
    temp_cell=temp_cell,
    Ea=Ea,
)

irr_weighted_avg_a = pvdeg.degradation.IwaArrhenius(
    weather_df=WEATHER,
    meta=META,
    poa=poa_df,
    rh_outdoor=WEATHER["relative_humidity"],
    temp_cell=temp_cell,
    Ea=Ea,
)


# ## 5. Quick Method (Degradation)
#
# For quick calculations, you can omit POA and both module and cell temperature. The function will calculate these figures as needed using the available weather data with the default options for PV module configuration.

# In[7]:


# chamber settings
I_chamber = 1000
temp_chamber = 60
rh_chamber = 15

# activation energy
Ea = 40

vantHoff_deg = pvdeg.degradation.vantHoff_deg(
    weather_df=WEATHER, meta=META, I_chamber=I_chamber, temp_chamber=temp_chamber
)

irr_weighted_avg_v = pvdeg.degradation.IwaVantHoff(weather_df=WEATHER, meta=META)


# In[8]:


rh_surface = pvdeg.humidity.surface_relative(
    rh_ambient=WEATHER["relative_humidity"],
    temp_ambient=WEATHER["temp_air"],
    temp_module=temp_module,
)

arrhenius_deg = pvdeg.degradation.arrhenius_deg(
    weather_df=WEATHER,
    meta=META,
    rh_outdoor=rh_surface,
    I_chamber=I_chamber,
    rh_chamber=rh_chamber,
    temp_chamber=temp_chamber,
    Ea=Ea,
)

irr_weighted_avg_a = pvdeg.degradation.IwaArrhenius(
    weather_df=WEATHER, meta=META, rh_outdoor=WEATHER["relative_humidity"], Ea=Ea
)


# ## 6. Solder Fatigue
#
# Estimate the thermomechanical fatigue of flat plate photovoltaic module solder joints over the time range given using estimated cell temperature. Like other `pvdeg` funcitons, the minimal parameters are (weather_df, meta). Running the function with only these two inputs will use default PV module configurations ( open_rack_glass_polymer ) and the 'sapm' temperature model over the entire length of the weather data.

# In[9]:


fatigue = pvdeg.fatigue.solder_fatigue(weather_df=WEATHER, meta=META)


# If you wish to reduce the span of time or use a non-default temperature model, you may specify the parameters manually. Let's try an explicit example.
# We want the solder fatigue estimated over the month of June for a roof mounted glass-front polymer-back module.
#
# 1. Lets create a datetime-index for the month of June.
# 2. Next, generate the cell temperature. Make sure to explicity restrict the weather data to our dt-index for June. Next, declare the PV module configuration.
# 3. Calculate the fatigue. Explicity specify the time_range (our dt-index for June from step 1) and the cell temperature as we caculated in step 2

# In[ ]:


# select the month of June
time_range = WEATHER.index[WEATHER.index.month == 6]

# calculate cell temperature over our selected date-time range.
# specify the module configuration
temp_cell = pvdeg.temperature.cell(
    weather_df=WEATHER.loc[time_range],
    meta=META,
    temp_model="sapm",
    conf="insulated_back_glass_polymer",
)


fatigue = pvdeg.fatigue.solder_fatigue(
    weather_df=WEATHER, meta=META, time_range=time_range, temp_cell=temp_cell
)
