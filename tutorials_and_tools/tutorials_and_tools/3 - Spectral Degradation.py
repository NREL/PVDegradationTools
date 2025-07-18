#!/usr/bin/env python
# coding: utf-8

# # 3 - Spectral Degradation
#
# **Requirements:**
# - spectral irradiance (measured or simulated)
# - wavelengths of spectral irradiance data
# - module RH
# - module temperature
#
#
# **Objectives:**
# 1. Read in spectral irradiance
# 2. Calculate spectral degradation

# In[1]:


# if running on google colab, uncomment the next line and execute this cell to install the dependencies and prevent "ModuleNotFoundError" in later cells:
# !pip install pvdeg==0.3.3


# In[1]:


import os
import pandas as pd
import numpy as np
import pvdeg
from pvdeg import DATA_DIR


# In[ ]:


# This information helps with debugging and getting support :)
import sys
import platform

print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)
print("pvdeg version ", pvdeg.__version__)


# ## 1. Read in spectral irradiance data
#
# Spectral degradation has 4 main requirements:
# - Spectral Irradiance [W/m^2 nm]
# - Wavelength [nm]
# - Module Relative Humidity [%]
# - Module Temperature [C]
#
# For more advanced scenarios, you may want to calculate the degradation of a particular layer within the module. Below, we are using *backside* irradiance and therefore a slightly different temperature and humidity have been calculated. To calculate degradation on the backside, we used `pvdeg.humidity.rh_backsheet`. For the the front side, you should use `pvdeg.humidity.rh_surface_outside` or `rh_front_encap`
#
#
# For this tutorial we are using pre-generated data from a ray-tracing simulation. To calculate the degradation rate, we will need the wavelengths used in the simulation.

# In[2]:


wavelengths = np.array(range(280, 420, 20))

SPECTRA = pd.read_csv(os.path.join(DATA_DIR, "spectra.csv"), header=0, index_col=0)
SPECTRA.head()


# ### 2. Calculate Degradation
#
# The spectral degradation function has several optional paramters. For more information, refer to the documentation. Below is a function call with the minimum required information.

# In[3]:


degradation = pvdeg.degradation.degradation(
    spectra=SPECTRA["Spectra"],
    rh_module=SPECTRA["RH"],
    temp_module=SPECTRA["Temperature"],
    wavelengths=wavelengths,
)
