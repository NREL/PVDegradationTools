#!/usr/bin/env python
# coding: utf-8

# # Tools - Module Standoff for IEC TS 63126
# ## Calculation of module standoff distance according to IEC TS 63126
#
#
# **Requirements:**
# - Local weather data file or site longitude and latittude
#
# **Objectives:**
# 1. Import weather data.
# 2. Calculate installation standoff - Level 1 and Level 2.
# 3. Calculate $X_{eff}$ from provided module temperature data.
# 4. Calculate $T_{98}$ for a given azimuth, tilt, and $X_{eff}$.
# 5. Plot $X_{min}$ for all azimuth and tilt for a given $T_{98}$.
# 6. Plot $X_{min}$ for Level 1, Level 2, or a $T_{98}$ for a given region.
#
# **Background:**
#
# This notebook calculates the a minimum effective standoff distance ($X_{eff}$) necessary for roof-mounted PV modules to ensure that the $98^{th}$ percentile operating temperature, $T_{98}$, remains under 70°C for compliance to IEC 61730 and IEC 61215. For higher $T_{98}$ values above 70°C or 80°C testing must be done to the specifications for Level 1 and Level 2 of IEC TS 63126. This method is outlined in the appendix of IEC TS 63126 and is based on the model from *[King 2004] and data from **[Fuentes, 1987] to model the approximate exponential decay in temperature, $T(X)$, with increasing standoff distance, $X$, as,
#
# $$ X = -X_0 \ln\left(1-\frac{T_0-T}{\Delta T}\right), Equation 1 $$
#
# where $T_0$ is the temperature for $X=0$ (insulated-back) and $\Delta T$ is the temperature difference between an insulated-back ($X=0$) and open-rack mounting configuration ($X=\infty)$.
#
#  We used pvlib and data from the National Solar Radiation Database (NSRDB) to calculate the module temperatures for the insulated-back and open-rack mounting configurations and apply our model to obtain the minimum standoff distance for roof-mounted PV systems to achieve a temperature lower than a specified $T_{98}$. The following figure showcases this calulation for the entire world for an $X_{eff}$ that results in $T_{98}$=70°C. Values of $X_{eff}$ higher than this will require Level 1 or Level 2 certification.
#
# $*$ D. L. King, W. E. Boyson, and J. A. Kratochvil, "Photovoltaic array performance model," SAND2004-3535, Sandia National Laboratories, Albuquerque, NM, 2004. '\
# $**$ M. K. Fuentes, "A simplified thermal model for Flat-Plate photovoltaic arrays," United States, 1987-05-01 1987. https://www.osti.gov/biblio/6802914
#

# ![T98 70C standoff Map.png](attachment:62279573-41e3-45dd-bf62-4fa60c1e7e69.png)

# In[ ]:


# if running on google colab, uncomment the next line and execute this cell to install the dependencies and prevent "ModuleNotFoundError" in later cells:
# !pip install pvdeg==0.3.3


# In[3]:


import os
import pvdeg
import pandas as pd
from pvdeg import DATA_DIR
import dask
import matplotlib.pyplot as plt
import seaborn as sns
import math


# In[2]:


# This information helps with debugging and getting support :)
import sys
import platform

print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)
print("pvdeg version ", pvdeg.__version__)
print("dask version", dask.__version__)
print(DATA_DIR)


# ## 1. Import Weather Data
#
# The function has these minimum requirements when using a weather data file:
# - Weather data containing (at least) DNI, DHI, GHI, Temperature, RH, and Wind-Speed data at module level.
# - Site meta-data containing (at least) latitude, longitude, and time zone
#
# Alternatively one may can get meterological data from the NSRDB with just the longitude and latitude.
#

# In[4]:


# Get data from a supplied data file (Do not use the next box of code if using your own file)
weather_file = os.path.join(DATA_DIR, "psm3_demo.csv")
WEATHER, META = pvdeg.weather.read(weather_file, "psm")


# In[5]:


# From Tutorial 5 EXAMPLE, this works.
# API_KEY = 'your_api_key_here'
API_KEY = "DEMO_KEY"  # you can activate this line to use the demonstration API key but it has limited usage.
# The example API key here is for demonstation and is rate-limited per IP.
# To get your own API key, visit https://developer.nrel.gov/signup/

weather_db = "PSM3"
weather_id = (33.4152, -111.8315)
weather_arg = {
    "api_key": API_KEY,
    "email": "user@mail.com",
    "names": "tmy",
    "attributes": [],
    "map_variables": True,
}

WEATHER_df, META = pvdeg.weather.get(weather_db, weather_id, **weather_arg)
print(META)


# ## 2. Calculate Installation Standoff Minimum - Level 1 and Level 2
#
# According to IEC TS 63126, Level 0, Level 1 and Level 2 certification is limited to T₉₈<70°C, <80°C and <90°C, respectively. Level 0 certification is essentially compliance to IEC 61730 and IEC 61215. The default value of T₉₈<70°C represents the minimium gap to avoid higher temperature certification according to IEC TS 63126. This minimum standoff ($x_{min}$) is the distance between the bottom of the module frame and the roof and can be extimated for a given environment as,
#
# $$ X_{min} = -X_0 \ln\left(1-\frac{T_{98,0}-T}{ T_{98,0}- T_{98,inf}}\right), Equation 2 $$
#
# where $T_{98,0}$ is the $98^{th}$ percentile temperature for an insulated back module and $T_{98,inf}$ is the $98^{th}$ percentile temperature for an open rack mounted module.
#
# Once the meterological data has been obtained, the input parameter possibilities are:
#
# - T₉₈ : Does not necessarily need to be set at 70°C or 80°C for IEC TS 63216, you might want to use a different number to compensate for a thermal aspect of the particular system you are considering. The default is 70°C.
# - tilt : tilt from horizontal of PV module. The default is 0°.
# - azimuth : azimuth in degrees from North. The default is 180° for south facing.
# - sky_model : pvlib compatible model for generating sky characteristics (Options: 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'). The default is 'isotropic'.
# - temp_model : pvlib compatible module temperature model. (Options: 'sapm', 'pvsyst', 'faiman', 'sandia'). The default is 'sapm'.
# - conf_0 : Temperature model for hotest mounting configuration. Default is "insulated_back_glass_polymer".
# - conf_inf : Temperature model for open rack mounting. Default is "open_rack_glass_polymer".
# - x_0 : thermal decay constant [cm] (see documentation). The default is 6.5 cm.
# - wind_factor : Wind speed power law correction factor to account for different wind speed measurement heights between weather database (e.g. NSRDB) and the tempeature model (e.g. SAPM). The default is 0.33.

# The following is the minimum function call. It defaults to horizontal tilt.

# In[6]:


standoff = pvdeg.standards.standoff(weather_df=WEATHER_df, meta=META)
output = pvdeg.standards.interpret_standoff(standoff)
print(output)


# The following is a full function call for both T₉₈=70°C and 80°C. This also includes the ability to print out a detailed interpretation of the results. With this function, one may also want to change the tilt, azimuth, or T_98

# In[8]:


standoff_1 = pvdeg.standards.standoff(
    weather_df=WEATHER_df,
    meta=META,
    T98=70,
    tilt=META["latitude"],
    azimuth=None,
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    x_0=6.5,
    wind_factor=0.33,
)
print("First calculation standoff = ", "%.1f" % standoff_1["x"].iloc[0], " cm.")
standoff_2 = pvdeg.standards.standoff(
    weather_df=WEATHER_df,
    meta=META,
    T98=80,
    tilt=META["latitude"],
    azimuth=None,
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    x_0=6.5,
    wind_factor=0.33,
)
print("Second calculation standoff = ", "%.1f" % standoff_2["x"].iloc[0], " cm.")
print("\n")
print(pvdeg.standards.interpret_standoff(standoff_1, standoff_2))


# ## 3. Calculate $X_{eff}$ from provided module temperature data.
#
# To do this calculation, one must use a set of data with:
#    - meterological irradiance data sufficient to calculate the POA irradiance (DHI, GHI, and DNI),
#    - ambient temperature data,
#    - wind speed at module height, (wind_factor=0.33 will be used unless otherwise specified)
#    - temperature measurements of the module in the test system. Ideally this would be measured under a worst case scenario that maximizes the module temperature for a given site,
#    - geographic meta data including longitude and latitude,
#
# To create a weather file of your own, copy the format of the example file 'xeff_demo.csv'. This is formatted with the first row containing meta data variable names, the second row containing the corresponding values, the third row containing meteorological data headers, and all the remaining rows containing the meteorological data.
#
# To do this calculation, one should also filter the data to remove times when the sun is not shining or when snow is likely to be on the module. The recommendations and programmed defaults are to use poa_min=100 W/m² and data when the minimum ambient temperature t_amb_min=0.

# In[9]:


# Read the weather file
weather_file = os.path.join(DATA_DIR, "psm3_demo.csv")
Xeff_WEATHER, Xeff_META = pvdeg.weather.read(weather_file, "psm3")

# Get module data from a supplied data file
measured = pd.read_csv(os.path.join(DATA_DIR, "module_temperature.csv"))

# Pull measured temperature and calculate theoretical insulated back module temperature and open rack module temperature
T_0, T_inf, T_measured, temp_air, POA = pvdeg.standards.eff_gap_parameters(
    weather_df=Xeff_WEATHER,
    meta=Xeff_META,
    module_temp=measured["Module_Temperature"],
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    tilt=39.73,
    azimuth=180,
    wind_factor=0.33,
)

# Now calculate X_eff.
x_eff = pvdeg.standards.eff_gap(
    T_0, T_inf, T_measured, temp_air, POA, x_0=6.5, poa_min=100, t_amb_min=0
)
print("The effective standoff for this system is", "%.1f" % x_eff, "cm.")


# ## 4. Calculate $T_{98}$ for a given azimuth, tilt, and $X_{eff}$.
#
# Equation 2 can be reorganized as,
#
# $$ T_{98} = T_{98,0}  -( T_{98,0}- T_{98,inf}) \left(1-e^{-\frac{x_{eff}}{x_{0}}}\right), Equation 3 $$
#
# and used to calculate the $98^{th}$ percential temperature, $T_{98}$, for a PV system having a given effective standoff height, $X_{eff}$,  for an arbitrarily oriented module. Here, $T_{98,0}$ is the $98^{th}$ percentile for an insulated-back module and $T_{98,inf}$ is the $98^{th}$ percentile for a rack-mounted module. The input parameter possibilities are the same as shown in Objective #2 above, but the example below uses the default parameters. The actual tilt [degrees], azimuth [degrees] and $X_{eff}$ [cm] can be modifed as desired.

# In[10]:


# This is the minimal function call using the common default settings to estimate T₉₈.
T_98 = pvdeg.standards.T98_estimate(
    weather_df=WEATHER_df,
    meta=META,
    tilt=META["latitude"],
    azimuth=None,
    x_eff=0,
)
print("The 98ᵗʰ percential temperature is estimated to be", "%.1f" % T_98, "°C.")


# ## 5. Plot $X_{min}$ for all azimuth and tilt for a given $T_{98}$.
#
# The temperature of a system is affected by the orientation. This section will scan all possible tilts and azimuths calculating the minimum standoff distance for a given $T_{98}$. Similar additional factors as above can also be modified but are not included here for simplicity. The tilt_step and azimuth_step are the number of degrees for each step for the 90° and 180° tilt and azimuth spans, respectively. The default for this calculation is for $T_{98}$=70°C, the boundary between Level 0 and Level 1 requirements. The temperature model information given below is unnecessary as these are default values that would get populated automatically. However, they were included here for clarity into a standard practice as per IEC TS 63126.
#
# $$ X_{min} = -X_0 \ln\left(1-\frac{T_{98,0}-T}{ T_{98,0}- T_{98,inf}}\right), Equation 2 $$

# In[11]:


# Scans through all the azimuth and tilt running the minimum standoff calculation
# Set up keyword parameters for the calculation

kwarg_x = dict(
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    T98=70,
    x_0=6.5,
    wind_factor=0.33,
)
# Run the calculation
x_azimuth_step = 10
x_tilt_step = 10
standoff_series = pvdeg.utilities.tilt_azimuth_scan(
    weather_df=WEATHER_df,
    meta=META,
    tilt_step=x_tilt_step,
    azimuth_step=x_azimuth_step,
    func=pvdeg.standards.standoff_x,
    **kwarg_x,
)


# The next cell creates a plot of the calculated data. Some of the things you may want to change are:
# - cmap="Spectral_r": Change to have different colors
# - plt.title : This will change the plot title.
# - figsize=(16,4) : Change the plot dimensions and/or aspect ratio.
# - vmax=None : This can be set to a numeric value to control the depth scale maximum
# - vmin=0 : This controls the minimum of the depth scale.
# - v_ticks=37 : This changes the number of vertical tick marks
# - h_ticks=10 : This changes the number of horizontal tick marks
# - Unblock the last two lines to ouput the plot as an *.png image file

# In[12]:


standoff_series_df = pd.DataFrame(
    {
        "Tilt": standoff_series[:, 0],
        "Azimuth": standoff_series[:, 1],
        "Xₘᵢₙ": standoff_series[:, 2],
    }
)
x_fig = plt.figure(figsize=(16, 4))
plt.title(
    r"Plot of $\it{Xₘᵢₙ}$ for all orientations for $\it{T₉₈}$="
    + "%.0f" % kwarg_x["T98"]
    + "°C.",
    fontsize=15,
    y=1.08,
)
x_fig = sns.heatmap(
    standoff_series_df.pivot(index="Tilt", columns="Azimuth", values="Xₘᵢₙ"),
    cbar_kws={"label": "Xₘᵢₙ", "format": "%.0f", "pad": 0.02},
    cmap="Spectral_r",
    vmin=0,
    vmax=None,
)

h_ticks = 37
x_number = math.ceil(360 / x_azimuth_step) + 1
x_ticks = [
    (x * (360 / (h_ticks - 1)) / x_azimuth_step + 0.5) for x in range(h_ticks - 1)
]
x_labels = [("%.0f" % (360 / (h_ticks - 1) * x)) for x in range(h_ticks)]
x_ticks.append(x_number - 0.5)
x_fig.set_xticks(x_ticks)
x_fig.set_xticklabels(x_labels, rotation=90)

v_ticks = 10
y_number = math.ceil(90 / x_tilt_step) + 1
y_ticks = [(x * (90 / (v_ticks - 1)) / x_tilt_step + 0.5) for x in range(v_ticks - 1)]
y_labels = [("%.0f" % (90 / (v_ticks - 1) * x)) for x in range(v_ticks)]
y_ticks.append(y_number - 0.5)
x_fig.set_yticks(y_ticks)
x_fig.set_yticklabels(y_labels, rotation=0)

x_fig.set_xlabel("Azimuth [°]", fontsize=15, labelpad=10)
x_fig.set_ylabel("Tilt [°]", fontsize=15)
x_fig.figure.axes[-1].set_ylabel(r"$\it{Xₘᵢₙ}$ [cm]", size=15)
x_fig.invert_yaxis()

output_folder = os.path.join(
    os.path.dirname(os.path.dirname(os.getcwd())), "TEMP", "results"
)
try:
    os.mkdir(output_folder)
except OSError as error:
    print(error)

plt.savefig(
    output_folder + r"\Standoff_Scan.png", bbox_inches="tight"
)  # Creates an image file of the standoff plot
plt.show()


# ## 6. Plot $T_{98}$  for all azimuth and tilt for a given $X_{eff}$.
#
# The temperature of a system is affected by the orientation and the effective standoff, $X_{eff}$, of the system. This section will scan all possible tilts and azimuths calculating the $T_{98}$ for a given $X_{eff}$. As above, additional factors can be modified but are not included here for simplicity. The tilt_step and azimuth_step are the number of degrees for each step for the 90° and 180° tilt and azimuth spans, respectively. The default for this calculation is for $X_{eff}$=10 cm, a common effective standoff distance on a rooftop system. A value of $X_{eff}$=None will run the calculations for an open rack system and $X_{eff}$=0 for an insulated-back system.

# In[13]:


# Scans through all the azimuth and tilt running the 98ᵗʰ percentile temperature calculation.
# Set up keyword parameters for the calculation
kwarg_T = dict(
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    x_eff=5,
    x_0=6.5,
    wind_factor=0.33,
)
# Run the calculation
T_azimuth_step = 10
T_tilt_step = 10
T98_series = pvdeg.utilities.tilt_azimuth_scan(
    weather_df=WEATHER_df,
    meta=META,
    tilt_step=T_tilt_step,
    azimuth_step=T_azimuth_step,
    func=pvdeg.standards.T98_estimate,
    **kwarg_T,
)


# The next cell creates a plot of the calculated data. Some of the things you may want to change are:
# - cmap="Spectral_r": Change to have different colors
# - plt.title : This will change the plot title.
# - figsize=(16,4) : Change the plot dimensions and/or aspect ratio.
# - vmax=None : This can be set to a numeric value to control the depth scale maximum
# - vmin=None : This controls the minimum of the depth scale.
# - v_ticks=37 : This changes the number of vertical tick marks
# - h_ticks=10 : This changes the number of horizontal tick marks
# - Unblock the last two lines to ouput the plot as an *.png image file

# In[14]:


# This produces the plot of the data
T98_series_df = pd.DataFrame(
    {"Tilt": T98_series[:, 0], "Azimuth": T98_series[:, 1], "T₉₈": T98_series[:, 2]}
)
T98_fig = plt.figure(figsize=(16, 4))
if kwarg_T["x_eff"] == None:
    plt.title(
        r"Plot of $\it{T₉₈}$ for all orientations for an open-rack mounting.",
        fontsize=15,
        y=1.08,
    )
else:
    plt.title(
        r"Plot of $\it{T₉₈}$ for all orientations for $X_{eff}$="
        + "%.0f" % kwarg_T["x_eff"]
        + " cm.",
        fontsize=15,
        y=1.08,
    )
T98_fig = sns.heatmap(
    T98_series_df.pivot(index="Tilt", columns="Azimuth", values="T₉₈"),
    cbar_kws={"label": "Xₘᵢₙ", "format": "%.0f", "pad": 0.02},
    cmap="Spectral_r",
    vmin=None,
    vmax=None,
)

h_ticks = 37
x_number = math.ceil(360 / T_azimuth_step) + 1
x_ticks = [
    (x * (360 / (h_ticks - 1)) / T_azimuth_step + 0.5) for x in range(h_ticks - 1)
]
x_labels = [("%.0f" % (360 / (h_ticks - 1) * x)) for x in range(h_ticks)]
x_ticks.append(x_number - 0.5)
T98_fig.set_xticks(x_ticks)
T98_fig.set_xticklabels(x_labels, rotation=90)

v_ticks = 10
y_number = math.ceil(90 / T_tilt_step) + 1
y_ticks = [(x * (90 / (v_ticks - 1)) / T_tilt_step + 0.5) for x in range(v_ticks - 1)]
y_labels = [("%.0f" % (90 / (v_ticks - 1) * x)) for x in range(v_ticks)]
y_ticks.append(y_number - 0.5)
T98_fig.set_yticks(y_ticks)
T98_fig.set_yticklabels(y_labels, rotation=0)

T98_fig.set_xlabel("Azimuth [°]", fontsize=15, labelpad=10)
T98_fig.set_ylabel("Tilt [°]", fontsize=15)
T98_fig.figure.axes[-1].set_ylabel(r"$\it{T₉₈}$ [°C]", size=15)
T98_fig.invert_yaxis()

plt.savefig(
    os.path.join(output_folder, "T98_Scan.png"), bbox_inches="tight"
)  # Creates an image file of the standoff plot
plt.show(T98_fig)


# ## 7. Plot $X_{min}$ for Level 1, Level 2, and $T_{98}$ for a given region.
#
# This last Objective is much more complicated and is set up to utilize acess to a lot of computational power to run many sites simultaneously to create a regional map of standoff distance. This is presented as doing the computations on Amazon Web Services (AWS) for which you will need a paid account, and will be covered on a subsequent journal.

# ## 8. Save data outputs.
#
# This cell contains a number of pre-scripted commands for exporting and saving data. The code to save plots is located after the plot creation and is blocked by default. First check that the output folder exists, then unblock the code for data you would like to save.

# In[15]:


print("Your results will be stored in %s" % output_folder)
print("The folder must already exist or the file will not be created")

pvdeg.weather.write(
    data_df=WEATHER_df,
    metadata=META,
    savefile=os.path.join(output_folder, "WeatherFile.csv"),
)  # Writes the meterological data to an *.csv file.

pd.DataFrame(standoff_series_df).to_csv(
    os.path.join(output_folder, "Standoff_Scan.csv")
)  # Writes a file with the Tilt and Azimuth scan calculations of standoff.

pd.DataFrame(T98_series_df).to_csv(os.path.join(output_folder, "T98_Scan.csv"))
