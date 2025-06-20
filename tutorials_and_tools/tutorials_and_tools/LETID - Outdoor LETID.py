#!/usr/bin/env python
# coding: utf-8

# # LETID - Outdoor Environments
#
# This is an example on how to model LETID progression in outdoor environments
#
# We can use the equations in this library to model LETID progression in a simulated outdoor environment, given that we have weather and system data. This example makes use of tools from the fabulous [pvlib](https://pvlib-python.readthedocs.io/en/stable/) library to calculate system irradiance and temperature, which we use to calculate progression in LETID states.
#
# This will illustrate the potential of "Temporary Recovery", i.e., the backwards transition of the LETID defect B->A that can take place with carrier injection at lower temperatures.
#
#
# **Requirements:**
# - `pvlib`, `pandas`, `numpy`, `matplotlib`
#
# **Objectives:**
# 1. Use `pvlib` and provided weather files to set up a temperature and injection timeseries
# 2. Define necessary solar cell device parameters
# 3. Define necessary degradation parameters: degraded lifetime and defect states
# 4. Run through timeseries, calculating defect states
# 5. Calculate device degradation and plot
#

# In[1]:


# if running on google colab, uncomment the next line and execute this cell to install the dependencies and prevent "ModuleNotFoundError" in later cells:
# !pip install pvdeg==0.3.3


# In[1]:


from pvdeg import letid, collection, utilities, DATA_DIR

import pvlib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# This information helps with debugging and getting support :)
import sys
import platform

print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)
print("pvlib version ", pvlib.__version__)
print("pvdeg version ", pvdeg.__version__)


# First, we'll use pvlib to create and run a model system, and use the irradiance, temperature, and operating point of that model to set up our LETID model
# For this example, we'll model a fixed latitude tilt system at NREL, in Golden, CO, USA, using [NSRDB](https://nsrdb.nrel.gov/) hourly PSM weather data, SAPM temperature models, and module and inverter models from the CEC database.

# In[2]:


# load weather and location data, use pvlib read_psm3 function with map_variables = True

sam_file = "psm3.csv"
weather, meta = pvlib.iotools.read_psm3(
    os.path.join(DATA_DIR, sam_file), map_variables=True
)


# In[3]:


weather


# In[4]:


# if our weather file doesn't have precipitable water, calculate it with pvlib
if "precipitable_water" not in weather.columns:
    weather["precipitable_water"] = pvlib.atmosphere.gueymard94_pw(
        weather["temp_air"], weather["relative_humidity"]
    )


# In[5]:


# rename some columns for pvlib if they haven't been already
weather.rename(
    columns={
        "GHI": "ghi",
        "DNI": "dni",
        "DHI": "dhi",
        "Temperature": "temp_air",
        "Wind Speed": "wind_speed",
        "Relative Humidity": "relative_humidity",
        "Precipitable Water": "precipitable_water",
    },
    inplace=True,
)
weather = weather[
    [
        "ghi",
        "dni",
        "dhi",
        "temp_air",
        "wind_speed",
        "relative_humidity",
        "precipitable_water",
    ]
]


# In[6]:


weather


# In[7]:


# import pvlib stuff and pick a module and inverter. Choice of these things will slightly affect the pvlib results which we later use to calculate injection.
# we'll use the SAPM temperature model open-rack glass/polymer coeffecients.

from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain

cec_modules = pvlib.pvsystem.retrieve_sam("CECMod")
cec_inverters = pvlib.pvsystem.retrieve_sam("cecinverter")

cec_module = cec_modules["Jinko_Solar_Co___Ltd_JKM260P_60"]
cec_inverter = cec_inverters["ABB__ULTRA_750_TL_OUTD_1_US_690_x_y_z__690V_"]

temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS["sapm"][
    "open_rack_glass_polymer"
]


# In[8]:


# set up system in pvlib
lat = meta["latitude"]
lon = meta["longitude"]
tz = meta["Local Time Zone"]
elevation = meta["altitude"]
surface_tilt = lat  # fixed, latitude tilt
surface_azimuth = 180  # south-facing

location = Location(lat, lon, tz, elevation, "Golden, CO, USA")

system = PVSystem(
    surface_tilt=surface_tilt,
    surface_azimuth=surface_azimuth,
    module_parameters=cec_module,
    inverter_parameters=cec_inverter,
    temperature_model_parameters=temperature_model_parameters,
)


# In[9]:


# create and run pvlib modelchain
mc = ModelChain(system, location, aoi_model="physical")
mc.run_model(weather)


# ## Set up timeseries
# In this example, injection is a function of both the operating point of the module (which we will assume is maximum power point) and irradiance. Maximum power point injection is equivalent to $(I_{sc}-I_{mp})/I_{sc}\times Ee$, where $Ee$ is effective irradiance, the irradiance absorbed by the module's cells. We normalize it to 1-sun irradiance, 1000 $W/m^2$.
#
# We will use the irradiance, DC operating point, and cell temperature from the pvlib modelchain results.

# In[10]:


ee = mc.results.effective_irradiance
# injection = (mc.results.dc['i_sc']-mc.results.dc['i_mp'])/(mc.results.dc['i_sc'])*(ee/1000)
injection = letid.calc_injection_outdoors(mc.results)
temperature = mc.results.cell_temperature

timesteps = pd.DataFrame(
    {"Temperature": temperature, "Injection": injection}
)  # create a DataFrame with cell temperature and injection
timesteps.reset_index(
    inplace=True
)  # reset the index so datetime is a column. I prefer integer indexing.
timesteps.rename(columns={"index": "Datetime"}, inplace=True)


# In[11]:


# filter out times when injection is NaN, these won't progress LETID, and it'll make the calculations below run faster
timesteps = timesteps[timesteps["Injection"].notnull()]
timesteps.reset_index(inplace=True, drop=True)


# In[12]:


timesteps


# ## Device parameters
# To define a device, we need to define several important quantities about the device: wafer thickness (in $\mu m$), rear surface recombination velocity (in cm/s), and cell area (in cm<sup>2</sup>).

# In[1]:


wafer_thickness = 180  # um
s_rear = 46  # cm/s
cell_area = 243  # cm^2


#  <b> Other device parameters </b>
# Other required device parameters: base diffusivity (in cm<sup>2</sup>/s), and optical generation profile, which allow us to estimate current collection in the device.

# In[14]:


generation_df = pd.read_excel(
    os.path.join(DATA_DIR, "PVL_GenProfile.xlsx"), header=0
)  # this is an optical generation profile generated by PVLighthouse's OPAL2 default model for 1-sun, normal incident AM1.5 sunlight on a 180-um thick SiNx-coated, pyramid-textured wafer.
generation = generation_df["Generation (cm-3s-1)"]
depth = generation_df["Depth (um)"]

d_base = 27  # cm^2/s electron diffusivity. See https://www2.pvlighthouse.com.au/calculators/mobility%20calculator/mobility%20calculator.aspx for details


# ## Degradation parameters
# To model the device's degradation, we need to define several more important quantities about the degradation the device will experience. These include undegraded and degraded lifetime (in $\mu s$).

# In[15]:


tau_0 = 115  # us, carrier lifetime in non-degraded states, e.g. LETID/LID states A or C
tau_deg = 55  # us, carrier lifetime in fully-degraded state, e.g. LETID/LID state B


# <b>Remaining degradation parameters: </b>
#
# The rest of the quantities to define are: the initial percentage of defects in each state (A, B, and C), and the dictionary of mechanism parameters.
#
# In this example, we'll assume the device starts in the fully-undegraded state (100% state A), and we'll use the kinetic parameters for LETID degradation from Repins.

# In[16]:


# starting defect state percentages
nA_0 = 100
nB_0 = 0
nC_0 = 0

mechanism_params = utilities.get_kinetics("repins")

timesteps[["NA", "NB", "NC", "tau"]] = (
    np.nan
)  # create columns for defect state percentages and lifetime, fill with NaNs for now, to fill iteratively below

timesteps.loc[0, ["NA", "NB", "NC"]] = (
    nA_0,
    nB_0,
    nC_0,
)  # assign first timestep defect state percentages
timesteps.loc[0, "tau"] = letid.tau_now(
    tau_0, tau_deg, nB_0
)  # calculate tau for the first timestep


# ## Run through timesteps
# Since each timestep depends on the preceding timestep, we need to calculate in a loop. This will take a few minutes depending on the length of the timeseries.

# In[17]:


for index, timestep in timesteps.iterrows():
    # first row tau has already been assigned
    if index == 0:
        # calc device parameters for first row
        tau = tau_0
        jsc = collection.calculate_jsc_from_tau_cp(
            tau, wafer_thickness, d_base, s_rear, generation, depth
        )
        voc = letid.calc_voc_from_tau(tau, wafer_thickness, s_rear, jsc, temperature=25)
        timesteps.at[index, "Jsc"] = jsc
        timesteps.at[index, "Voc"] = voc

    # loop through rows, new tau calculated based on previous NB. Reaction proceeds based on new tau.
    else:
        n_A = timesteps.at[index - 1, "NA"]
        n_B = timesteps.at[index - 1, "NB"]
        n_C = timesteps.at[index - 1, "NC"]

        tau = letid.tau_now(tau_0, tau_deg, n_B)
        jsc = collection.calculate_jsc_from_tau_cp(
            tau, wafer_thickness, d_base, s_rear, generation, depth
        )

        temperature = timesteps.at[index, "Temperature"]
        injection = timesteps.at[index, "Injection"]

        # calculate defect reaction kinetics: reaction constant and carrier concentration factor.
        k_AB = letid.k_ij(
            mechanism_params["v_ab"], mechanism_params["ea_ab"], temperature
        )
        k_BA = letid.k_ij(
            mechanism_params["v_ba"], mechanism_params["ea_ba"], temperature
        )
        k_BC = letid.k_ij(
            mechanism_params["v_bc"], mechanism_params["ea_bc"], temperature
        )
        k_CB = letid.k_ij(
            mechanism_params["v_cb"], mechanism_params["ea_cb"], temperature
        )

        x_ab = letid.carrier_factor(
            tau,
            "ab",
            temperature,
            injection,
            jsc,
            wafer_thickness,
            s_rear,
            mechanism_params,
        )
        x_ba = letid.carrier_factor(
            tau,
            "ba",
            temperature,
            injection,
            jsc,
            wafer_thickness,
            s_rear,
            mechanism_params,
        )
        x_bc = letid.carrier_factor(
            tau,
            "bc",
            temperature,
            injection,
            jsc,
            wafer_thickness,
            s_rear,
            mechanism_params,
        )

        # calculate the instantaneous change in NA, NB, and NC
        dN_Adt = (k_BA * n_B * x_ba) - (k_AB * n_A * x_ab)
        dN_Bdt = (
            (k_AB * n_A * x_ab) + (k_CB * n_C) - ((k_BA * x_ba + k_BC * x_bc) * n_B)
        )
        dN_Cdt = (k_BC * n_B * x_bc) - (k_CB * n_C)

        t_step = (
            timesteps.at[index, "Datetime"] - timesteps.at[index - 1, "Datetime"]
        ).total_seconds()

        # assign new defect state percentages
        timesteps.at[index, "NA"] = n_A + dN_Adt * t_step
        timesteps.at[index, "NB"] = n_B + dN_Bdt * t_step
        timesteps.at[index, "NC"] = n_C + dN_Cdt * t_step

        # calculate device parameters
        timesteps.at[index, "tau"] = tau
        timesteps.at[index, "Jsc"] = jsc
        timesteps.at[index, "Voc"] = letid.calc_voc_from_tau(
            tau, wafer_thickness, s_rear, jsc, temperature=25
        )


# ## Finish calculating degraded device parameters.
# Now that we have calculated defect states, we can calculate all the quantities that depend on defect states.

# In[18]:


timesteps["tau"] = letid.tau_now(tau_0, tau_deg, timesteps["NB"])

# calculate device Jsc for every timestep. Unfortunately this requires an integration so I think we have to run through a loop. Device Jsc allows calculation of device Voc.
for index, timestep in timesteps.iterrows():
    jsc_now = collection.calculate_jsc_from_tau_cp(
        timesteps.at[index, "tau"], wafer_thickness, d_base, s_rear, generation, depth
    )
    timesteps.at[index, "Jsc"] = jsc_now
    timesteps.at[index, "Voc"] = letid.calc_voc_from_tau(
        timesteps.at[index, "tau"], wafer_thickness, s_rear, jsc_now, temperature=25
    )


# In[19]:


timesteps = letid.calc_device_params(
    timesteps, cell_area=243
)  # this function quickly calculates the rest of the device parameters: Isc, FF, max power, and normalized max power

timesteps


# Note of course that all these calculated device parameters are modeled STC device parameters, not the instantaneous, weather-dependent values. This isn't a robust performance model of a degraded module.

# ## Plot the results

# In[20]:


from cycler import cycler

plt.style.use("default")

fig, ax = plt.subplots()

ax.set_prop_cycle(
    cycler("color", ["tab:blue", "tab:orange", "tab:green"])
    + cycler("linestyle", ["-", "--", "-."])
)

ax.plot(timesteps["Datetime"], timesteps[["NA", "NB", "NC"]].values)
ax.legend(labels=["$N_A$", "$N_B$", "$N_C$"], loc="upper left")
ax.set_ylabel("Defect state percentages [%]")
ax.set_xlabel("Datetime")

ax2 = ax.twinx()
ax2.plot(
    timesteps["Datetime"],
    timesteps["Pmp_norm"],
    c="black",
    label="Normalized STC $P_{MP}$",
)
ax2.legend(loc="upper right")
ax2.set_ylabel("Normalized STC $P_{MP}$")

ax.set_title(f"Outdoor LETID \n{location.name}")

plt.show()


# The example data provided for Golden, CO, shows how $N_A$ increases in cold weather, and power temporarily recovers, due to temporary recovery of LETID (B->A).

# ##### The function `calc_letid_outdoors` wraps all of the steps above into a single function:

# In[21]:


nA_0 = 100
nB_0 = 0
nC_0 = 0
mechanism_params = "repins"

letid.calc_letid_outdoors(
    tau_0,
    tau_deg,
    wafer_thickness,
    s_rear,
    nA_0,
    nB_0,
    nC_0,
    weather,
    meta,
    mechanism_params,
    generation_df,
    module_parameters=cec_module,
)


# In[ ]:
