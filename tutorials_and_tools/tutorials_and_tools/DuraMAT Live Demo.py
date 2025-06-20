#!/usr/bin/env python
# coding: utf-8

# # DuraMAT Workshop Live Demo - Geospatial analysis
#
# ![PVDeg Logo](../PVD_logo.png)
#
#
# **Steps:**
# 1. Initialize weather data into xarray
# 2. Calculate installation standoff for New Mexico
# 3. Plot results
#
# **Xarray: multi-dimensional data frame**
#
# ![Xarray](./images/xarray.webp)

# In[2]:


import pandas as pd
import pvdeg


# In[ ]:


# This information helps with debugging and getting support :)
import sys
import platform

print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)
print("pvdeg version ", pvdeg.__version__)


# ## 1 Start distributed compute cluster - DASK

# In[3]:


pvdeg.geospatial.start_dask()


# In[4]:


# Get weather data
weather_db = "NSRDB"

weather_arg = {
    "satellite": "Americas",
    "names": 2022,
    "NREL_HPC": True,
    "attributes": [
        "air_temperature",
        "wind_speed",
        "dhi",
        "ghi",
        "dni",
        "relative_humidity",
    ],
}

weather_ds, meta_df = pvdeg.weather.get(weather_db, geospatial=True, **weather_arg)


# In[5]:


weather_ds


# In[6]:


meta_df["state"].unique()


# In[7]:


meta_NM = meta_df[meta_df["state"] == "New Mexico"]


# In[8]:


meta_NM_sub, gids_NM_sub = pvdeg.utilities.gid_downsampling(meta_NM, 4)
weather_NM_sub = weather_ds.sel(gid=meta_NM_sub.index)


# In[9]:


geo = {
    "func": pvdeg.standards.standoff,
    "weather_ds": weather_NM_sub,
    "meta_df": meta_NM_sub,
}

standoff_res = pvdeg.geospatial.analysis(**geo)


# In[10]:


standoff_res


# In[11]:


fig, ax = pvdeg.geospatial.plot_USA(
    standoff_res["x"],
    cmap="viridis",
    vmin=0,
    vmax=None,
    title="Minimum estimated air standoff to qualify as level 1 system",
    cb_title="Standoff (cm)",
)


# # Relative Humidity Example - Time dimension

# In[12]:


# State bar of new mexico: (35.16482, -106.58979)

weather_db = "NSRDB"
weather_id = (35.16482, -106.58979)  # NREL (39.741931, -105.169891)
weather_arg = {
    "satellite": "Americas",
    "names": 2022,
    "NREL_HPC": True,
    "attributes": [
        "air_temperature",
        "wind_speed",
        "dhi",
        "ghi",
        "dni",
        "relative_humidity",
    ],
}

weather_df, meta = pvdeg.weather.get(
    weather_db, weather_id, geospatial=False, **weather_arg
)


# In[13]:


RH_module = pvdeg.humidity.module(weather_df=weather_df, meta=meta)


# In[14]:


RH_module


# In[15]:


RH_module.plot(ls="--")


# In[16]:


geo = {
    "func": pvdeg.humidity.module,
    "weather_ds": weather_NM_sub,
    "meta_df": meta_NM_sub,
}

RH_module = pvdeg.geospatial.analysis(**geo)


# In[17]:


RH_module


# In[18]:


# from matplotlib.animation import FuncAnimation
# from matplotlib.animation import PillowWriter
# import matplotlib.animation as animation
# import datetime
# ims = []
# for n in range(1, 13):
#     for i, np_t in enumerate(RH_module.time):
#         t = pd.Timestamp(np_t.values).time()
#         d = pd.Timestamp(np_t.values).day
#         m = pd.Timestamp(np_t.values).month
#         if m == n:
#             if d == 15:
#                 if t == datetime.time(12):
#                     fig, ax = pvdeg.geospatial.plot_USA(RH_module['RH_surface_outside'].sel(time=np_t),
#                             cmap='viridis', vmin=0, vmax=100,
#                             title=f'RH_surface_outside  - 2022-{m}-{d} 12:00',
#                             cb_title='Relative humidity (%)')
#                     plt.savefig(f'./images/RH_animation_{n}.png', dpi=600)

# import imageio
# ims = [imageio.imread(f'./images/RH_animation_{n}.png') for n in range(1, 13)]
# imageio.mimwrite(f'./images/RH_animation.gif', ims, format='GIF', duration=1000, loop=10)


# ![PVDeg Logo](./images/RH_animation.gif)

# In[ ]:
