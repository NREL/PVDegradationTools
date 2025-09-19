#!/usr/bin/env python
# coding: utf-8

# # LETID - Outdoor Geospatioal Demo
#
# ![PVDeg Logo](../PVD_logo.png)
#

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pvdeg
from pvdeg import DATA_DIR
import os


# In[2]:


# This information helps with debugging and getting support :)
import sys
import platform

print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)
print("pvdeg version ", pvdeg.__version__)


# ## Single location example

# In[2]:


weather_file = os.path.join(DATA_DIR, "psm3_demo.csv")
WEATHER, META = pvdeg.weather.read(weather_file, "psm")


# In[3]:


kwargs = {
    "tau_0": 115,  # us, carrier lifetime in non-degraded states, e.g. LETID/LID states A or C
    "tau_deg": 55,  # us, carrier lifetime in fully-degraded state, e.g. LETID/LID state B
    "wafer_thickness": 180,  # um
    "s_rear": 46,  # cm/s
    "cell_area": 243,  # cm^2
    "na_0": 100,
    "nb_0": 0,
    "nc_0": 0,
    "mechanism_params": 'D037',
}


# In[4]:


pvdeg.letid.calc_letid_outdoors(weather_df=WEATHER, meta=META, **kwargs)


# ### Start distributed compute cluster - DASK

# In[2]:


local = {
    "manager": "local",
    "n_workers": 1,
    "threads_per_worker": 8,  # Number of CPUs
}

kestrel = {
    "manager": "slurm",
    "n_jobs": 1,  # Number of nodes used for parallel processing
    "cores": 104,
    "memory": "256GB",
    "account": "pvsoiling",
    "queue": "debug",
    "walltime": "1:00:00",
    "processes": 104,
    "job_extra_directives": ["-o ./logs/slurm-%j.out"],
}

pvdeg.geospatial.start_dask(hpc=kestrel)


# In[11]:


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

# Define geographical region
meta_SW = meta_df[meta_df["state"].isin(["Colorado", "New Mexico", "Utah", "Arizona"])]
meta_SW_sub, gids_SW_sub = pvdeg.utilities.gid_downsampling(meta_SW, 6)

weather_SW_sub = weather_ds.sel(gid=meta_SW_sub.index)


# In[12]:


weather_SW_sub


# In[13]:


# Define desired analysis
geo = {
    "func": pvdeg.letid.calc_letid_outdoors,
    "weather_ds": weather_SW_sub,
    "meta_df": meta_SW_sub,
    "tau_0": 115,  # us, carrier lifetime in non-degraded states, e.g. LETID/LID states A or C
    "tau_deg": 55,  # us, carrier lifetime in fully-degraded state, e.g. LETID/LID state B
    "wafer_thickness": 180,  # um
    "s_rear": 46,  # cm/s
    "cell_area": 243,  # cm^2
    "na_0": 100,
    "nb_0": 0,
    "nc_0": 0,
    "mechanism_params": 'D037',
}

letid_res = pvdeg.geospatial.analysis(**geo)


# In[14]:


letid_res


# In[18]:


import datetime

ims = []
for n in range(1, 13):
    for i, np_t in enumerate(letid_res.time):
        t = pd.Timestamp(np_t.values).time()
        d = pd.Timestamp(np_t.values).day
        m = pd.Timestamp(np_t.values).month
        if m == n:
            if d == 15:
                if t == datetime.time(12):
                    fig, ax = pvdeg.geospatial.plot_USA(
                        letid_res["Pmp_norm"].sel(time=np_t),
                        cmap="viridis",
                        vmin=0.95,
                        vmax=1,
                        title=f"Normalized Power  - 2022-{m}-{d} 12:00",
                        cb_title="Normalized Power",
                    )
                    # plt.savefig(f'./images/RH_animation_{n}.png', dpi=600)

# import imageio
# ims = [imageio.imread(f'./images/RH_animation_{n}.png') for n in range(1, 13)]
# imageio.mimwrite(f'./images/RH_animation.gif', ims, format='GIF', duration=1000, loop=10)


# In[34]:


import datetime

ims = []
dates = []
subarctics = []
coldsemiarids = []
hotdeserts = []

for n in range(1, 13):
    for i, np_t in enumerate(letid_res.time):
        t = pd.Timestamp(np_t.values).time()
        d = pd.Timestamp(np_t.values).day
        m = pd.Timestamp(np_t.values).month
        if m == n:
            if d == 15:
                if t == datetime.time(12):
                    dates.append(np_t.values)

                    # subartic: near Crested Butte CO
                    # cold semi-arid: near Springfield CO
                    # hot desert: near Yuma AZ

                    subarctic = letid_res.sel(
                        time=np_t, latitude=39.01, longitude=-107.1
                    )
                    subarctics.append(subarctic["Pmp_norm"])

                    coldsemiarid = letid_res.sel(
                        time=np_t, latitude=37.57, longitude=-102.3
                    )
                    coldsemiarids.append(coldsemiarid["Pmp_norm"])

                    hotdesert = letid_res.sel(
                        time=np_t, latitude=32.77, longitude=-114.3
                    )
                    hotdeserts.append(hotdesert["Pmp_norm"])

                    fig, ax = plt.subplots()
                    ax.plot(
                        dates,
                        subarctics,
                        marker="o",
                        c="C0",
                        label="Central CO - Subarctic Dfc",
                    )
                    ax.plot(
                        dates,
                        coldsemiarids,
                        marker="o",
                        c="C1",
                        label="Southeast CO - Cold Semi-Arid BSk",
                    )
                    ax.plot(
                        dates,
                        hotdeserts,
                        marker="o",
                        c="C2",
                        label="Southwest AZ - Hot Desert BWh",
                    )

                    ax.legend(loc="upper right")

                    ax.set_xlim([datetime.date(2022, 1, 1), datetime.date(2023, 1, 1)])

                    ax.set_ylim([0.945, 1.005])
                    ax.set_ylabel("Normalized Power")

                    plt.savefig(f"./images/LETID_plot_animation_{n}.png", dpi=600)


# In[ ]:


import imageio

ims = [imageio.imread("./images/LETID_plot_animation_{n}.png") for n in range(1, 13)]
imageio.mimwrite(
    "./images/LETID_plot_animation.gif", ims, format="GIF", duration=1000, loop=10
)
