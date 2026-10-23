# %% [markdown]
# # Temperature Scenario
#

# %%
# if running on google colab, uncomment the next line and execute this cell to install the dependencies and prevent "ModuleNotFoundError" in later cells:
# # !pip install pvdeg

# %%
import pvdeg
import os

# %%
# This information helps with debugging and getting support :)
import sys
import platform

print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("pvdeg version ", pvdeg.__version__)

# %% [markdown]
# # Adding Modules and Pipeline Jobs (Run Functions on Scenario Object)
#
# Material: `OX003` corresponds to a set of EVA material parameters from the default file `O2Permeation.json` in the `pvdeg/data` directory. Look in these files to see available options.

# %%
scene_temp = pvdeg.Scenario(
    name="temperature and degradation",
    api_key="DEMO_KEY",
    email="user@mail.com",
)

scene_temp.addLocation(
    lat_long=(25.783388, -80.189029),
)

# this module will be overwritten because another with the same name is added afterwards
scene_temp.addModule(module_name="sapm_1", temperature_model="sapm")

scene_temp.addModule(
    module_name="sapm_1",
    racking="open_rack_glass_polymer",
    materials="OX003",
    temperature_model="sapm",
    irradiance_kwarg={"azimuth": 120, "tilt": 30},
    model_kwarg={"irrad_ref": 1100},
)

scene_temp.addModule(
    module_name="pvsyst_1",
    racking="freestanding",
    materials="OX003",
    temperature_model="pvsyst",
    irradiance_kwarg={"azimuth": 180, "tilt": 0},
    model_kwarg={"module_efficiency": 0.15},
)
scene_temp.addModule(
    module_name="sapm_2",
    racking="open_rack_glass_polymer",
    materials="OX003",
    temperature_model="sapm",
    irradiance_kwarg={"azimuth": 120, "tilt": 30},
    model_kwarg={"irrad_ref": 1000},
)
scene_temp.addModule(
    module_name="sapm_3",
    racking="open_rack_glass_polymer",
    materials="OX003",
    temperature_model="sapm",
    irradiance_kwarg={"azimuth": 180, "tilt": 0},
    model_kwarg={"irrad_ref": 1000},
)

scene_temp.addModule(
    module_name="pvsyst_2",
    racking="freestanding",
    materials="OX003",
    temperature_model="pvsyst",
    irradiance_kwarg={"azimuth": 180, "tilt": 0},
    model_kwarg={"module_efficiency": 0.2},
)

scene_temp.addJob(
    func=(pvdeg.temperature.temperature, {"cell_or_mod": "cell"}),
)

scene_temp.addJob(
    func=(pvdeg.degradation.vantHoff_deg, {"I_chamber": 1000, "temp_chamber": 25}),
)

scene_temp.addJob(
    func=(pvdeg.degradation.vantHoff_deg, {"I_chamber": 1000, "temp_chamber": 30}),
)

scene_temp.addJob(
    func=pvdeg.degradation.IwaVantHoff,
)

# %% [markdown]
# # Run and View Scenario Results

# %%
scene_temp.run()

scene_temp

# %%
scene_temp.dump()

# %% [markdown]
# # Plotting and Extracting Results
# These methods are independent of one another (i.e. you do not need to extract before plotting but both are shown below.)

# %%
import datetime

t0 = datetime.datetime(1970, 1, 1, 0, 0)
tf = datetime.datetime(1970, 1, 1, 23, 59)

# scene_temp.results is keyed by module_name, whose values are dicts keyed by
# job_id. Grab the first job_id from any module to use as the ('function', ...)
# target for extract/plot.
first_module_results = next(iter(scene_temp.results.values()))
first_job_id = next(iter(first_module_results.keys()))

temp_df = scene_temp.extract(
    ("function", first_job_id), tmy=True, start_time=t0, end_time=tf
)
display(temp_df)


# %%
scene_temp.plot(
    ("function", first_job_id),
    tmy=True,
    start_time=t0,
    end_time=tf,
    title="single day cell temperature",
)


# %% [markdown]
# # Create a Copy of a Scenario

# %%

# the scenario fixture ships inside the pvdeg package, so this resolves from a pip
# install as well as a repo clone
new_path = pvdeg.DATA_DIR / "temperature_and_degradation.json"

copy = pvdeg.scenario.Scenario.load_json(
    file_path=str(new_path),
    email="user@mail.com",
    api_key="DEMO_KEY",
)

# copy
