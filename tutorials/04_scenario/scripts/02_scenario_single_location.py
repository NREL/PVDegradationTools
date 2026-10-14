# %% [markdown]
# # Single Location
#
# Author: Tobin Ford | tobin.ford@nrel.gov
#
# 2024
# ****
#
# A simple object orented workflow walkthrough using pvdeg.

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
# # Define Single Point Scenario Object
# Scenario is a general class that can be used to replace the legacy functional pvdeg analysis approach with an object orented one. ``Scenario`` can preform single location or geospatial analysis. The scenario constructor takes many arguments but the only required one for the following use cases is the ``name`` attribute. It is visible in when we display the entire scenario and is present in the file of saved information about the scenario. We also need to provide the class constructor with our API key and email.
#
# A way around this is to provide the weather and metadata in the pipeline job arguments or you can load data from somewhere else and provide it in the same fashion.
#
# <div class="alert alert-block alert-info">
# <b>Please use your own API key: The block below makes an NSRDB API to get weather and meta data. This tutorial will work with the DEMO Key provided, but it will take you less than 3 minutes to obtain your own at <a ref="https://developer.nlr.gov/signup/">https://developer.nlr.gov/signup/</a> so register now.)
# </div>

# %%
simple_scenario = pvdeg.Scenario(
    name="Point Minimum Standoff", email="user@mail.com", api_key="DEMO_KEY"
)

# %% [markdown]
# # Adding A Location
# To add a single point using data from the Physical Solar Model (PSM3), simply feed the scenario a single coordinate in tuple form via the ``addLocation`` method. Currently this is the only way to add a location to a non-geospatial scenario, all of the other arguments are unusable when ``Scenario.geospatial == False``.
#
# Attempting to add a second location by calling the method again with a different coordinate pair will overwrite the old location data stored in the class instance.

# %%
simple_scenario.addLocation(
    lat_long=(25.783388, -80.189029),
)

# %%
simple_scenario.weather_data

# %% [markdown]
# # Scenario Pipelines
#
# The pipeline is a list of tasks called jobs for the scenario to run. We will populate the pipeline with a list of jobs before executing them all at once.
#
# To add a job to the pipeline use the ``updatePipeline`` method. Two examples of adding functions to the pipeline will be shown below.

# %% [markdown]
# # Adding a job without function arguments
#
# The simplest case of adding a job to the pipeline is when it only requires us to provide simple weather and metadata. In the function definition and docstring these appear as ``weather_df`` and ``meta``. Since these attributes are contained in our scenario class instance we do not have to worry about them. We can simply add the function as shown below.

# %%
simple_scenario.addJob(func=pvdeg.standards.standoff)

# %% [markdown]
# # Adding a job with function arguments
#
# When adding a job that contains a function requiring other arguments such as ``solder_fatigue`` which requires a value for ``wind_factor``, we will need to provide it. The most straightforeward way to do this is using a kwargs dictionary and passing it to the function. We do not unpack the dictionary before passing it. This is done inside of the scenario at pipeline runtime (when ``runPipeline`` is called).

# %%
kwargs = {"wind_factor": 0.33}

simple_scenario.addJob(func=(pvdeg.fatigue.solder_fatigue, kwargs))

# %% [markdown]
# # Adding a job with weather and metadata from outside of the class
# ## Not functional
#
# could just directly set weather data with scenario.weather_data = weather and scenario.meta_data = meta but that would only work for all of the jobs in the pipeline
#
# Say local weather data is available or other, if we want to use this rather than the PSM3 data at a latitude and longitude we can also provide the weather and metadata in the function arguments. This is probably the best if avoided but follows the same syntax as providing other function arguments. See the example below.

# %%
PSM_FILE = os.path.join(pvdeg.DATA_DIR, "psm3_demo.csv")
weather, meta = pvdeg.weather.read(PSM_FILE, "psm")

kwargs = {"weather_df": weather, "meta": meta}

simple_scenario.addJob(func=(pvdeg.standards.standoff, kwargs))

# FIX THIS CASE IN SCENARIO CLASS
# (simple_scenario.pipeline[1]['job'])(**simple_scenario.pipeline[1]['params'])

# %% [markdown]
# # View Scenario
#
# The ``viewScenario`` method provides an overview of the information contained within your scenario object. Here you can see if it contains the location weather and metadata. As well as the jobs in the pipeline and their arguments.

# %%
simple_scenario.viewScenario()

# %% [markdown]
# # Display
#
# The fancier cousin of viewScenario. Only works in a jupyter environemnt as it uses a special ipython backend to render the html and javascript.
#
# It can be called with just the Scenario instance as follows
# `simple_scenario`
#
# or using the display function
# `display(simple_scenario)`

# %%
simple_scenario

# %% [markdown]
# # Executing Pipeline Jobs
# To run the pipeline after we have populated it with the desired jobs call the ``runPipeline`` method on our scenario instance. This will run all of the jobs we have previously added. The functions that need weather and metadata will grab it from the scenario instance using the correct location added above. The pipeline jobs results will be saved to the scenario instance.

# %%
simple_scenario.run()

# %% [markdown]
# # Results Series ##
# We will use a series to store the various return values of functions run in our pipeline. These can partially obfuscate the dataframes within them so to access the dataframes, use the function name to access it. To get one of the results we can index it using dictionary syntax. If the job was called `'KSDJQ'` do `'simple_scenario.results['KSDJQ']` to directly access the result for that job

# %%
print(simple_scenario.results)
print("We can't see out data in here so we need to do another step", end="\n\n")

# to see all available ouputs of results do
print(
    f"this is the list of all available frames in results : {simple_scenario.results.index}\n"
)

# loop over all results and display
for keys, results in simple_scenario.results.items():
    print(keys)
    display(results)

# %% [markdown]
# # Cleaning Up the Scenario
#
# Each scenario object creates a directory named ``pvd_job_...`` that contains information about the scenario instance. To remove the directory and all of its information call ``clean`` on the scenario. This will permanently delete the directory created by the scenario.

# %%
simple_scenario.clean()
