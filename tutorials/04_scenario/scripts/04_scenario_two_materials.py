# %% [markdown]
# # Two-Material Scenario
#
# 1. Load two materials from the pvdeg material databases
# 2. Bind a different degradation function to each material layer
# 3. Run the pipeline and inspect the results

# %%
import pvdeg
import os
import pandas as pd
import json
import numpy as np

# Data files are read from pvdeg.DATA_DIR (the packaged data directory).
# This is robust regardless of os.getcwd(), which Scenario.__init__
# changes by calling os.chdir() into the job folder.

# %% [markdown]
# ## Load weather data
#
# Read a local PSM4 weather file that ships with the pvdeg tutorials.

# %%
weather_df = pd.read_csv(
    os.path.join(pvdeg.DATA_DIR, "psm4_golden.csv"),
    index_col=0,
    parse_dates=True,
)
with open(os.path.join(pvdeg.DATA_DIR, "meta_golden.json"), "r") as f:
    meta = json.load(f)

wavelengths = np.array(range(280, 420, 20))
SPECTRA = pd.read_csv(os.path.join(pvdeg.DATA_DIR, "spectra.csv"), header=0, index_col=0)

# %% [markdown]
# ## Create the scenario and load two materials
#
# Pass `materials` as a dict to assign a named layer to each material.
# Each entry points to a material key in one of the pvdeg JSON databases
# (`"O2permeation"`, `"H2Opermeation"`, or `"AApermeation"`).
#
# Here we load **OX003** (EVA encapsulant) and **OX004** (AAA polyamide backsheet)
# from `O2permeation.json` as two separate layers.

# %%
scenario = pvdeg.Scenario(
    name="two-material-demo",
    weather_data=weather_df,
    meta_data=meta,
)

scenario.addModule(
    module_name="test_module",
    materials={
        "encapsulant": {
            "material_file": "O2permeation",
            "material_name": "OX003",
        },
        "backsheet": {
            "material_file": "O2permeation",
            "material_name": "OX004",
        },
    },
)

# %% [markdown]
# ## Add one degradation job per material layer
#
# Use a 2-tuple `(function, layer_name)` so that `run()` knows which material
# parameters to inject for each job.

# %%
# degradation_spectral does not use weather_df/meta at all — its inputs are
# passed directly as extra kwargs via the 3-tuple (func, kwargs, layer_name).
# arrhenius only needs weather_df, which the scenario injects automatically.

scenario.addJob(
    func=[
        (
            pvdeg.degradation.degradation_spectral,
            {
                "spectra": SPECTRA["Spectra"],
                "rh": SPECTRA["RH"],
                "temp": SPECTRA["Temperature"],
                "wavelengths": wavelengths,
                "time": SPECTRA.index,
            },
            "encapsulant",
        ),
        (pvdeg.degradation.arrhenius, "backsheet"),
    ]
)

# %% [markdown]
# ## Run the pipeline

# %%
scenario.run()

# %% [markdown]
# ## Inspect results
#
# `results` is a nested dict: `results[module_name][job_id]`.
# Use `display(scenario)` to see job IDs, then index into them directly.

# %%
display(scenario)

# %%
module_results = scenario.results["test_module"]

for job_id, result in module_results.items():
    print(f"job {job_id}: {result}")
