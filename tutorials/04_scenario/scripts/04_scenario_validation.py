# %% [markdown]
# # Validation: Multilayer Pipeline vs Direct Function Calls
#
# Validates multilayer `Scenario.run()` by comparison against legacy individual
# layer-by-layer and job-by-job approach.

# %%
import pvdeg
import os
import json
import subprocess
import numpy as np
import pandas as pd

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

# %%
# Reference result 1: degradation_spectral called directly
ref_spectral = pvdeg.degradation.degradation_spectral(
    spectra=SPECTRA["Spectra"],
    rh=SPECTRA["RH"],
    temp=SPECTRA["Temperature"],
    wavelengths=wavelengths,
    time=SPECTRA.index,
)
print(f"Reference degradation_spectral: {ref_spectral}")

# %%
# Reference result 2: arrhenius called directly with weather_df
ref_arrhenius = pvdeg.degradation.arrhenius(weather_df=weather_df)
print(f"Reference arrhenius: {ref_arrhenius}")

# %%
# multilayer result

scenario = pvdeg.Scenario(
    name="validation-run",
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

scenario.run()

job_ids = list(scenario.results["test_module"].keys())
pipeline_spectral = scenario.results["test_module"][job_ids[0]]
pipeline_arrhenius = scenario.results["test_module"][job_ids[1]]

print(f"Pipeline degradation_spectral: {pipeline_spectral}")
print(f"Pipeline arrhenius:            {pipeline_arrhenius}")
