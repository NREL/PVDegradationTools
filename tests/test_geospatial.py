import pvdeg
from pvdeg import TEST_DATA_DIR
import pickle
import pandas as pd
import numpy as np
import xarray as xr
import os


GEO_META = pd.read_csv(os.path.join(TEST_DATA_DIR, "summit-meta.csv"), index_col=0)

with open(os.path.join(TEST_DATA_DIR, "summit-weather.pkl"), "rb") as f:
    GEO_WEATHER = pickle.load(f)

HUMIDITY_TEMPLATE = xr.open_dataset(
    os.path.join(TEST_DATA_DIR, "humidity_template.nc")
).compute()


def test_analysis_standoff():
    res_ds = pvdeg.geospatial.analysis(
        weather_ds=GEO_WEATHER,
        meta_df=GEO_META,
        func=pvdeg.standards.standoff,
    )

    data_var = res_ds["x"]

    # Stack the latitude and longitude coordinates into a single dimension
    # convert to dataframe, this can be done with xr.dataset.to_dataframe as well
    stacked = data_var.stack(z=("latitude", "longitude"))
    latitudes = stacked["latitude"].values
    longitudes = stacked["longitude"].values
    data_values = stacked.values
    combined_array = np.column_stack((latitudes, longitudes, data_values))

    res = pd.DataFrame(combined_array).dropna()
    ans = pd.read_csv(
        os.path.join(TEST_DATA_DIR, "summit-standoff-res.csv"), index_col=0
    )
    res.columns = ans.columns

    pd.testing.assert_frame_equal(res, ans, check_dtype=False, check_names=False)


def test_autotemplate():
    autotemplate_result = pvdeg.geospatial.auto_template(
        func=pvdeg.humidity.module, ds_gids=GEO_WEATHER
    ).compute()

    assert pvdeg.utilities.compare_templates(
        autotemplate_result, HUMIDITY_TEMPLATE
    )  # custom function because we cant use equals or identical because of empty like values


def test_template():
    shapes = {
        "RH_surface_outside": ("gid", "time"),
        "RH_front_encap": ("gid", "time"),
        "RH_back_encap": ("gid", "time"),
        "RH_backsheet": ("gid", "time"),
    }

    manual_template = pvdeg.geospatial.output_template(
        shapes=shapes, ds_gids=GEO_WEATHER
    ).compute()

    assert pvdeg.utilities.compare_templates(manual_template, HUMIDITY_TEMPLATE)
