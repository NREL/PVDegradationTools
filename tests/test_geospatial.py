import pvdeg
from pvdeg import TEST_DATA_DIR
import pickle
import pandas as pd
import numpy as np
import xarray as xr
import os

# NOTE ON UNCHUNKED INPUTS FOR TESTING
# The output template result will always be chunked (because they are dask arrays)
# this is fine for unchunked inputs if the number of chunks along all axes is 1
# AND the size of the chunk will contain all elements in the entire axis,

GEO_META = pd.read_csv(os.path.join(TEST_DATA_DIR, "summit-meta.csv"), index_col=0)

with open(os.path.join(TEST_DATA_DIR, "summit-weather.pkl"), "rb") as f:
    GEO_WEATHER = pickle.load(f)  # unchunked

HUMIDITY_TEMPLATE = xr.open_dataset(
    os.path.join(TEST_DATA_DIR, "humidity_template.nc"), engine="h5netcdf"
).compute()  # unchunked


# unchunked input
def test_analysis_standoff_unchunked():
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
    assert not res_ds.chunks  # output is never chunked


# chunked input
def test_analysis_standoff_chunked():
    chunked_weather = GEO_WEATHER.chunk({"gid": 3})  # artifically chunked

    # the result is always unchunked, checking if function works on chunked input
    res_ds = pvdeg.geospatial.analysis(
        weather_ds=chunked_weather,
        meta_df=GEO_META,
        func=pvdeg.standards.standoff,
    )

    data_var = res_ds["x"]
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
    assert not res_ds.chunks  # output is never chunked, even if input is chunked


def test_autotemplate():
    autotemplate_result = pvdeg.geospatial.auto_template(
        func=pvdeg.humidity.module, ds_gids=GEO_WEATHER
    ).compute()

    assert pvdeg.utilities.compare_templates(
        autotemplate_result, HUMIDITY_TEMPLATE
    )  # custom function because we cant use equals or identical because of empty like values


def test_output_template_unchunked():
    shapes = {
        "RH_surface_outside": ("gid", "time"),
        "RH_front_encap": ("gid", "time"),
        "RH_back_encap": ("gid", "time"),
        "RH_backsheet": ("gid", "time"),
    }

    manual_template = pvdeg.geospatial.output_template(
        shapes=shapes, ds_gids=GEO_WEATHER
    )

    assert pvdeg.utilities.compare_templates(manual_template, HUMIDITY_TEMPLATE)
    for k, v in manual_template.chunks.items():
        if len(v) != 1:
            raise ValueError(f"""
                            Need one chunk per axis for an unchunked input
                            dimension {k} has {len(v)} chunks.
                            """)


def test_output_template_chunked():
    chunked_weather = GEO_WEATHER.chunk({"gid": 3})  # artifically chunked

    shapes = {
        "RH_surface_outside": ("gid", "time"),
        "RH_front_encap": ("gid", "time"),
        "RH_back_encap": ("gid", "time"),
        "RH_backsheet": ("gid", "time"),
    }

    chunked_template = pvdeg.geospatial.output_template(
        shapes=shapes, ds_gids=chunked_weather
    )

    assert pvdeg.utilities.compare_templates(
        chunked_template, HUMIDITY_TEMPLATE.chunk({"gid": 3})
    )
