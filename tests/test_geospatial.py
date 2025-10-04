import pvdeg
from pvdeg import TEST_DATA_DIR
import pandas as pd
import numpy as np
import xarray as xr
import os

import pytest

# NOTE ON UNCHUNKED INPUTS FOR TESTING
# The output template result will always be chunked (because they are dask arrays)
# this is fine for unchunked inputs if the number of chunks along all axes is 1
# AND the size of the chunk will contain all elements in the entire axis,

GEO_META = pd.read_csv(os.path.join(TEST_DATA_DIR, "summit-meta.csv"), index_col=0)

GEO_WEATHER = xr.load_dataset(os.path.join(pvdeg.TEST_DATA_DIR, "summit-weather.nc"))

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

    assert pvdeg.utilities.compare_templates(autotemplate_result, HUMIDITY_TEMPLATE)
    # custom function because we cant use equals or identical because of empty
    # like values


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
            raise ValueError(
                f"""
                            Need one chunk per axis for an unchunked input
                            dimension {k} has {len(v)} chunks.
                            """
            )


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


def mixed_res_dict(weather_df, meta):
    """Geospatial test function.

    returns have mixed dimensions. they are returned in dictionary form the first key
    value is a timeseries, the second key value is a float.
    """

    timeseries_df = pd.DataFrame(pvdeg.temperature.module(weather_df, meta))
    avg_temp = timeseries_df[0].mean()

    return {"temperatures": timeseries_df, "avg_temp": avg_temp}


def mixed_res_dataset(weather_df, meta):
    """Geospatial test function.

    returns have mixed dimensions. which are correctly stored in a xr.Dataset
    """

    return xr.Dataset(
        data_vars={
            "temperatures": (("time"), np.full((8760,), fill_value=80)),
            "avg_temp": 80,
        },
        coords={"time": pd.date_range(start="2001-01-01", periods=8760, freq="1h")},
    )


# functions must have homogenous return shapes
# see decorators.geospatial_quick_shape for documentation
# however it is possible to work around this
def test_mixed_res_dict():
    # dict template with varing dimensions
    mixed_res_dict_template = pvdeg.geospatial.output_template(
        ds_gids=GEO_WEATHER,
        shapes={
            "temperatures": ("gid", "time"),
            "avg_temp": ("gid",),
        },
    )

    with pytest.raises(
        NotImplementedError,
        match=r"function return type: <class 'dict'> not available",
    ):
        pvdeg.geospatial.analysis(
            weather_ds=GEO_WEATHER,
            meta_df=GEO_META,
            func=mixed_res_dict,
            template=mixed_res_dict_template,
        )


# this should not raise any errors
def test_mixed_res_dataset():
    template = pvdeg.geospatial.output_template(
        ds_gids=GEO_WEATHER,
        shapes={
            "temperatures": ("gid", "time"),
            "avg_temp": ("gid",),
        },
    )

    res = pvdeg.geospatial.analysis(
        weather_ds=GEO_WEATHER,
        meta_df=GEO_META,
        func=mixed_res_dataset,
        template=template,
    )

    assert isinstance(res, xr.Dataset)
