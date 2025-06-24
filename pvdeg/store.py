"""store.py"""

import xarray as xr
import pandas as pd
import numpy as np
import dask.array as da
import os

from pvdeg import METOROLOGICAL_DOWNLOAD_PATH


def my_path():
    """Find path to your zarr store of data if it exists."""
    if os.path.exists(os.path.join(METOROLOGICAL_DOWNLOAD_PATH, ".zattrs")):
        print(METOROLOGICAL_DOWNLOAD_PATH)

    else:
        print("Directory not found")


def _combine_geo_weather_meta(weather_ds: xr.Dataset, meta_df: pd.DataFrame):
    """Combine weather dataset and meta dataframe into a single dataset."""
    meta_ds = xr.Dataset.from_dataframe(meta_df).rename({"index": "gid"})

    combined = xr.merge([weather_ds, meta_ds])

    combined["Source"] = combined["Source"].astype(str)  # save as strings

    return combined


def _seperate_geo_weather_meta(
    ds_from_zarr: xr.Dataset,
):
    """
    Separate datasets.

    Take loaded dataset in the zarr store schema (weather and meta combined) and
    seperate it into `weather_ds` and `meta_df`.
    """
    ds_from_zarr["Source"] = ds_from_zarr["Source"].astype(
        object
    )  # geospatial.mapblocks needs this to be an object

    # there may be a more optimal way to do this
    data = np.column_stack(
        [
            ds_from_zarr.gid.to_numpy().astype(int),
            ds_from_zarr.latitude.to_numpy().astype(float),
            ds_from_zarr.longitude.to_numpy().astype(float),
            ds_from_zarr.altitude.to_numpy().astype(float),
            ds_from_zarr.Source.to_numpy(),
            ds_from_zarr.wind_height.to_numpy(),
        ]
    )

    seperated_meta_df = pd.DataFrame(
        data,
        columns=["gid", "latitude", "longitude", "altitude", "Source", "wind_height"],
    ).set_index("gid")

    seperated_weather_ds = ds_from_zarr.drop_vars(
        ("latitude", "longitude", "altitude", "Source", "wind_height")
    )

    return seperated_weather_ds, seperated_meta_df


def _make_coords_to_gid_da(ds_from_zarr: xr.Dataset):
    """Create a 2D indexable array that maps lat/lon to gid stored in zarr store."""
    # only want to do this if the arrays are dask arrays
    lats = ds_from_zarr.latitude.to_numpy()
    lons = ds_from_zarr.longitude.to_numpy()

    gids = -1 * da.empty((len(lats), len(lons)))

    points = xr.DataArray(
        data=gids,
        coords={"latitude": lats, "longitude": lons},
        dims=["latitude", "longitude"],
    )

    points.set_index(latitude="latitude", longitude="longitude")

    return points


def _create_sample_sheet(
    fill_value,
    latitude: float = 999,
    longitude: float = 999,
    altitude: float = -1,
    wind_height: int = -1,
    Source: str = "SampleSheet",
):
    """Create a dummy sample dataset containing weather for one gid.

    This will be called
    a sheet, a single location of weather_data from the dataset with the gid coordinate
    still present.

    The sizes of the dimensions of the sheet will be {"gid": 1, "time": 8760}

    Parameters
    ----------
    fill_value: numeric
        value to populate weather_ds single sheet with
    latitude: float
        dummy latitude WSG84
    longitude: float
        dummy longitude WSG84
    altitude: float
        dummy altitude of measured data [m]
    wind_height: int
        dummy height of measure sample dataset's wind measurement

    Returns
    -------
    sheet_ds : xr.Dataset
        Dummy weather data sheet for a single location using a dask array backend.
        As mentioned above this will look maintain the gid coordinate.
    meta_df : pd.DataFrame
        Dummy metadata for test location in pandas.DataFrame.
    """
    meta_dict = {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "wind_height": wind_height,
        "Source": Source,
    }

    meta_df = pd.DataFrame(meta_dict, index=[0])

    sheet_ds = pvgis_hourly_empty_weather_ds(gids_size=1)

    dummy_da = da.full(shape=(1, sheet_ds.sizes["time"]), fill_value=fill_value)

    for var in sheet_ds.data_vars:
        dim = sheet_ds[var].dims
        sheet_ds[var] = (dim, dummy_da)

    return sheet_ds, meta_df
