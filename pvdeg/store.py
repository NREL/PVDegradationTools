from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import zarr
import os

from pvdeg import METOROLOGICAL_DOWNLOAD_PATH

def get(group):
    """
    Extract a weather xarray dataset and metadata pandas dataframe from your zarr store. 
    `get` pulls the entire datastore into these objects. PVDeg does not make indexing available at this stage. 
    This is practical because all datavariables are stored in dask arrays so they are loaded lazily instead of into memmory when this is called.
    Choose the points you need after this method is called by using `sel`, `isel`, `loc, `iloc`.

    `store.get` is meant to match the API of other geospatial weather api's from pvdeg like `pvdeg.weather.get`, `pvdeg.weather.distributed_weather`, `GeospatialScenario.get_geospatial_data`

    Parameters
    -----------
    group : str
        name of the group to access from your local zarr store. 
        Groups are created automatically in your store when you save data using `pvdeg.store.store`.

        *From `pvdeg.store.store` docstring*   
        Hourly PVGIS data will be saved to "PVGIS-1hr", 30 minute PVGIS to "PVGIS-30min", similarly 15 minute PVGIS will be saved to "PVGIS-15min"

    Returns
    -------
    weather_ds : xr.Dataset
        Weather data for all locations requested in an xarray.Dataset using a dask array backend. This may be larger than memory.
    meta_df : pd.DataFrame
        Pandas DataFrame containing metadata for all requested locations. Each row maps to a single entry in the weather_ds.
    """

    combined_ds = xr.open_zarr(
        store=METOROLOGICAL_DOWNLOAD_PATH,
        group=group 
    )

    weather_ds, meta_df = _seperate_geo_weather_meta(ds_from_zarr=combined_ds)

    return weather_ds, meta_df

def store(weather_ds, meta_df):
    """
    Add geospatial meteorolical data to your zarr store. Data will be saved to the correct group based on its periodicity.

    Hourly PVGIS data will be saved to "PVGIS-1hr", 30 minute PVGIS to "PVGIS-30min", similarly 15 minute PVGIS will be saved to "PVGIS-15min"

    Parameters
    -----------
    weather_ds : xr.Dataset
        Weather data for all locations requested in an xarray.Dataset using a dask array backend. This may be larger than memory.
    meta_df : pd.DataFrame
        Pandas DataFrame containing metadata for all requested locations. Each row maps to a single entry in the weather_ds.

    Returns
    --------
    None
    """

    group = meta_df.iloc[0]["Source"]
    rows_per_entry = weather_ds.isel(gid=0).time.size

    if rows_per_entry == 8760:
        periodicity = "1hr"
    elif rows_per_entry == 17520:
        periodicity = "30min"
    elif rows_per_entry == 35040:
        periodicity = "15min"
    else:
        raise ValueError(f"first location to store has {rows_per_entry} rows, must have 8670, 17520, 35040 rows")

    combined_ds = _combine_geo_weather_meta(weather_ds, meta_df)

    # what mode should this be
    # we want to add to indexes if need be or overwrite old ones
    combined_ds.to_zarr(
        store=METOROLOGICAL_DOWNLOAD_PATH, 
        group=f"{group}-{periodicity}"
    )

    print(f"dataset saved to zarr store at {METOROLOGICAL_DOWNLOAD_PATH}")



def check_store():
    """Check if you have a zarr store at the default download path defined in pvdeg.config"""
    if os.path.exists(os.path.join(METOROLOGICAL_DOWNLOAD_PATH, ".zattrs")):

        size = sum(f.stat().st_size for f in METOROLOGICAL_DOWNLOAD_PATH.glob('**/*') if f.is_file())

        print(f"""
            You have a zarr store at {METOROLOGICAL_DOWNLOAD_PATH}.

            It has {size} bytes.
            """)

    elif os.path.exists(METOROLOGICAL_DOWNLOAD_PATH):
        print(f"You have a directory but no zarr store at {METOROLOGICAL_DOWNLOAD_PATH}")

    else:
        raise FileNotFoundError(f"No Directory exists at {METOROLOGICAL_DOWNLOAD_PATH}. Data has not been saved here.")


def my_path():
    """Finds path to your zarr store of data if it exists"""
    if os.path.exists(os.path.join(METOROLOGICAL_DOWNLOAD_PATH, ".zattrs")):
        print(METOROLOGICAL_DOWNLOAD_PATH)

    else:
        print("Directory not found")

def _combine_geo_weather_meta(
    weather_ds: xr.Dataset, 
    meta_df: pd.DataFrame
    ):
    """Combine weather dataset and meta dataframe into a single dataset"""

    meta_ds = xr.Dataset.from_dataframe(meta_df)
    # we could do some encoding scheme here, dont need to store source? unless the zarr compression handles it for us

    meta_ds['gid'] = meta_ds['index'].values.astype(np.int32)
    meta_ds = meta_ds.drop_vars(["index"])

    combined = xr.merge([weather_ds, meta_ds]).assign_coords(
        latitude=("gid", meta_ds.latitude.values),
        longitude=('gid', meta_ds.longitude.values),
    )

    return combined


def _seperate_geo_weather_meta(
    ds_from_zarr: xr.Dataset,
):
    """
    Take loaded dataset in the zarr store schema (weather and meta combined) 
    and seperate it into `weather_ds` and `meta_df`.
    """

    # there may be a more optimal way to do this
    data = np.column_stack(
        [
        ds_from_zarr.gid.to_numpy(),
        ds_from_zarr.latitude.to_numpy(),
        ds_from_zarr.longitude.to_numpy(),
        ds_from_zarr.altitude.to_numpy(),
        ds_from_zarr.Source.to_numpy(),
        ds_from_zarr.wind_height.to_numpy(),
        ]
    )

    seperated_meta_df = pd.DataFrame(data, columns=["gid", "latitude", "longitude", "altitude", "Source", "wind_height"]).set_index("gid")

    seperated_weather_ds = ds_from_zarr.drop_vars(("latitude", "longitude", "altitude", "Source", "wind_height"))

    return seperated_weather_ds, seperated_meta_df

def _make_coords_to_gid_da(
    ds_from_zarr: xr.Dataset
    ):
    """Create a 2D indexable array that maps coordinates (lat and lon) to gid stored in zarr store"""

    import dask.array as da    

    # only want to do this if the arrays are dask arrays
    lats = ds_from_zarr.latitude.to_numpy()
    lons = ds_from_zarr.longitude.to_numpy()

    gids = -1 * da.empty((len(lats), len(lons)))

    points = xr.DataArray(
        data=gids,
        coords={
            "latitude": lats,
            "longitude": lons
        },
        dims=["latitude", "longitude"],
    )

    points.set_index(latitude="latitude", longitude="longitude")
    
    return points