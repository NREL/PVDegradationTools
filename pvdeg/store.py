from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import dask.array as da    
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

    if not os.path.exists(os.path.join(METOROLOGICAL_DOWNLOAD_PATH, ".zmetadata")): # no zstore in directory
        print("Creating Zarr")

        combined_ds.to_zarr(
            store=METOROLOGICAL_DOWNLOAD_PATH, 
            group=f"{group}-{periodicity}",
            mode="w-",      # only use for first time creating store
        )
    else: # store already exists
        print("adding to store")

        print("opening store")
        stored_ds = xr.open_zarr(
            store=METOROLOGICAL_DOWNLOAD_PATH,
            group=f"{group}-{periodicity}",
            # consolidated=True
        )

        lat_lon_gid_2d_map = _make_coords_to_gid_da(ds_from_zarr=stored_ds)        

        for gid, values in meta_df.iterrows():

            target_lat = values["latitude"]
            target_lon = values["longitude"]

            lat_exists = np.any(lat_lon_gid_2d_map.latitude == target_lat)
            lon_exists = np.any(lat_lon_gid_2d_map.longitude == target_lon)

            if lat_exists and lon_exists:
                print("(lat, lon) exists already")

                raise NotImplementedError

                # stored_gid = lat_lon_gid_2d_map.sel(latitude=target_lat, longitude=target_lon)

                # # overwrite previous value at that lat-lon, keeps old gid

                # # need to set the gid of the current "sheet" to the stored gid
                # updated_entry = combined_ds.loc[{"gid": gid}].assign_coords({"gid": stored_gid}) # this may need to be a list s.t. [stored_gid]
                # # loc may remove the gid dimension so we might have to add it back with .expand_dims

                # # overwrite the current entry at the gid = stored_gid entry of the zarr 
                # updated_entry.to_zarr(store=METOROLOGICAL_DOWNLOAD_PATH, group=f"{group}-{periodicity}", mode='w') 


            else: # coordinate pair doesnt exist and it needs to be added, this will be a HEAVY operation
                print("add entry to dataset")

                # we are trying to save 1 "sheet" of weather (weather at a single gid)
                # need to update the index to fit into the stored data after we concatenate

                # this concatenates along the the gid axis
                # gid has no guarantee of being unqiue but duplicate gids are fine for xarray
                # we slice so we can get a Dataset with dimensions of (gid, time) indexing to grab one gid will drop the gid dimension
                new_gid = stored_ds.sizes["gid"]

                weather_sheet = combined_ds.sel(gid=slice(gid))
                updated_entry = weather_sheet.assign_coords({"gid": [new_gid]})
                updated_entry.to_zarr(store=METOROLOGICAL_DOWNLOAD_PATH, group=f"{group}-{periodicity}", mode="a", append_dim="gid")

                # new_entry_added_ds = xr.concat([stored_ds, updated_entry], dim="gid")

                # new_entry_added_ds.to_zarr(store=METOROLOGICAL_DOWNLOAD_PATH, group=f"{group}-{periodicity}", mode="a", append_dim="gid")
                
    print(f"dataset saved to zarr store at {METOROLOGICAL_DOWNLOAD_PATH}")


def check_store():
    """Check if you have a zarr store at the default download path defined in pvdeg.config"""
    if os.path.exists(os.path.join(METOROLOGICAL_DOWNLOAD_PATH, ".zmetadata")):

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

    # if meta_df.index.name == 'index':
    meta_ds = xr.Dataset.from_dataframe(meta_df).rename({'index' : 'gid'})


    combined = xr.merge([weather_ds, meta_ds]).assign_coords(
        latitude=("gid", meta_ds.latitude.values),
        longitude=('gid', meta_ds.longitude.values),
    )

    combined["Source"] = combined["Source"].astype(str) # save as strings

    return combined


def _seperate_geo_weather_meta(
    ds_from_zarr: xr.Dataset,
):
    """
    Take loaded dataset in the zarr store schema (weather and meta combined) 
    and seperate it into `weather_ds` and `meta_df`.
    """

    ds_from_zarr["Source"] = ds_from_zarr["Source"].astype(object) # geospatial.mapblocks needs this to be an object

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