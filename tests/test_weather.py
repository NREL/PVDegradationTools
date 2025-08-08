"""Using pytest to create unit tests for pvdeg.

to run unit tests, run pytest from the command line in the pvdeg directory to run
coverage tests, run py.test --cov-report term-missing --cov=pvdeg
"""

import os
import pandas as pd
import pvdeg
import pytest
import xarray as xr
from pvdeg.weather import map_meta

from pvdeg import TEST_DATA_DIR

FILES = {
    "tmy3": os.path.join(TEST_DATA_DIR, "tmy3_pytest.csv"),
    "psm3": os.path.join(TEST_DATA_DIR, "psm3_pytest.csv"),
    "epw": os.path.join(TEST_DATA_DIR, "epw_pytest.epw"),
    "h5": os.path.join(TEST_DATA_DIR, "h5_pytest.h5"),
}

DSETS = [
    "air_temperature",
    "albedo",
    "dew_point",
    "dhi",
    "dni",
    "ghi",
    "relative_humidity",
    "time_index",
    "wind_speed",
]
META_KEYS = [""]

DISTRIBUTED_PVGIS_WEATHER = xr.load_dataset(
    os.path.join(TEST_DATA_DIR, "distributed_pvgis_weather.nc")
)
DISTRIBUTED_PVGIS_META = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "distributed_pvgis_meta.csv"), index_col=0
)


def test_colum_name():
    df, meta_data = pvdeg.weather.read(
        os.path.join(TEST_DATA_DIR, "psm3_pytest.csv"), "csv"
    )
    assert "City" in meta_data.keys()
    assert "tz" in meta_data.keys()
    assert "Year" in df
    assert "dew_point" in df


def test_get():
    """Test with (lat,lon) and gid options."""
    # TODO: Test with AWS

    # #Test with lat, lon on NREL HPC
    # weather_db = 'NSRDB'
    # weather_id = (39.741931, -105.169891) #NREL
    # weather_arg = {'satellite' : 'GOES',
    #                'names' : 2021,
    #                'NREL_HPC' : True,
    #                'attributes' : ['air_temperature', 'wind_speed', 'dhi',
    #                                'ghi', 'dni','relative_humidity']}

    # weather_df, meta = pvdeg.weather.load(weather_db, weather_id, **weather_arg)

    # assert isinstance(meta, dict)
    # assert isinstance(weather_df, pd.DataFrame)
    # assert len(weather_df) != 0

    # #Test with gid on NREL HPC
    # weather_id = 1933572
    # weather_df, meta = pvdeg.weather.load(weather_db, weather_id, **weather_arg)
    pass


def test_read():
    """
    Test pvdeg.utilities.read_weather.

    TODO: enable the final assertion which checks column names.
    This may require troubleshooting with PVLIB devs. varaible mapping apears
    inconsistent.

    Requires:
    ---------
    WEATHERFILES dicitonary of all verifiable weather files and types
    """
    for type, path in FILES.items():
        if type != "h5":
            weather_df, meta = pvdeg.weather.read(file_in=path, file_type=type)
            assert isinstance(meta, dict)
            assert isinstance(weather_df, pd.DataFrame)
            assert len(weather_df) != 0
            # assert all(item in weather_df.columns for item in DSETS)


def test_read_h5():
    pass


def test_get_NSRDB_fnames():
    pass


def test_get_NSRDB():
    """Contained within get_weather()"""
    pass


def test_weather_distributed_no_client():
    with pytest.raises(
        RuntimeError, match="No Dask scheduler found. Ensure a dask client is running."
    ):
        # function should fail because we do not have a running dask scheduler or client
        pvdeg.weather.weather_distributed(
            database="PVGIS",
            coords=None,
        )


def test_weather_distributed_client_bad_database(capsys):
    pvdeg.geospatial.start_dask()

    with pytest.raises(
        NotImplementedError,
        match="Only 'PVGIS' and 'PSM3' are implemented, you entered fakeDB",
    ):
        pvdeg.weather.weather_distributed(
            database="fakeDB",
            coords=None,
        )

    captured = capsys.readouterr()
    assert "Connected to a Dask scheduler" in captured.out


def test_weather_distributed_pvgis():
    weather, meta, failed_gids = pvdeg.weather.weather_distributed(
        database="PVGIS",
        coords=[
            (39.7555, 105.2211),
            (40.7555, 105.2211),
        ],
    )

    assert DISTRIBUTED_PVGIS_WEATHER.equals(weather)
    assert DISTRIBUTED_PVGIS_META.equals(meta)
    assert failed_gids == []


def test_empty_weather_ds_invalid_database():
    """Test that emtpy_weather_ds raises ValueError for an invalid database."""
    gids_size = 10
    periodicity = "1h"
    invalid_database = "INVALID_DB"

    with pytest.raises(
        ValueError, match=f"database must be PVGIS, NSRDB, PSM3 not {invalid_database}"
    ):
        pvdeg.weather.empty_weather_ds(gids_size, periodicity, invalid_database)

def test_map_meta_dict():
    meta = {
        "Elevation": 150,
        "Time Zone": "UTC-7",
        "Longitude": -120.5,
        "Latitude": 38.5,
        "SomeKey": "value"
    }
    mapped = map_meta(meta)
    assert "altitude" in mapped and mapped["altitude"] == 150
    assert "tz" in mapped and mapped["tz"] == "UTC-7"
    assert "longitude" in mapped and mapped["longitude"] == -120.5
    assert "latitude" in mapped and mapped["latitude"] == 38.5
    assert "SomeKey" in mapped  # unchanged keys remain

def test_map_meta_dataframe():
    df = pd.DataFrame({
        "Elevation": [100, 200],
        "Time Zone": ["UTC-5", "UTC-6"],
        "Longitude": [-80, -81],
        "Latitude": [35, 36],
        "ExtraCol": [1, 2]
    })
    mapped_df = map_meta(df)
    assert "altitude" in mapped_df.columns
    assert "tz" in mapped_df.columns
    assert "longitude" in mapped_df.columns
    assert "latitude" in mapped_df.columns
    assert "ExtraCol" in mapped_df.columns
    # Original column names should not exist
    assert "Elevation" not in mapped_df.columns
    assert "Time Zone" not in mapped_df.columns

def test_map_meta_invalid_input():
    with pytest.raises(TypeError):
        map_meta(["invalid", "input"])
