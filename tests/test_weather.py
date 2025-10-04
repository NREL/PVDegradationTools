"""Using pytest to create unit tests for pvdeg.

to run unit tests, run pytest from the command line in the pvdeg directory to run
coverage tests, run py.test --cov-report term-missing --cov=pvdeg
"""

import os
import pandas as pd
import pvdeg
import pytest
import xarray as xr
import pathlib
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

UNSORTED_TMY_DIR = [
    "/datasets/NSRDB/current/nsrdb_tmy-2024.h5",
    "/datasets/NSRDB/current/nsrdb_tmy-2021.h5",
    "/datasets/NSRDB/current/nsrdb_tmy-2022.h5",
    "/datasets/NSRDB/current/nsrdb_tmy-2023.h5",
]

SORTED_TMY_DIR = sorted(UNSORTED_TMY_DIR)

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
    pvdeg.geospatial.start_dask()

    weather, meta, failed_gids = pvdeg.weather.weather_distributed(
        database="PVGIS",
        coords=[
            (39.7555, 105.2211),
            (40.7555, 105.2211),
        ],
    )

    assert DISTRIBUTED_PVGIS_WEATHER.equals(weather)

    # Strict comparison - must have common columns to pass
    expected_meta = DISTRIBUTED_PVGIS_META
    common_cols = list(set(meta.columns) & set(expected_meta.columns))

    assert (
        len(common_cols) > 0
    ), f"No common columns. Actual: {list(meta.columns)}, expected: {list(expected_meta.columns)}"  # noqa

    # Compare the common columns
    pd.testing.assert_frame_equal(meta[common_cols], expected_meta[common_cols])

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
        "SomeKey": "value",
    }
    mapped = map_meta(meta)
    assert "altitude" in mapped and mapped["altitude"] == 150
    assert "tz" in mapped and mapped["tz"] == "UTC-7"
    assert "longitude" in mapped and mapped["longitude"] == -120.5
    assert "latitude" in mapped and mapped["latitude"] == 38.5
    assert "SomeKey" in mapped  # unchanged keys remain


def test_map_meta_dataframe():
    df = pd.DataFrame(
        {
            "Elevation": [100, 200],
            "Time Zone": ["UTC-5", "UTC-6"],
            "Longitude": [-80, -81],
            "Latitude": [35, 36],
            "ExtraCol": [1, 2],
        }
    )
    mapped_df = map_meta(df)
    assert "altitude" in mapped_df.columns
    assert "tz" in mapped_df.columns
    assert "longitude" in mapped_df.columns
    assert "latitude" in mapped_df.columns
    assert "ExtraCol" in mapped_df.columns
    # Original column names should not exist
    assert "Elevation" not in mapped_df.columns
    assert "Time Zone" not in mapped_df.columns
    assert "Longitude" not in mapped_df.columns
    assert "Latitude" not in mapped_df.columns


def test_map_meta_invalid_input():
    with pytest.raises(TypeError):
        map_meta(["invalid", "input"])


def test_get_invalid_database():
    with pytest.raises(NameError, match="Weather database not found."):
        pvdeg.weather.get(database="INVALID_DB", id=(39.7555, -105.2211))


def test_get_no_location():
    with pytest.raises(TypeError, match="Specify location via tuple"):
        pvdeg.weather.get(database="PSM3")


def test_get_pvgis():
    location = (48.86, 2.35)  # Paris area
    weather_df, meta = pvdeg.weather.get(database="PVGIS", id=location)

    assert isinstance(weather_df, pd.DataFrame)
    assert isinstance(meta, dict)
    assert len(weather_df) > 0
    assert "latitude" in meta
    assert "longitude" in meta
    assert "Source" in meta
    assert meta["Source"] == "PVGIS"
    assert "wind_height" in meta
    assert meta["wind_height"] == 10

    expected_columns = ["temp_air", "ghi", "dhi", "dni"]
    for col in expected_columns:
        assert col in weather_df.columns, f"Column {col} not found"


def test_get_psm3():
    location = (39.7555, -105.2211)  # Golden area
    weather_df, meta = pvdeg.weather.get(
        database="PSM3", id=location, api_key="DEMO_KEY", email="user@mail.com"
    )

    assert isinstance(weather_df, pd.DataFrame)
    assert isinstance(meta, dict)
    assert len(weather_df) > 0
    assert "latitude" in meta
    assert "longitude" in meta
    assert "Source" in meta
    assert meta["Source"] == "NSRDB"
    assert "wind_height" in meta
    assert meta["wind_height"] == 2

    expected_columns = ["temp_air", "ghi", "dhi", "dni"]
    for col in expected_columns:
        assert col in weather_df.columns, f"Column {col} not found"


def test_get_geospatial_not_implemented():
    with pytest.raises(NameError, match="Geospatial analysis not implemented"):
        pvdeg.weather.get(database="PVGIS", geospatial=True)


def test_get_meta_mapping():
    _, meta = pvdeg.weather.read(file_in=FILES["psm3"], file_type="csv")
    assert "tz" in meta and "Time Zone" not in meta
    assert "altitude" in meta and "Elevation" not in meta


def test_get_local_file():
    weather_df, meta = pvdeg.weather.get(
        database="local", id=0, file=FILES["psm3"]  # dummy gid
    )
    assert isinstance(weather_df, pd.DataFrame)
    assert isinstance(meta, dict)
    assert len(weather_df) > 0
    assert "latitude" in meta
    assert "longitude" in meta


def test_get_nsrdb_fnames_tmy(monkeypatch):
    called = {}

    def fake_glob(pattern):
        called["pattern"] = pattern
        return UNSORTED_TMY_DIR

    import glob

    monkeypatch.setattr(glob, "glob", fake_glob)

    files, hsds = pvdeg.weather.get_NSRDB_fnames(
        satellite="Americas",
        names="TMY",
        NREL_HPC=True,
    )

    # HPC path -> h5py (not HSDS)
    assert hsds is False
    assert files == SORTED_TMY_DIR
    # pattern must match HPC + Americas + *_tmy*.h5
    # assert called["pattern"] == "/datasets/NSRDB/current/*_tmy*.h5"
    path = called["pattern"]
    assert pathlib.Path(path).as_posix() == "/datasets/NSRDB/current/*_tmy*.h5"


def test_get_NSRDB_ds_has_kestrel_nsrdb_fnames_tmy(monkeypatch):
    """For TMY, get_NSRDB should store only the last element of the sorted list."""

    # Fake get_NSRDB_fnames to return UNSORTED list + hsds flag
    def fake_get_NSRDB_fnames(satellite, names, NREL_HPC):
        assert satellite == "Americas"
        assert names == "TMY"
        assert NREL_HPC is True
        return SORTED_TMY_DIR, False

    # Fake ini_h5_geospatial to return an empty dataset/meta (no attrs set here)
    def fake_ini_h5_geospatial(nsrdb_fnames):
        # ensure get_NSRDB passes the sliced list in
        # (TMY → single last element from sorted list)
        assert nsrdb_fnames == [SORTED_TMY_DIR[-1]]
        ds = xr.Dataset()  # no attrs at this point; get_NSRDB adds them
        meta = pd.DataFrame()
        return ds, meta

    monkeypatch.setattr(pvdeg.weather, "get_NSRDB_fnames", fake_get_NSRDB_fnames)
    monkeypatch.setattr(pvdeg.weather, "ini_h5_geospatial", fake_ini_h5_geospatial)

    ds, meta = pvdeg.weather.get_NSRDB(
        satellite="Americas",
        names="TMY",
        NREL_HPC=True,
        geospatial=True,
    )

    # The attribute is added inside get_NSRDB
    assert "kestrel_nsrdb_fnames" in ds.attrs
    assert ds.attrs["kestrel_nsrdb_fnames"] == [SORTED_TMY_DIR[-1]]
    assert isinstance(meta, pd.DataFrame)


def test_get_NSRDB_ds_has_kestrel_nsrdb_fnames_year(monkeypatch):
    """For a specific year, get_NSRDB should store the full sorted list."""

    def fake_get_NSRDB_fnames(satellite, names, NREL_HPC):
        assert satellite == "Americas"
        assert names == 2024
        assert NREL_HPC is False
        return SORTED_TMY_DIR, True

    def fake_ini_h5_geospatial(nsrdb_fnames):
        # For year input, no slicing; expect the entire list (whatever order
        # get_NSRDB_fnames returned). The attribute stores exactly what
        # get_NSRDB passes in — the function under test does not re-sort here.
        assert nsrdb_fnames == SORTED_TMY_DIR
        return xr.Dataset(), pd.DataFrame()

    monkeypatch.setattr(pvdeg.weather, "get_NSRDB_fnames", fake_get_NSRDB_fnames)
    monkeypatch.setattr(pvdeg.weather, "ini_h5_geospatial", fake_ini_h5_geospatial)

    ds, meta = pvdeg.weather.get_NSRDB(
        satellite="Americas",
        names=2024,
        NREL_HPC=False,
        geospatial=True,
    )

    assert "kestrel_nsrdb_fnames" in ds.attrs
    assert ds.attrs["kestrel_nsrdb_fnames"] == SORTED_TMY_DIR
    assert isinstance(meta, pd.DataFrame)
