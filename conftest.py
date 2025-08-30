import pytest
import pvdeg
import pandas as pd
import xarray as xr
import os


def monkeypatch_addLocation(self, *args, **kwargs):
    """
    Mock function for GeospatialScenario.addLocation to avoid HPC calls during testing.
    This function directly loads local test data instead of calling pvdeg.weather.get().
    """
    GEO_META = pd.read_csv(
        os.path.join(pvdeg.TEST_DATA_DIR, "summit-meta.csv"), index_col=0
    )
    GEO_WEATHER = xr.load_dataset(
        os.path.join(pvdeg.TEST_DATA_DIR, "summit-weather.nc")
    )

    # Ensure altitude column exists for downstream processing
    if "altitude" not in GEO_META.columns:
        GEO_META["altitude"] = 1000.0

    self.weather_data = GEO_WEATHER
    self.meta_data = GEO_META
    self.gids = GEO_WEATHER.gid.values


@pytest.fixture(autouse=True)
def patch_geospatial_scenario(monkeypatch):
    """
    Auto-apply monkey patch for GeospatialScenario.addLocation.
    This single patch prevents the network call.
    """
    monkeypatch.setattr(
        pvdeg.GeospatialScenario, "addLocation", monkeypatch_addLocation
    )
