import pytest
import pvdeg
import pandas as pd
import xarray as xr
import os

print("conftest.py is being loaded")  # debugging line to be removed


def monkeypatch_addLocation(self, *args, **kwargs) -> None:
    """
    Mocker function to be monkey patched at runtime for Scenario.addLocation.

    Avoids psm3 api calls and uses local weather files instead.
    """
    self.gids, self.weather_data, self.meta_data = None, None, None

    GEO_META = pd.read_csv(
        os.path.join(pvdeg.TEST_DATA_DIR, "summit-meta.csv"), index_col=0
    )
    GEO_WEATHER = xr.load_dataset(
        os.path.join(pvdeg.TEST_DATA_DIR, "summit-weather.nc")
    )

    self.weather_data = GEO_WEATHER
    self.meta_data = GEO_META
    self.gids = GEO_WEATHER.gid.values


@pytest.fixture(autouse=True)
def mock_geospatial_data(monkeypatch):
    """
    Automatically applied fixture that mocks GeospatialScenario.addLocation
    to use local test data instead of making API calls.
    """
    monkeypatch.setattr(
        target=pvdeg.GeospatialScenario,
        name="addLocation",
        value=monkeypatch_addLocation,
    )
