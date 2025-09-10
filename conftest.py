"""
Pytest configuration for notebook testing.

This file automatically applies monkey patches during nbval testing
to avoid HPC/API calls while keeping notebooks usable for regular users.
"""

import pytest
import os
import sys
import pandas as pd
import xarray as xr
import pvdeg


def monkeypatch_addLocation(self, *args, **kwargs) -> None:
    """
    Mock function to replace GeospatialScenario.addLocation during testing.

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
def mock_hpc_connections(monkeypatch):
    """
    apply fixture only during nbval testing.
    """
    # Only apply monkey patch during nbval testing
    if "--nbval" in sys.argv or any("nbval" in arg for arg in sys.argv):
        monkeypatch.setattr(
            target=pvdeg.GeospatialScenario,
            name="addLocation",
            value=monkeypatch_addLocation,
        )