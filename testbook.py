import sys
from testbook import testbook

def monkeypatch_addLocation_code():
    # This string will be injected into the notebook to monkeypatch GeospatialScenario.addLocation
    return """
import pandas as pd
import xarray as xr
import pvdeg
import os

def monkeypatch_addLocation(self, *args, **kwargs):
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

pvdeg.GeospatialScenario.addLocation = monkeypatch_addLocation
"""

def monkeypatch_cells(tb):
    # Inject the monkeypatch code at the start of the notebook
    tb.inject(cell=0, code=monkeypatch_addLocation_code())

def main(notebook_path):
    with testbook(notebook_path, execute=True) as tb:
        monkeypatch_cells(tb)
        # Optionally, add assertions to check outputs

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_notebooks_with_testbook.py <notebook_path>")
        sys.exit(1)
    main(sys.argv[1])
