import sys
from testbook import testbook


def monkeypatch_addLocation():
    """String to monkeypatch GeospatialScenario.addLocation"""
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


def monkeypatch_hpc_check():
    """String to monkeypatch pvdeg.utilities.nrel_kestrel_check"""
    return """
def monkeypatch_nrel_kestrel_check():
    pass # This function now does nothing, preventing the ConnectionError.

pvdeg.utilities.nrel_kestrel_check = monkeypatch_nrel_kestrel_check
"""


def monkeypatch_cells(tb):
    # Inject hpc_check at the beginning to patch before any pvdeg import
    tb.inject(monkeypatch_hpc_check(), 0)

    # Find a cell that contains 'import pvdeg' and inject addLocation after it
    for i, cell in enumerate(tb.cells):
        if 'import pvdeg' in str(cell.source):
            # Inject the addLocation monkey patch after the import cell
            tb.inject(monkeypatch_addLocation(), i + 1)
            break
    else:
        # Fallback: inject at the beginning if no pvdeg import found
        # Note: hpc_check is already at 0, so this starts at 1.
        tb.inject(monkeypatch_addLocation(), 1)


def main(notebook_path):
    with testbook(notebook_path, execute=False) as tb:
        monkeypatch_cells(tb)
        tb.execute()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python notebooks_testbook.py <notebook_path>")
        sys.exit(1)
    main(sys.argv[1])
