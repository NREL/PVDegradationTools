import sys
from testbook import testbook
import os
import pvdeg


def monkeypatch_addLocation():
    """String to monkeypatch GeospatialScenario.addLocation"""
    return """
import pandas as pd
import xarray as xr
import pvdeg
import os

def monkeypatch_addLocation(self, *args, **kwargs):
    print("Monkeypatch for addLocation applied!")
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
    print("Monkeypatch for nrel_kestrel_check applied!")
    pass # This function now does nothing, preventing the ConnectionError.

pvdeg.utilities.nrel_kestrel_check = monkeypatch_nrel_kestrel_check
"""

def monkeypatch_cells(tb):
    # Find a cell that contains 'import pvdeg' and inject after it
    for i, cell in enumerate(tb.cells):
        if 'import pvdeg' in str(cell.source):
            # Inject both monkey patches after the import cell
            tb.inject(monkeypatch_addLocation(), i + 1)
            tb.inject(monkeypatch_hpc_check(), i + 2)
            print(f"Monkeypatches injected after cell {i}")
            break
    else:
        # Fallback: inject at the beginning if no pvdeg import found
        tb.inject(monkeypatch_addLocation(), 0)
        tb.inject(monkeypatch_hpc_check(), 1)
        print("Monkeypatches injected at beginning")

def main(notebook_path):
    with testbook(notebook_path, execute=False) as tb:
        monkeypatch_cells(tb)
        tb.execute()
    print("Notebook executed successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python notebooks_testbook.py <notebook_path>")
        sys.exit(1)
    main(sys.argv[1])