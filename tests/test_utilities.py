import os
import pandas as pd
import pvdeg
from rex import Outputs

from pvdeg import TEST_DATA_DIR

FILES = {
    "tmy3": os.path.join(TEST_DATA_DIR, "tmy3_pytest.csv"),
    "psm3": os.path.join(TEST_DATA_DIR, "psm3_pytest.csv"),
    "epw": os.path.join(TEST_DATA_DIR, "epw_pytest.epw"),
    "h5": os.path.join(TEST_DATA_DIR, "h5_pytest.h5"),
}

DSETS = [
    "temp_air",
    "albedo",
    "dew_point",
    "dhi",
    "dni",
    "ghi",
    "meta",
    "relative_humidity",
    "time_index",
    "wind_speed",
]


def test_gid_downsampling():
    pass


def test_write_gids():
    pass


def test_convert_tmy():
    """
    Test pvdeg.utilites.convert_tmy

    Requires:
    ---------
    tmy3 or tmy-like .csv weather file (WEATHERFILES['tmy3'])
    """
    pvdeg.utilities.convert_tmy(file_in=FILES["tmy3"], file_out=FILES["h5"])
    with Outputs(FILES["h5"], "r") as f:
        datasets = f.dsets
    assert datasets.sort() == DSETS.sort()
