"""
Using pytest to create unit tests for pvdeg

to run unit tests, run pytest from the command line in the pvdeg directory
to run coverage tests, run py.test --cov-report term-missing --cov=pvdeg
"""

import os
import json
import pandas as pd
import pvdeg
from rex import Outputs

from pvdeg import TEST_DATA_DIR, DATA_LIBRARY

fname="kinetic_parameters.json"
fpath = os.path.join(DATA_LIBRARY, fname)

with open(fpath) as f:
    data = json.load(f)

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

def test_get_kinetics():
    """
    Test pvdeg.utilities.get_kinetics

    Requires:
    --------
    data : dict, from DATA_LIBRARY/kinetic_parameters.json 
    """
    
    result = pvdeg.utilities.get_kinetics('repins')

    assert data['repins'] == result
