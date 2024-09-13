"""
Using pytest to create unit tests for pvdeg

to run unit tests, run pytest from the command line in the pvdeg directory
to run coverage tests, run py.test --cov-report term-missing --cov=pvdeg
"""

import os
import pandas as pd
import numpy as np
import pvdeg
from rex import Outputs
import json

import pytest
from pvdeg import TEST_DATA_DIR, DATA_DIR

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


# def test_write_gids():
#     pass


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


def test_gid_downsampling():
    lats = np.linspace(-90, 90, num=10)
    longs = np.linspace(-180, 180, num=10)
    gids = pd.DataFrame({"latitude": lats, "longitude": longs}, index=np.arange(0, 10))

    downsapled_gids = pvdeg.utilities.gid_downsampling(gids, n=1)
    no_downsample_gids = pvdeg.utilities.gid_downsampling(gids, n=0)

    np.testing.assert_array_almost_equal(downsapled_gids[1], np.arange(0, 10, step=2))
    np.testing.assert_array_almost_equal(no_downsample_gids[1], np.arange(0, 10))


def test_get_kinetics_bad():
    # no name provided case
    fpath = os.path.join(DATA_DIR, "kinetic_parameters.json")
    with open(fpath) as f:
        data = json.load(f)
    parameters_list = data.keys()

    desired_output = ("Choose a set of kinetic parameters:", [*parameters_list])

    res = pvdeg.utilities.get_kinetics(name=None)

    assert res == desired_output


def test_read_material_bad():
    # no name case
    fpath = os.path.join(DATA_DIR, "O2permeation.json")
    with open(fpath) as f:
        data = json.load(f)

    material_list = data.keys()

    res = pvdeg.utilities._read_material(name=None)

    assert res == [*material_list]


def test_add_material():
    # new material parameters
    new_mat = {
        "alias": "test_material",
        "fickian": True,
        "Ead": 1,
        "Do": 1,
        "Eas": 1,
        "So": 1,
        "Eap": 1,
        "Po": 1,
    }

    # add new material to file
    pvdeg.utilities._add_material(name="tmat", **new_mat)

    # read updated file
    fpath = os.path.join(DATA_DIR, "O2permeation.json")
    with open(fpath) as f:
        data = json.load(f)

    # rename key, because we are comparing to original dictionary and func params do not align with the json keys
    new_mat["Fickian"] = new_mat["fickian"]
    new_mat.pop("fickian")

    # check
    assert data["tmat"] == new_mat

    # restore file to original state
    fpath = os.path.join(DATA_DIR, "O2permeation.json")
    with open(fpath) as f:
        data = json.load(f)
    data.pop("tmat")  # reset to default state

    with open(fpath, "w") as f:  # write default state
        json.dump(data, f, indent=4)


# this only works because we are not running on kestrel
def test_nrel_kestrel_check_bad():

    with pytest.raises(ConnectionError):
        pvdeg.utilities.nrel_kestrel_check()
