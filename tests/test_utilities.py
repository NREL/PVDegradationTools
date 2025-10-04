"""Using pytest to create unit tests for pvdeg.

to run unit tests, run pytest from the command line in the pvdeg directory to run
coverage tests, run py.test --cov-report term-missing --cov=pvdeg
"""

import os
import json
import pandas as pd
import numpy as np
import pvdeg
from rex import Outputs
import shutil
import io
import sys

import pytest
from pvdeg import TEST_DATA_DIR, DATA_DIR
from collections import OrderedDict

FILES = {
    "tmy3": os.path.join(TEST_DATA_DIR, "tmy3_pytest.csv"),
    "psm3": os.path.join(TEST_DATA_DIR, "psm3_pytest.csv"),
    "epw": os.path.join(TEST_DATA_DIR, "epw_pytest.epw"),
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


def load_json(filename):
    fpath = os.path.join(DATA_DIR, filename)
    with open(fpath, encoding="utf-8") as f:
        return json.load(f)


def test_read_material_basic():
    """Test pvdeg.utilities.read_material returns correct dict for a known key."""
    data = load_json("O2permeation.json")
    known_key = next(iter(data.keys()))
    expected = data[known_key]
    result = pvdeg.utilities.read_material(pvdeg_file="O2permeation", key=known_key)
    assert result == expected


def test_read_material_parameters():
    """Test pvdeg.utilities.read_material_property returns only requested parameters."""
    data = load_json("O2permeation.json")
    known_key = next(iter(data.keys()))
    params = ["name", "alias"]
    expected = {k: data[known_key].get(k, None) for k in params}
    result = pvdeg.utilities.read_material_property(
        pvdeg_file="O2permeation", key=known_key, parameters=params
    )
    assert result == expected


def test_search_json_name():
    """Test pvdeg.utilities.search_json with name lookup."""
    # Find a known name in H2Opermeation.json
    data = load_json("H2Opermeation.json")
    # Use the first entry's 'name' or 'alias' field
    known_key = next(iter(data.keys()))
    name = data[known_key].get("name", None)
    if name:
        result = pvdeg.utilities.search_json(
            pvdeg_file="H2Opermeation", name_or_alias=name
        )
        assert result == known_key


def test_search_json_alias():
    """Test pvdeg.utilities.search_json with alias lookup."""
    data = load_json("H2Opermeation.json")
    known_key = next(iter(data.keys()))
    alias = data[known_key].get("alias", None)
    if alias:
        result = pvdeg.utilities.search_json(
            pvdeg_file="H2Opermeation", name_or_alias=alias
        )
        assert result == known_key


def test_search_json_fp():
    """Test pvdeg.utilities.search_json with explicit file path."""
    filename = "H2Opermeation.json"
    data = load_json(filename)
    fpath = os.path.join(DATA_DIR, filename)
    known_key = next(iter(data.keys()))
    alias = data[known_key].get("alias", None)

    if alias:
        result = pvdeg.utilities.search_json(fp=fpath, name_or_alias=alias)
        assert result == known_key


def test_display_json_basic():
    """Test pvdeg.utilities.display_json prints JSON for a known file."""
    # Capture stdout
    captured_output = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = captured_output
    try:
        pvdeg.utilities.display_json(pvdeg_file="H2Opermeation")
    finally:
        sys.stdout = sys_stdout
    output = captured_output.getvalue()
    # Check that output contains expected keys from the file
    data = load_json("H2Opermeation.json")
    for key in list(data.keys())[:2]:  # Check first two keys for brevity
        assert key in output


def test_display_json_fp():
    """Test pvdeg.utilities.display_json with explicit file path."""
    fpath = os.path.join(DATA_DIR, "H2Opermeation.json")
    captured_output = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = captured_output
    try:
        pvdeg.utilities.display_json(fp=fpath)
    finally:
        sys.stdout = sys_stdout
    output = captured_output.getvalue()
    with open(fpath) as f:
        data = json.load(f)
    for key in list(data.keys())[:2]:
        assert key in output


def test_convert_tmy(tmp_path):
    """Test pvdeg.utilites.convert_tmy.

    Requires:
    ---------
    tmy3 or tmy-like .csv weather file (WEATHERFILES['tmy3'])
    """
    fp_h5 = os.path.join(tmp_path, "h5_pytest.h5")
    pvdeg.utilities.convert_tmy(file_in=FILES["tmy3"], file_out=fp_h5)
    with Outputs(fp_h5, "r") as f:
        datasets = f.dsets
    assert datasets.sort() == DSETS.sort()


def test_get_kinetics():
    """
    Test pvdeg.utilities.get_kinetics

    Requires:
    --------
    data : dict, from kinetic_parameters.json
    """
    data = load_json("kinetic_parameters.json")
    result = pvdeg.utilities.get_kinetics("repins")
    assert data["repins"] == result


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
    data = load_json("kinetic_parameters.json")
    parameters_list = data.keys()

    desired_output = ("Choose a set of kinetic parameters:", [*parameters_list])

    res = pvdeg.utilities.get_kinetics(name=None)

    assert res == desired_output


def test_add_material(tmp_path):
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

    # Ensure the test file exists
    src_file = os.path.join(DATA_DIR, "O2permeation.json")
    test_file = os.path.join(tmp_path, "O2permeation.json")
    shutil.copy(src_file, test_file)

    # add new material to file
    pvdeg.utilities._add_material(
        name="tmat", fp=tmp_path, fname="O2permeation.json", **new_mat
    )

    # read updated file
    with open(test_file) as f:
        data = json.load(f)

    # rename key, because we are comparing to original dictionary and func params do not
    # align with the json keys
    new_mat["Fickian"] = new_mat["fickian"]
    new_mat.pop("fickian")

    # check
    assert data["tmat"] == new_mat


# this only works because we are not running on kestrel
def test_nrel_kestrel_check_bad():
    with pytest.raises(ConnectionError):
        pvdeg.utilities.nrel_kestrel_check()


# NEW MATERIAL UTIL FUNCTIONS
# These tests will likely fail if the associated materials are changed
# ===========================
def test_read_material_special():
    template_material = pvdeg.utilities.read_material(
        pvdeg_file="AApermeation", key="AA000"
    )

    assert len(template_material) == 1
    assert "comment" in template_material


def test_read_material_normal():
    res = {
        "name": "ST504",
        "alias": "PET1",
        "contributor": "Michael Kempe",
        "source": "unpublished measurements",
        "Fickian": True,
        "Ead": {
            "name": "Diffusivity Activation Energy",
            "units": "kJ/mol",
            "value": 47.603,
        },
        "Do": {
            "name": "Diffusivity Prefactor",
            "units": "cm²/s",
            "value": 0.554153,
        },
        "Eas": {
            "name": "Solubility Activation Energy",
            "units": "kJ/mol",
            "value": -11.5918,
        },
        "So": {
            "name": "Solubility Prefactor",
            "units": "g/cm³/atm",
            "value": 9.554366e-07,
        },
        "Eap": {
            "name": "Permeability Activation Energy",
            "units": "kJ/mol",
            "value": 34.2011,
        },
        "Po": {
            "name": "Permeability Prefactor",
            "units": "g·mm/m²/day/atm",
            "value": 2128.8937,
        },
    }

    template_material = pvdeg.utilities.read_material(
        pvdeg_file="O2permeation", key="OX002"
    )

    assert template_material == res


def test_read_material_property_select_params():
    res = {
        "name": "ST504",
        "Fickian": True,
    }

    template_material = pvdeg.utilities.read_material_property(
        pvdeg_file="O2permeation", key="OX002", parameters=["name", "Fickian"]
    )

    assert template_material == res


def test_read_material_property_missing_params():
    res = {
        "namenotindict1": None,
        "namenotindict2": None,
    }

    template_material = pvdeg.utilities.read_material_property(
        pvdeg_file="O2permeation",
        key="OX002",
        parameters=["namenotindict1", "namenotindict2"],
    )

    assert template_material == res


def test_read_material_fp_override():
    res = {
        "name": "ST504",
        "alias": "PET1",
        "contributor": "Michael Kempe",
        "source": "unpublished measurements",
        "Fickian": True,
        "Ead": {
            "name": "Diffusivity Activation Energy",
            "units": "kJ/mol",
            "value": 47.603,
        },
        "Do": {
            "name": "Diffusivity Prefactor",
            "units": "cm²/s",
            "value": 0.554153,
        },
        "Eas": {
            "name": "Solubility Activation Energy",
            "units": "kJ/mol",
            "value": -11.5918,
        },
        "So": {
            "name": "Solubility Prefactor",
            "units": "g/cm³/atm",
            "value": 9.554366e-07,
        },
        "Eap": {
            "name": "Permeability Activation Energy",
            "units": "kJ/mol",
            "value": 34.2011,
        },
        "Po": {
            "name": "Permeability Prefactor",
            "units": "g·mm/m²/day/atm",
            "value": 2128.8937,
        },
    }

    template_material = pvdeg.utilities.read_material(
        pvdeg_file="O2permeation",
        fp=os.path.join(DATA_DIR, "AApermeation.json"),
        key="OX002",
    )

    assert template_material == res


def test_search_json():
    name_res = pvdeg.utilities.search_json(
        pvdeg_file="H2Opermeation", name_or_alias="Ethylene Vinyl Acetate"
    )
    alias_res = pvdeg.utilities.search_json(
        pvdeg_file="H2Opermeation", name_or_alias="EVA"
    )

    assert name_res == "W001"
    assert alias_res == "W001"


def test_meta_as_dict():
    rec = np.array([(1, 2.0, "a")], dtype=[("x", "i4"), ("y", "f4"), ("z", "U1")])[0]
    d = pvdeg.utilities.meta_as_dict(rec)
    assert d == {"x": 1, "y": 2.0, "z": "a"}


def test_get_state_bbox():
    bbox = pvdeg.utilities.get_state_bbox("CO")
    assert isinstance(bbox, np.ndarray)
    assert bbox.shape == (2, 2)


def test_new_id():
    d = OrderedDict({"ABCDE": 1, "FGHIJ": 2})
    new = pvdeg.utilities.new_id(d)
    assert isinstance(new, str)
    assert len(new) == 5
    assert new not in d


def test_strip_normalize_tmy():
    idx = pd.date_range("2023-01-01 00:00", periods=24, freq="H", tz="UTC")
    df = pd.DataFrame({"ghi": range(24)}, index=idx)
    start = idx[5].to_pydatetime()
    end = idx[10].to_pydatetime()
    sub = pvdeg.utilities.strip_normalize_tmy(df, start, end)
    assert isinstance(sub, pd.DataFrame)
    assert sub.index[0].hour == 5
    assert sub.index[-1].hour == 10


def test_tilt_azimuth_scan():
    def dummy_func(tilt, azimuth, **kwarg):
        return tilt + azimuth


def test_tilt_azimuth_scan_basic():
    """Test pvdeg.utilities.tilt_azimuth_scan with a dummy function."""

    def dummy_func(tilt, azimuth, **kwarg):
        return tilt + azimuth

    arr = pvdeg.utilities.tilt_azimuth_scan(
        weather_df=None, meta=None, tilt_step=45, azimuth_step=90, func=dummy_func
    )
    # Check output type and shape
    assert isinstance(arr, np.ndarray)
    assert arr.shape[1] == 3
    # Check that the dummy function is applied correctly
    for row in arr:
        tilt, azimuth, value = row
        assert value == tilt + azimuth
    arr = pvdeg.utilities.tilt_azimuth_scan(
        weather_df=None, meta=None, tilt_step=45, azimuth_step=90, func=dummy_func
    )
    assert isinstance(arr, np.ndarray)
    assert arr.shape[1] == 3
    assert arr.shape[0] == 15


# def test_search_json_bad():
#     ...
