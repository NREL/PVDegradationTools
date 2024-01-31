import os
import json
import pandas as pd
import pvdeg
from pytest import approx
from pvdeg import TEST_DATA_DIR

# Load weather data
WEATHER = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "weather_day_pytest.csv"), index_col=0, parse_dates=True
)
with open(os.path.join(TEST_DATA_DIR, "meta.json"), "r") as file:
    META = json.load(file)

# Load expected results
rh_expected = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "input_day_pytest.csv"), index_col=0, parse_dates=True
)
rh_cols = [col for col in rh_expected.columns if "RH" in col]
rh_expected = rh_expected[rh_cols]


def test_module():
    """
    test pvdeg.humidity.calc_rel_humidity

    Requires:
    ---------
    weather dataframe and meta dictionary
    """
    result = pvdeg.humidity.module(WEATHER, META)
    pd.testing.assert_frame_equal(result, rh_expected, check_dtype=False)


def test_psat():
    """
    test pvdeg.humidity.psat

    Requires:
    ---------
    weahter dataframe and meta dictionary
    """
    psat_avg = pvdeg.humidity.psat(temp=WEATHER["temp_air"])[1]
    assert psat_avg == approx(0.47607, abs=5e-5)


"""
Legacy Tests
TODO do we need to individually test RH functions?

def test_rh_surface_outside():
    # test calculation for the RH just outside a module surface
    # requires PSM3 weather file

    rh_surface = pvdeg.StressFactors.rh_surface_outside(50, 35, 55)
    assert rh_surface == pytest.approx(18.247746468009066, abs=0.0000001)

def test_rh_front_encap():
    # test calculation for RH of module fronside encapsulant
    # requires PSM3 weather file

    rh_front_encap = pvdeg.StressFactors.rh_front_encap(rh_ambient=PSM['Relative Humidity'],
                                                    temp_ambient=PSM['Temperature'],
                                                    temp_module=PSM['temp_module'])
    assert rh_front_encap.__len__() == PSM.__len__()
    assert rh_front_encap.iloc[17] == pytest.approx(50.289, abs=.001)

def test_rh_back_encap():
    # test calculation for RH of module backside encapsulant
    # requires PSM3 weather file

    rh_back_encap = pvdeg.StressFactors.rh_back_encap(rh_ambient=PSM['Relative Humidity'],
                                                    temp_ambient=PSM['Temperature'],
                                                    temp_module=PSM['temp_module'])
    assert rh_back_encap.__len__() == PSM.__len__()
    assert rh_back_encap[17] == pytest.approx(80.4576, abs=0.001)

def test_rh_backsheet_from_encap():
    # test the calculation for backsheet relative humidity
    # requires PSM3 weather file

    rh_back_encap = pvdeg.StressFactors.rh_back_encap(rh_ambient=PSM['Relative Humidity'],
                                                    temp_ambient=PSM['Temperature'],
                                                    temp_module=PSM['temp_module'])
    rh_surface = pvdeg.StressFactors.rh_surface_outside(rh_ambient=PSM['Relative Humidity'],
                                                        temp_ambient=PSM['Temperature'],
                                                        temp_module=PSM['temp_module'])
    rh_backsheet = pvdeg.StressFactors.rh_backsheet_from_encap(rh_back_encap=rh_back_encap,
                                                            rh_surface_outside=rh_surface)
    assert rh_backsheet.__len__() == PSM.__len__()
    assert rh_backsheet[17] == pytest.approx(81.2238, abs=0.001)

def test_rh_backsheet():
    # test the calculation for backsheet relative humidity directly from weather variables
    # requires PSM3 weather file

    rh_backsheet = pvdeg.StressFactors.rh_backsheet(rh_ambient=PSM['Relative Humidity'],
                                                    temp_ambient=PSM['Temperature'],
                                                    temp_module=PSM['temp_module'])
    assert rh_backsheet.__len__() == PSM.__len__()
    assert rh_backsheet[17] == pytest.approx(81.2238, abs=0.001)
"""
