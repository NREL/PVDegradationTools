"""Using pytest to create unit tests for pvdeg.

to run unit tests, run pytest from the command line in the pvdeg directory to run
coverage tests, run py.test --cov-report term-missing --cov=pvdeg
"""

import os
import json
import pandas as pd
import pvdeg
import numpy as np
import pytest
from pvdeg import TEST_DATA_DIR
import numpy as np

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


def test_relative_float():
    result = pvdeg.humidity.relative(40.0, 30.0)
    assert result == pytest.approx(57.45, abs=0.01)


def test_relative_series():
    temp = pd.Series([25.0, 30.0, 35.0])
    dew = pd.Series([15.0, 10.0, 5.0])
    result = pvdeg.humidity.relative(temp, dew)
    expected = pd.Series([53.83, 28.94, 15.51])
    np.testing.assert_allclose(result, expected, atol=0.01)


def test_relative_nan_combinations():
    test_cases = [
        (pd.Series([25.0, np.nan]), pd.Series([15.0, 10.0])),  # nan in temp series
        (pd.Series([25.0, 30.0]), pd.Series([15.0, np.nan])),  # nan in dew series
        (np.nan, 10.0),  # nan in temp float
        (25.0, np.nan),  # nan in dew float
    ]

    for temp, dew in test_cases:
        with pytest.warns(UserWarning, match="Input contains NaN values"):
            pvdeg.humidity.relative(temp, dew)


def test_module():
    """Test pvdeg.humidity.calc_rel_humidity.

    Requires:
    ---------
    weather dataframe and meta dictionary
    """
    result = pvdeg.humidity.module(
        WEATHER,
        META,
        temp_model="sapm",
        conf="open_rack_glass_glass",
        wind_factor=0,
    )
    pd.testing.assert_frame_equal(result, rh_expected, check_dtype=False)


def test_water_saturation_pressure():
    """Test pvdeg.humidity.water_saturation_pressure.

    Requires:
    ---------
    weather dataframe and meta dictionary
    """
    water_saturation_pressure_avg = pvdeg.humidity.water_saturation_pressure(
        temp=WEATHER["temp_air"])[1]
    assert water_saturation_pressure_avg == pytest.approx(0.47607, abs=5e-5)

def test_module_basic():
    """Test pvdeg.humidity.module with basic input and default parameters."""
    result = pvdeg.humidity.module(
        WEATHER,
        META,
        temp_model="sapm",
        conf="open_rack_glass_glass",
        wind_factor=0,
    )
    # Check output type and columns
    assert isinstance(result, pd.DataFrame)
    expected_cols = [
        "RH_surface_outside",
        "RH_front_encap",
        "RH_back_encap",
        "Ce_back_encap",
        "RH_backsheet",
    ]
    for col in expected_cols:
        assert col in result.columns
    # Check shape matches input
    assert result.shape[0] == WEATHER.shape[0]

def test_module_with_params():
    """Test pvdeg.humidity.module with explicit parameters."""
    result = pvdeg.humidity.module(
        WEATHER,
        META,
        tilt=30,
        azimuth=180,
        temp_model="sapm",
        conf="open_rack_glass_glass",
        wind_factor=0.33,
        Po_b=1e9,
        Ea_p_b=55.4,
        backsheet_thickness=0.3,
        So_e=1.8,
        Ea_s_e=16.7,
        Ea_d_e=38.1,
        back_encap_thickness=0.46,
        backsheet="W017",
        encapsulant="W001",
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == WEATHER.shape[0]

def test_module_edge_cases():
    """Test pvdeg.humidity.module with edge case input (extreme weather values)."""
    weather_df = pd.DataFrame({
        "relative_humidity": [0, 100, 50],
        "temp_air": [-20, 50, 25],
        "wind_speed": [0, 10, 5],
        "ghi": [0, 1000, 500],
        "dni": [0, 900, 400],
        "dhi": [0, 100, 100],
    }, index=pd.date_range("2020-01-01", periods=3, freq="H"))
    meta = {"latitude": 40, "longitude": -105, "altitude": 1600}
    result = pvdeg.humidity.module(
        weather_df,
        meta,
        temp_model="sapm",
        conf="open_rack_glass_glass",
        wind_factor=0,
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 3
def test_dew_yield_basic():
    """Test pvdeg.humidity.dew_yield with basic input."""
    # Example input values (replace with realistic ones if available)
    temp_air = pd.Series([10, 12, 14])
    rh = pd.Series([80, 85, 90])
    wind_speed = pd.Series([1, 2, 3])
    # Call dew_yield function
    result = pvdeg.humidity.dew_yield(temp_air=temp_air, rh=rh, wind_speed=wind_speed)
    # Check result type and shape
    assert isinstance(result, pd.Series)
    assert len(result) == 3


def test_dew_yield_zero_wind():
    """Test pvdeg.humidity.dew_yield with zero wind speed."""
    temp_air = pd.Series([15, 16, 17])
    rh = pd.Series([70, 75, 80])
    wind_speed = pd.Series([0, 0, 0])
    result = pvdeg.humidity.dew_yield(temp_air=temp_air, rh=rh, wind_speed=wind_speed)
    assert isinstance(result, pd.Series)
    assert all(result >= 0)


def test_dew_yield_high_humidity():
    """Test pvdeg.humidity.dew_yield with high humidity values."""
    temp_air = pd.Series([20, 22, 24])
    rh = pd.Series([95, 98, 100])
    wind_speed = pd.Series([2, 2, 2])
    result = pvdeg.humidity.dew_yield(temp_air=temp_air, rh=rh, wind_speed=wind_speed)
    assert isinstance(result, pd.Series)
    assert all(result > 0)


def test_diffusivity_weighted_water_basic():
    """Test pvdeg.humidity.diffusivity_weighted_water with basic input."""
    rh_ambient = pd.Series([50, 55, 60])
    temp_ambient = pd.Series([20, 22, 24])
    temp_module = pd.Series([25, 27, 29])
    result = pvdeg.humidity.diffusivity_weighted_water(rh_ambient, temp_ambient, temp_module)
    assert isinstance(result, float) or isinstance(result, np.floating)


def test_diffusivity_weighted_water_with_params():
    """Test pvdeg.humidity.diffusivity_weighted_water with explicit parameters."""
    rh_ambient = pd.Series([40, 45, 50])
    temp_ambient = pd.Series([15, 18, 21])
    temp_module = pd.Series([20, 23, 26])
    So = 1.8
    Eas = 16.7
    Ead = 38.1
    result = pvdeg.humidity.diffusivity_weighted_water(rh_ambient, temp_ambient, temp_module, So=So, Eas=Eas, Ead=Ead)
    assert isinstance(result, float) or isinstance(result, np.floating)


def test_diffusivity_weighted_water_edge_cases():
    """Test pvdeg.humidity.diffusivity_weighted_water with edge case input."""
    rh_ambient = pd.Series([0, 100, 50])
    temp_ambient = pd.Series([-10, 50, 25])
    temp_module = pd.Series([0, 100, 50])
    result = pvdeg.humidity.diffusivity_weighted_water(rh_ambient, temp_ambient, temp_module)
    assert isinstance(result, float) or isinstance(result, np.floating)

def test_front_encap_basic():
    """Test pvdeg.humidity.front_encap (RHfront_series) with basic input."""
    rh_ambient = pd.Series([50, 55, 60])
    temp_ambient = pd.Series([20, 22, 24])
    temp_module = pd.Series([25, 27, 29])
    result = pvdeg.humidity.front_encap(rh_ambient, temp_ambient, temp_module)
    assert isinstance(result, pd.Series)
    assert len(result) == 3

def test_front_encap_with_params():
    """Test pvdeg.humidity.front_encap with explicit parameters."""
    rh_ambient = pd.Series([40, 45, 50])
    temp_ambient = pd.Series([15, 18, 21])
    temp_module = pd.Series([20, 23, 26])
    So = 1.8
    Eas = 16.7
    Ead = 38.1
    result = pvdeg.humidity.front_encap(rh_ambient, temp_ambient, temp_module, So=So, Eas=Eas, Ead=Ead)
    assert isinstance(result, pd.Series)
    assert len(result) == 3

def test_front_encap_edge_cases():
    """Test pvdeg.humidity.front_encap with edge case input."""
    rh_ambient = pd.Series([0, 100, 50])
    temp_ambient = pd.Series([-10, 50, 25])
    temp_module = pd.Series([0, 100, 50])
    result = pvdeg.humidity.front_encap(rh_ambient, temp_ambient, temp_module)
    assert isinstance(result, pd.Series)
    assert len(result) == 3

def test_csat_basic():
    """Test pvdeg.humidity.csat with basic input."""
    temp_module = pd.Series([25, 30, 35])
    result = pvdeg.humidity.csat(temp_module)
    assert isinstance(result, pd.Series)
    assert len(result) == 3

def test_csat_with_params():
    """Test pvdeg.humidity.csat with explicit parameters."""
    temp_module = pd.Series([20, 25, 30])
    So = 1.8
    Eas = 16.7
    result = pvdeg.humidity.csat(temp_module, So=So, Eas=Eas)
    assert isinstance(result, pd.Series)
    assert len(result) == 3

def test_csat_edge_cases():
    """Test pvdeg.humidity.csat with edge case input."""
    temp_module = pd.Series([-10, 0, 100])
    result = pvdeg.humidity.csat(temp_module)
    assert isinstance(result, pd.Series)
    assert len(result) == 3

def test_ceq_basic():
    """Test pvdeg.humidity.ceq with basic input."""
    Csat = pd.Series([0.5, 0.6, 0.7])
    rh_SurfaceOutside = pd.Series([50, 60, 70])
    result = pvdeg.humidity.ceq(Csat, rh_SurfaceOutside)
    assert isinstance(result, pd.Series)
    assert len(result) == 3
    # Check calculation
    expected = Csat * (rh_SurfaceOutside / 100)
    pd.testing.assert_series_equal(result, expected)

def test_ceq_edge_cases():
    """Test pvdeg.humidity.ceq with edge case input."""
    Csat = pd.Series([0, 1, 100])
    rh_SurfaceOutside = pd.Series([0, 100, 50])
    result = pvdeg.humidity.ceq(Csat, rh_SurfaceOutside)
    assert isinstance(result, pd.Series)


def test_backsheet_basic():
    """Test pvdeg.humidity.backsheet with basic input."""
    rh_ambient = pd.Series([40, 50, 60])
    temp_ambient = pd.Series([20, 22, 24])
    temp_module = pd.Series([25, 27, 29])
    result = pvdeg.humidity.backsheet(rh_ambient, temp_ambient, temp_module)
    assert isinstance(result, pd.Series)
    assert len(result) == 3


def test_backsheet_edge_cases():
    """Test pvdeg.humidity.backsheet with edge case input."""
    rh_ambient = pd.Series([0, 100, 50])
    temp_ambient = pd.Series([-10, 50, 25])
    temp_module = pd.Series([0, 100, 50])
    result = pvdeg.humidity.backsheet(rh_ambient, temp_ambient, temp_module)
    assert isinstance(result, pd.Series)
    assert len(result) == 3


def test_backsheet_from_encap_basic():
    """Test pvdeg.humidity.backsheet_from_encap with basic input."""
    rh_back_encap = pd.Series([40, 50, 60])
    rh_surface_outside = pd.Series([20, 30, 40])
    result = pvdeg.humidity.backsheet_from_encap(rh_back_encap, rh_surface_outside)
    assert isinstance(result, pd.Series)
    assert len(result) == 3
    expected = (rh_back_encap + rh_surface_outside) / 2
    pd.testing.assert_series_equal(result, expected)


def test_backsheet_from_encap_edge_cases():
    """Test pvdeg.humidity.backsheet_from_encap with edge case input."""
    rh_back_encap = pd.Series([0, 100, 50])
    rh_surface_outside = pd.Series([100, 0, 50])
    result = pvdeg.humidity.backsheet_from_encap(rh_back_encap, rh_surface_outside)
    assert isinstance(result, pd.Series)
    assert len(result) == 3
    expected = (rh_back_encap + rh_surface_outside) / 2
    pd.testing.assert_series_equal(result, expected)
    assert len(result) == 3
    expected = Csat * (rh_SurfaceOutside / 100)
    pd.testing.assert_series_equal(result, expected)

def test_Ce_basic():
    """Test pvdeg.humidity.Ce with basic input."""
    temp_module = pd.Series([25, 30, 35])
    rh_surface = pd.Series([50, 60, 70])
    result = pvdeg.humidity.Ce(temp_module, rh_surface)
    assert isinstance(result, pd.Series)
    assert len(result) == 3

def test_Ce_with_params():
    """Test pvdeg.humidity.Ce with explicit parameters and output concentration."""
    temp_module = pd.Series([20, 25, 30])
    rh_surface = pd.Series([40, 45, 50])
    result = pvdeg.humidity.Ce(temp_module, rh_surface, output="Ce")
    assert isinstance(result, pd.Series)
    assert len(result) == 3

def test_Ce_edge_cases():
    """Test pvdeg.humidity.Ce with edge case input."""
    temp_module = pd.Series([-10, 0, 100])
    rh_surface = pd.Series([0, 100, 50])
    result = pvdeg.humidity.Ce(temp_module, rh_surface)
    assert isinstance(result, pd.Series)
    assert len(result) == 3






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

    rh_front_encap = pvdeg.StressFactors.rh_front_encap(
        rh_ambient=PSM['Relative Humidity'],
        temp_ambient=PSM['Temperature'],
        temp_module=PSM['temp_module'])
    assert rh_front_encap.__len__() == PSM.__len__()
    assert rh_front_encap.iloc[17] == pytest.approx(50.289, abs=.001)

def test_rh_back_encap():
    # test calculation for RH of module backside encapsulant
    # requires PSM3 weather file

    rh_back_encap = pvdeg.StressFactors.rh_back_encap(
        rh_ambient=PSM['Relative Humidity'],
        temp_ambient=PSM['Temperature'],
        temp_module=PSM['temp_module'])
    assert rh_back_encap.__len__() == PSM.__len__()
    assert rh_back_encap[17] == pytest.approx(80.4576, abs=0.001)

def test_rh_backsheet_from_encap():
    # test the calculation for backsheet relative humidity
    # requires PSM3 weather file

    rh_back_encap = pvdeg.StressFactors.rh_back_encap(
        rh_ambient=PSM['Relative Humidity'],
        temp_ambient=PSM['Temperature'],
        temp_module=PSM['temp_module'])
    rh_surface = pvdeg.StressFactors.rh_surface_outside(
        rh_ambient=PSM['Relative Humidity'],
        temp_ambient=PSM['Temperature'],
        temp_module=PSM['temp_module'])
    rh_backsheet = pvdeg.StressFactors.rh_backsheet_from_encap(
        rh_back_encap=rh_back_encap,
        rh_surface_outside=rh_surface)
    assert rh_backsheet.__len__() == PSM.__len__()
    assert rh_backsheet[17] == pytest.approx(81.2238, abs=0.001)

def test_rh_backsheet():
    # test calculation for backsheet relative humidity directly from weather variables
    # requires PSM3 weather file

    rh_backsheet = pvdeg.StressFactors.rh_backsheet(rh_ambient=PSM['Relative Humidity'],
                                                    temp_ambient=PSM['Temperature'],
                                                    temp_module=PSM['temp_module'])
    assert rh_backsheet.__len__() == PSM.__len__()
    assert rh_backsheet[17] == pytest.approx(81.2238, abs=0.001)
"""
