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
rh_cols = [col for col in rh_expected.columns if "RH_" in col or "Ce_" in col]
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


def test_water_saturation_pressure_mean():
    """Test pvdeg.humidity.water_saturation_pressure.

    Requires:
    ---------
    weather dataframe and meta dictionary
    """
    water_saturation_pressure_avg = pvdeg.humidity.water_saturation_pressure(
        temp=WEATHER["temp_air"]
    )
    assert water_saturation_pressure_avg[1] == pytest.approx(0.47607, abs=5e-5)
    assert water_saturation_pressure_avg[0][0] == pytest.approx(0.469731, abs=5e-5)
    assert water_saturation_pressure_avg[0][1] == pytest.approx(0.465908, abs=5e-5)
    assert water_saturation_pressure_avg[0][2] == pytest.approx(0.462112, abs=5e-5)
    assert water_saturation_pressure_avg[0][3] == pytest.approx(0.462112, abs=5e-5)
    assert water_saturation_pressure_avg[0][4] == pytest.approx(0.458343, abs=5e-5)


def test_water_saturation_pressure_individual_points():
    """Test pvdeg.humidity.water_saturation_pressure.

    Requires:
    ---------
    weather dataframe and meta dictionary
    """
    water_saturation_pressure = pvdeg.humidity.water_saturation_pressure(
        temp=WEATHER["temp_air"], average=False
    )
    assert water_saturation_pressure[0] == pytest.approx(0.469731, abs=5e-5)
    assert water_saturation_pressure[1] == pytest.approx(0.465908, abs=5e-5)
    assert water_saturation_pressure[2] == pytest.approx(0.462112, abs=5e-5)
    assert water_saturation_pressure[3] == pytest.approx(0.462112, abs=5e-5)
    assert water_saturation_pressure[4] == pytest.approx(0.458343, abs=5e-5)


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
        wind_factor=0.33,
        backsheet_thickness=0.3,
        back_encap_thickness=0.46,
    )

    # Check approximate equality for all columns
    assert result.shape == rh_expected.shape
    for col in rh_expected.columns:
        np.testing.assert_allclose(
            result[col].values, rh_expected[col].values, rtol=1e-3, atol=1e-3
        )


def test_module_basic():
    """Test pvdeg.humidity.module with basic input and default parameters."""
    result = pvdeg.humidity.module(
        WEATHER,
        META,
        temp_model="sapm",
        conf="open_rack_glass_glass",
        wind_factor=0.33,
        encapsulant="W002",
        backsheet="W002",
        backsheet_thickness=0.3,
        back_encap_thickness=0.46,
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
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == WEATHER.shape[0]


def test_module_edge_cases():
    """Test pvdeg.humidity.module with edge case input (extreme weather values)."""
    weather_df = pd.DataFrame(
        {
            "relative_humidity": [0, 100, 50],
            "temp_air": [-20, 50, 25],
            "wind_speed": [0, 10, 5],
            "ghi": [0, 1000, 500],
            "dni": [0, 900, 400],
            "dhi": [0, 100, 100],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="H"),
    )
    meta = {"latitude": 40, "longitude": -105, "altitude": 1600}
    result = pvdeg.humidity.module(
        weather_df,
        meta,
        temp_model="sapm",
        conf="open_rack_glass_glass",
        wind_factor=0.33,
        backsheet_thickness=0.3,
        back_encap_thickness=0.46,
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 3
    assert result["RH_surface_outside"].tolist() == pytest.approx(
        [0.0, 89.991403, 43.299585], abs=1e-3
    )
    assert result["RH_front_encap"].tolist() == pytest.approx(
        [430.443613, 73.762802, 122.953936], abs=1e-3
    )
    assert result["RH_back_encap"].tolist() == pytest.approx(
        [74.572295, 12.779052, 21.301181], abs=1e-3
    )
    assert result["RH_backsheet"].tolist() == pytest.approx(
        [37.286147, 51.385227, 32.300383], abs=1e-3
    )
    assert result["Ce_back_encap"].tolist() == pytest.approx(
        [0.000021991545, 0.0000219915458, 0.0000219915458], abs=1e-10
    )


def test_backsheet():
    rh_ambient = pd.Series([40, 60, 80])
    temp_ambient = pd.Series([20, 25, 30])
    temp_module = pd.Series([25, 30, 35])
    result = pvdeg.humidity.backsheet(
        rh_ambient,
        temp_ambient,
        temp_module,
        backsheet_thickness=0.3,
        back_encap_thickness=0.46,
    )
    # Should return a pandas Series and have same length as input
    assert result.tolist() == pytest.approx([24.535486, 31.149815, 38.113095], abs=1e-5)


def test_dew_yield():
    """Test pvdeg.humidity.dew_yield with basic input."""
    # Example input values (replace with realistic ones if available)
    temp_air = pd.Series([12, 20, 18])
    dew_point = pd.Series([80, 85, 90])
    wind_speed = pd.Series([1, 2, 3])
    n = pd.Series([4, 5, 3])
    # Call dew_yield function
    result = pvdeg.humidity.dew_yield(
        elevation=1, dry_bulb=temp_air, dew_point=dew_point, wind_speed=wind_speed, n=n
    )
    # Check result type and shape
    assert result.tolist() == pytest.approx([0.332943, 0.316928, 0.358373], abs=1e-6)


def test_diffusivity_weighted_water_basic():
    """Test pvdeg.humidity.diffusivity_weighted_water with basic input."""
    rh_ambient = pd.Series([50, 55, 60])
    temp_ambient = pd.Series([20, 22, 24])
    temp_module = pd.Series([25, 27, 29])
    result = pvdeg.humidity.diffusivity_weighted_water(
        rh_ambient, temp_ambient, temp_module
    )
    assert result == pytest.approx(0.0009117307352906477, abs=1e-9)


def test_diffusivity_weighted_water_with_params():
    """Test pvdeg.humidity.diffusivity_weighted_water with explicit parameters."""
    rh_ambient = pd.Series([40, 45, 50])
    temp_ambient = pd.Series([15, 18, 21])
    temp_module = pd.Series([20, 23, 26])
    So = 1.8
    Eas = 16.7
    Ead = 38.1
    result = pvdeg.humidity.diffusivity_weighted_water(
        rh_ambient, temp_ambient, temp_module, So=So, Eas=Eas, Ead=Ead
    )
    assert result == pytest.approx(0.0006841420183438176, abs=1e-9)


def test_csat_basic():
    """Test pvdeg.humidity.csat with basic input."""
    temp_module = pd.Series([25, 30, 35])
    result = pvdeg.humidity.csat(temp_module)
    assert result.tolist() == pytest.approx([0.002128, 0.002378, 0.002648], abs=1e-6)


def test_csat_with_params():
    """Test pvdeg.humidity.csat with explicit parameters."""
    temp_module = pd.Series([20, 25, 30])
    So = 1.8
    Eas = 16.7
    result = pvdeg.humidity.csat(temp_module, So=So, Eas=Eas)
    assert result.tolist() == pytest.approx([0.001904, 0.002136, 0.002387], abs=1e-6)


def test_ceq():
    """Test pvdeg.humidity.ceq with basic input."""
    Csat = pd.Series([0.5, 0.6, 0.7])
    rh_SurfaceOutside = pd.Series([50, 60, 70])
    result = pvdeg.humidity.ceq(Csat, rh_SurfaceOutside)
    assert result.tolist() == pytest.approx([0.25, 0.36, 0.49], abs=1e-6)


# Unit tests for backsheet_from_encap


def test_backsheet_from_encap():
    rh_back_encap = pd.Series([40, 60, 80])
    rh_surface_outside = pd.Series([20, 40, 60])
    result = pvdeg.humidity.backsheet_from_encap(rh_back_encap, rh_surface_outside)
    expected = pd.Series([30, 50, 70])
    np.testing.assert_allclose(result, expected, atol=1e-8)


def test_back_encapsulant_water_concentration_missing_t():
    temp_module = pd.Series([25, 30, 35])
    rh_surface = pd.Series([50, 55, 60])
    with pytest.raises(ValueError, match="backsheet_thickness must be specified"):
        pvdeg.humidity.back_encapsulant_water_concentration(
            temp_module=temp_module,
            rh_surface=rh_surface,
            Po_b=None,
            Ea_p_b=None,
            backsheet_thickness=None,
            backsheet="W017",
            back_encap_thickness=0.46,
        )


def test_back_encapsulant_water_concentration_missing_back_encap_thickness():
    temp_module = pd.Series([25, 30, 35])
    rh_surface = pd.Series([50, 55, 60])
    with pytest.raises(ValueError, match="back_encap_thickness must be specified"):
        pvdeg.humidity.back_encapsulant_water_concentration(
            temp_module=temp_module,
            rh_surface=rh_surface,
            Po_b=None,
            Ea_p_b=None,
            back_encap_thickness=None,
            backsheet="W017",
            backsheet_thickness=0.3,
        )
