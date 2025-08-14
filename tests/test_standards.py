"""Using pytest to create unit tests for pvdeg.

to run unit tests, run pytest from the command line in the pvdeg directory to run
coverage tests, run py.test --cov-report term-missing --cov=pvdeg
"""

import os
import json
import pytest
import pandas as pd
import pvdeg
from pvdeg import TEST_DATA_DIR

"""
TODO: during conversion from pkl to csv, a few fields dropped from float64 to float32.

This appears
to have altered the outcome for L2 results by roughly 1e-5. Is it worth correcting?
More specifically, the difference is of order:
x:       1e-7
T98_0:   1e-6
T98_inf: 1e-7
"""

# Load weather data
WEATHER = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "weather_year_pytest.csv"),
    index_col=0,
    parse_dates=True,
)

with open(os.path.join(TEST_DATA_DIR, "meta.json"), "r") as file:
    META = json.load(file)


def test_standoff():
    result_l1 = pvdeg.standards.standoff(
        WEATHER,
        META,
        tilt=None,
        azimuth=180,
        sky_model="isotropic",
        temp_model="sapm",
        conf_0="insulated_back_glass_polymer",
        conf_inf="open_rack_glass_polymer",
        T98=70,
        x_0=6.1,
        wind_factor=0.33,
    )

    result_l2 = pvdeg.standards.standoff(
        WEATHER,
        META,
        tilt=None,
        azimuth=180,
        sky_model="isotropic",
        temp_model="sapm",
        conf_0="insulated_back_glass_polymer",
        conf_inf="open_rack_glass_polymer",
        T98=80,
        x_0=6.1,
        wind_factor=0.33,
    )

    expected_result_l1 = {
        "x": 2.7381790837131876,
        "T98_0": 79.074638,
        "T98_inf": 53.982905,
    }

    df_expected_result_l1 = pd.DataFrame.from_dict(expected_result_l1, orient="index").T

    expected_result_l2 = {"x": 0, "T98_0": 79.074638, "T98_inf": 53.982905}

    df_expected_result_l2 = pd.DataFrame.from_dict(expected_result_l2, orient="index").T

    pd.testing.assert_frame_equal(result_l1, df_expected_result_l1)
    pd.testing.assert_frame_equal(result_l2, df_expected_result_l2)


def test_eff_gap():
    weather_file = os.path.join(TEST_DATA_DIR, "xeff_test.csv")
    xeff_weather, xeff_meta = pvdeg.weather.read(weather_file, "csv")
    T_0, T_inf, xeff_poa = pvdeg.standards.eff_gap_parameters(
        weather_df=xeff_weather,
        meta=xeff_meta,
        sky_model="isotropic",
        temp_model="sapm",
        conf_0="insulated_back_glass_polymer",
        conf_inf="open_rack_glass_polymer",
        wind_factor=0.33,
    )
    eff_gap = pvdeg.standards.eff_gap(
        T_0,
        T_inf,
        xeff_weather["module_temperature"],
        xeff_weather["temp_air"],
        xeff_poa["poa_global"],
        x_0=6.5,
        poa_min=400,
        t_amb_min=0,
    )

    assert eff_gap == pytest.approx(3.6767284845789825)
    # assert expected_result_l1 == pytest.approx(result_l1)
    # assert expected_result_l2 == pytest.approx(result_l2, abs=1e-5)


def test_T98_Xmin():
    WEATHER_df, META = pvdeg.weather.read(
        os.path.join(TEST_DATA_DIR, "psm3_pytest.csv"), "csv"
    )
    standoff = pvdeg.standards.standoff(weather_df=WEATHER_df, meta=META)
    assert standoff.x[0] == pytest.approx(2.008636)
    assert standoff.T98_0[0] == pytest.approx(77.038644)
    assert standoff.T98_inf[0] == pytest.approx(50.561112)
    kwarg_x = dict(
        sky_model="isotropic",
        temp_model="sapm",
        conf_0="insulated_back_glass_polymer",
        conf_inf="open_rack_glass_polymer",
        T98=70,
        x_0=6.5,
        wind_factor=0.33,
    )
    x_azimuth_step = 45
    x_tilt_step = 45
    standoff_series = pvdeg.utilities.tilt_azimuth_scan(
        weather_df=WEATHER_df,
        meta=META,
        tilt_step=x_tilt_step,
        azimuth_step=x_azimuth_step,
        func=pvdeg.standards.standoff_x,
        **kwarg_x,
    )
    print(standoff_series)
    print(WEATHER_df)
    print(META)
    assert standoff_series[13, 2] == pytest.approx(1.92868166)
