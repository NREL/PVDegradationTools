"""
Using pytest to create unit tests for pvdeg

to run unit tests, run pytest from the command line in the pvdeg directory
to run coverage tests, run py.test --cov-report term-missing --cov=pvdeg
"""

import os
import json
import pytest
import pandas as pd
import pvdeg
from pvdeg import TEST_DATA_DIR, DATA_DIR, TEST_DIR

"""
TODO: during conversion from pkl to csv, a few fields dropped from float64 to float32. This appears
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
        "T98": 70,
    }

    df_expected_result_l1 = pd.DataFrame.from_dict(expected_result_l1, orient="index").T

    expected_result_l2 = {
        "x": 0,
        "T98_0": 79.074638,
        "T98_inf": 53.982905,
        "T98": 80,
    }

    df_expected_result_l2 = pd.DataFrame.from_dict(expected_result_l2, orient="index").T
    print(result_l1)
    print(result_l2)
    pd.testing.assert_frame_equal(result_l1, df_expected_result_l1)
    pd.testing.assert_frame_equal(result_l2, df_expected_result_l2)

    # assert expected_result_l1 == pytest.approx(result_l1)
    # assert expected_result_l2 == pytest.approx(result_l2, abs=1e-5)
