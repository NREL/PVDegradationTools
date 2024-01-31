import os
import json
import pytest
import pandas as pd
import pvdeg
from pvdeg import TEST_DATA_DIR

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
        module_type="glass_polymer",
        level=1,
        x_0=6.1,
        wind_speed_factor=1.71,
    )

    result_l2 = pvdeg.standards.standoff(
        WEATHER,
        META,
        tilt=None,
        azimuth=180,
        sky_model="isotropic",
        temp_model="sapm",
        module_type="glass_polymer",
        level=2,
        x_0=6.1,
        wind_speed_factor=1.71,
    )

    expected_result_l1 = {
        "x": 2.3835484140461736,
        "T98_0": 79.03006155479213,
        "T98_inf": 51.11191792458173,
    }

    df_expected_result_l1 = pd.DataFrame.from_dict(expected_result_l1, orient="index").T

    expected_result_l2 = {
        "x": -0.20832926385165268,
        "T98_0": 79.03006155479213,
        "T98_inf": 51.11191792458173,
    }

    df_expected_result_l2 = pd.DataFrame.from_dict(expected_result_l2, orient="index").T

    pd.testing.assert_frame_equal(result_l1, df_expected_result_l1)
    pd.testing.assert_frame_equal(result_l2, df_expected_result_l2)

    # assert expected_result_l1 == pytest.approx(result_l1)
    # assert expected_result_l2 == pytest.approx(result_l2, abs=1e-5)
