"""Using pytest to create unit tests for pvdeg.

to run unit tests, run pytest from the command line in the pvdeg directory to run
coverage tests, run py.test --cov-report term-missing --cov=pvdeg
"""

import os
import pytest
from pvdeg import fatigue, weather, TEST_DATA_DIR

PSM_FILE = os.path.join(TEST_DATA_DIR, "psm3_pytest.csv")
WEATHER, META = weather.read(PSM_FILE, "psm")


def test_solder_fatigue():
    # test solder fatique with default parameters
    # requires PSM3 weather file

    damage = fatigue.solder_fatigue(weather_df=WEATHER, meta=META, wind_factor=1.0)
    assert damage == pytest.approx(15.646, abs=0.005)
