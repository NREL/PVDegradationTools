import os
import pytest
import pandas as pd
from pvdeg import fatigue, weather, TEST_DATA_DIR

PSM_FILE = os.path.join(TEST_DATA_DIR, "psm3_pytest.csv")
WEATHER, META = weather.read(PSM_FILE, "psm")


def test_solder_fatigue():
    # test solder fatique with default parameters
    # requires PSM3 weather file

    damage = fatigue.solder_fatigue(weather_df=WEATHER, meta=META)
    assert damage == pytest.approx(15.646, abs=0.005)
