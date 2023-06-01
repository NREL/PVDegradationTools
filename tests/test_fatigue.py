import os
import pvdeg
import pytest
import pandas as pd
from pvdeg import TEST_DATA_DIR

INPUT = pd.read_csv(os.path.join(TEST_DATA_DIR,'weather_year_pytest.csv'),
                    index_col=0, parse_dates=True)

def test_solder_fatigue():
    # test solder fatique with default parameters
    # requires PSM3 weather file

    damage = pvdeg.Degradation.solder_fatigue(time_range=INPUT.index,
                                            temp_cell=INPUT['temp_cell'])
    assert damage == pytest.approx(14.25, abs=0.1)