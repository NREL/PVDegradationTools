import os
import json
import pytest
import pandas as pd
import pvdeg 
from pvdeg import TEST_DATA_DIR

#Load weather data
weather_df = pd.read_pickle(os.path.join(TEST_DATA_DIR, 'weather_df_year.pkl'))
with open(os.path.join(TEST_DATA_DIR, 'meta.json')) as file:
    meta = json.load(file)

def test_calc_standoff():
    result_l1 = pvdeg.standards.calc_standoff(
        weather_df,
        meta,
        tilt=None,
        azimuth=180,
        sky_model='isotropic',
        temp_model='sapm',
        module_type='glass_polymer',
        level=1,
        x_0=6.1,
        wind_speed_factor=1.71)
    
    result_l2 = pvdeg.standards.calc_standoff(
        weather_df,
        meta,
        tilt=None,
        azimuth=180,
        sky_model='isotropic',
        temp_model='sapm',
        module_type='glass_polymer',
        level=2,
        x_0=6.1,
        wind_speed_factor=1.71)
    
    expected_result_l1 = {'x': 2.3835484140461736, 
                        'T98_0': 79.03006155479213, 
                        'T98_inf': 51.11191792458173}
    
    expected_result_l2 = {'x': -0.20832926385165268, 
                          'T98_0': 79.03006155479213, 
                          'T98_inf': 51.11191792458173}

    assert expected_result_l1 == pytest.approx(result_l1)
    assert expected_result_l2 == pytest.approx(result_l2)