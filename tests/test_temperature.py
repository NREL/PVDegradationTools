import os
import json
import pytest
import pandas as pd
import pvdeg 
from pvdeg import TEST_DATA_DIR

#Load weather data
weather_df = pd.read_pickle(os.path.join(TEST_DATA_DIR, 'weather_df_day.pkl'))
with open(os.path.join(TEST_DATA_DIR, 'meta.json')) as file:
    meta = json.load(file)

#Load input dataframes
solar_position = pd.read_pickle(os.path.join(
    TEST_DATA_DIR, 'solar_position_day.pkl'))

poa_irradiance = pd.read_pickle(os.path.join(
    TEST_DATA_DIR, 'poa_irradiance_day.pkl'))

#Load expected results
expected_module_temp = pd.read_pickle(os.path.join(
    TEST_DATA_DIR, 'module_temp_day.pkl'))

def test_module():
    result = pvdeg.temperature.module(
                weather_df, 
                poa_irradiance,
                temp_model='sapm', 
                conf='open_rack_glass_polymer',
                wind_speed_factor=1)
    
    pd.testing.assert_series_equal(result, 
                                   expected_module_temp, 
                                   check_dtype=False)


