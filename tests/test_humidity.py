import os
import json
import pytest
import pandas as pd
import pvdeg 

#Load weather data
weather_df = pd.read_pickle(os.path.join(pvdeg.TEST_DATA_DIR, 'weather_df_day.pkl'))
with open(os.path.join(pvdeg.TEST_DATA_DIR, 'meta.json')) as file:
    meta = json.load(file)

#Load expected results
expected_rel_hum = pd.read_pickle(os.path.join(
    pvdeg.TEST_DATA_DIR, 'rel_hum_day.pkl'))

def test_calc_rel_humidity():
    '''
    test pvdeg.humidity.calc_rel_humidity
    
    Requires:
    ---------
    weather dataframe and meta dictionary
    '''
    result = pvdeg.humidity.calc_rel_humidity(weather_df, meta)
    pd.testing.assert_frame_equal(result, 
                                  expected_rel_hum, 
                                  check_dtype=False)