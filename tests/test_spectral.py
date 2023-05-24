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

#Load expected results
expected_solar_position = pd.read_pickle(os.path.join(
    TEST_DATA_DIR, 'solar_position_day.pkl'))

expected_poa_irradiance = pd.read_pickle(os.path.join(
    TEST_DATA_DIR, 'poa_irradiance_day.pkl'))

def test_solar_position():
    '''
    test pvdeg.spectral.solar_position
    
    Requires:
    ---------
    weather dataframe and meta dictionary
    '''
    result = pvdeg.spectral.solar_position(weather_df, meta)
    pd.testing.assert_frame_equal(result, 
                                  expected_solar_position, 
                                  check_dtype=False)

def test_poa_irradiance():
    '''
    test pvdeg.spectral.poa_irradiance
    
    Requires:
    ---------
    weather dataframe, meta dictionary, and solar_position dataframe
    '''
    result = pvdeg.spectral.poa_irradiance(weather_df, 
                                         meta,
                                         expected_solar_position,
                                         tilt=None, 
                                         azimuth=180, 
                                         sky_model='isotropic')
    
    pd.testing.assert_frame_equal(result, 
                                  expected_poa_irradiance, 
                                  check_dtype=False)