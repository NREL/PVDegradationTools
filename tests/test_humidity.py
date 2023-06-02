import os
import json
import pandas as pd
import pvdeg
from pytest import approx
from pvdeg import TEST_DATA_DIR

#Load weather data
WEATHER = pd.read_csv(os.path.join(TEST_DATA_DIR, 'weather_day_pytest.csv'),
                         index_col= 0, parse_dates=True)
with open(os.path.join(TEST_DATA_DIR, 'meta.json'),'r') as file:
    META = json.load(file)

#Load expected results
rh_expected = pd.read_csv(os.path.join(TEST_DATA_DIR, 'input_day_pytest.csv'),
                          index_col=0, parse_dates=True)
rh_cols = [col for col in rh_expected.columns if 'RH' in col]
rh_expected = rh_expected[rh_cols]


def test_calc_rel_humidity():
    '''
    test pvdeg.humidity.calc_rel_humidity
    
    Requires:
    ---------
    weather dataframe and meta dictionary
    '''
    result = pvdeg.humidity.calc_rel_humidity(WEATHER, META)
    pd.testing.assert_frame_equal(result, 
                                  rh_expected, 
                                  check_dtype=False)

def test_psat():
    '''
    test pvdeg.humidity.psat
    
    Requires:
    ---------
    weahter dataframe and meta dictionary
    '''
    psat_avg = pvdeg.humidity.psat(temp=WEATHER['air_temperature'])[1]
    assert psat_avg == approx(0.47607, abs=5e-5)
