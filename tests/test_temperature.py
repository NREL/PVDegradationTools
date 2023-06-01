import os
import pandas as pd
import pvdeg
from pvdeg import TEST_DATA_DIR

#Load weather data
WEATHER = pd.read_csv(os.path.join(TEST_DATA_DIR,'weather_day_pytest.csv'),
                      index_col=0, parse_dates=True)

#Load input dataframes
input_data = pd.read_csv(os.path.join(TEST_DATA_DIR, r'input_day_pytest.csv'),
                               index_col=0, parse_dates=True)
poa = [col for col in input_data.columns if 'poa' in col]
poa = input_data[poa]

# solpos = ['apparent_zenith','zenith','apparent_elevation','elevation','azimuth','equation_of_time']
# solpos = input_data[solpos]

#Load expected results
modtemp_expected = input_data['module_temp']

def test_module():
    result = pvdeg.temperature.module(
                WEATHER,
                poa,
                temp_model='sapm',
                conf='open_rack_glass_polymer',
                wind_speed_factor=1)

    pd.testing.assert_series_equal(result,
                                   modtemp_expected, 
                                   check_dtype=False, check_names=False)
