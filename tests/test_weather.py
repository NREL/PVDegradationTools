import os
import pandas as pd
import pvdeg 
from rex import Outputs

try:
    os.chdir('tests')
except:
    pass

TESTDIR = os.path.dirname(__file__)  # this folder

FILES = {'tmy3': os.path.join('data','tmy3_pytest.csv'),
         'psm3': os.path.join('data','psm3_pytest.csv'),
         'epw' : os.path.join('data','epw_pytest.epw'),
         'h5'  : os.path.join('data','h5_pytest.h5')}

DSETS = ['air_temperature', 'albedo', 'dew_point', 'dhi', 'dni',
         'ghi', 'meta', 'relative_humidity', 'time_index', 'wind_speed']
META_KEYS = ['']

def test_get():
    '''
    Test with (lat,lon) and gid options
    '''
    #TODO: Test with AWS

    # #Test with lat, lon on NREL HPC
    # weather_db = 'NSRDB'
    # weather_id = (39.741931, -105.169891) #NREL
    # weather_arg = {'satellite' : 'GOES', 
    #                'names' : 2021, 
    #                'NREL_HPC' : True,
    #                'attributes' : ['air_temperature', 'wind_speed', 'dhi', 
    #                                'ghi', 'dni','relative_humidity']}
    
    # weather_df, meta = pvdeg.weather.load(weather_db, weather_id, **weather_arg)
 
    # assert isinstance(meta, dict)
    # assert isinstance(weather_df, pd.DataFrame)
    # assert len(weather_df) != 0

    # #Test with gid on NREL HPC
    # weather_id = 1933572  
    # weather_df, meta = pvdeg.weather.load(weather_db, weather_id, **weather_arg)
    pass

def test_read():
    '''
    test pvdeg.utilities.read_weather
    
    Requires:
    ---------
    WEATHERFILES dicitonary of all verifiable weather files and types
    '''
    for type, path in FILES.items():
        if type != 'h5':
            weather_df, meta = pvdeg.weather.read(file_in=path,
                                                    file_type=type)
            assert isinstance(meta, dict)
            assert isinstance(weather_df, pd.DataFrame)
            assert len(weather_df) != 0

def test_read_h5():
    pass

def test_get_NSRDB_fnames():
    pass

def test_get_NSRDB():
    '''
    Contained within get_weather()
    '''
    pass
