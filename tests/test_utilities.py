import os
# import numpy as np
# import pytest
import pandas as pd
import PVDegradationTools as PVD
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

def test_get_weather():
    '''
    Test with (lat,lon) and gid options
    '''
    lat_lon = (39.742,-105.179)
    gid = ()
    pass

def test_get_NSRDB_fnames():
    pass

def test_get_NSRDB():
    '''
    Contained within get_weather()
    '''
    pass

def test_read_weather():
    '''
    test PVD.utilities.read_weather
    
    Requires:
    ---------
    WEATHERFILES dicitonary of all verifiable weather files and types
    '''
    for key in FILES[:-1]:
        df, meta = PVD.utilities.read_weather(file_in= FILES[key],
                                              file_type= key)
        assert isinstance(meta, dict)
        assert isinstance(df, pd.DataFrame)
        assert len(df) != 0

def test_gid_downsampling():
    pass

def test_write_gids():
    pass

def test_convert_tmy():
    '''
    Test PVD.utilites.convert_tmy

    Requires:
    ---------
    tmy3 or tmy-like .csv weather file (WEATHERFILES['tmy3'])
    '''
    PVD.utilities.convert_tmy(file_in=FILES['tmy3'], file_out=FILES['h5'])
    with Outputs(FILES['h5'],'r') as f:
        datasets = f.dsets
    assert datasets == DSETS

def test_get_poa_irradiance():
    pass

def test_get_module_temperature():
    pass