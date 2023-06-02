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

DSETS = ['temp_air', 'albedo', 'dew_point', 'dhi', 'dni',
         'ghi', 'meta', 'relative_humidity', 'time_index', 'wind_speed']


def test_gid_downsampling():
    pass

def test_write_gids():
    pass

def test_convert_tmy():
    '''
    Test pvdeg.utilites.convert_tmy

    Requires:
    ---------
    tmy3 or tmy-like .csv weather file (WEATHERFILES['tmy3'])
    '''
    pvdeg.utilities.convert_tmy(file_in=FILES['tmy3'], file_out=FILES['h5'])
    with Outputs(FILES['h5'],'r') as f:
        datasets = f.dsets
    assert datasets == DSETS
