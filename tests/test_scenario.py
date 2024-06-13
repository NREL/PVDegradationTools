from pvdeg.scenario import Scenario
from pvdeg.standards import standoff
from pvdeg import TEST_DATA_DIR
import json
import pandas as pd 
import pytest
import os

# problems with scenario creating directory in test directory?
EMAIL = 'user@mail.com'
API_KEY = 'DEMO_KEY'

def test_Scenario_add():
    a = Scenario(name='test')
    a.clean()
    a.restore_credentials(email=EMAIL, api_key=API_KEY)
    a.addLocation(lat_long=(40.63336, -73.99458))
    a.addModule(module_name='test-module')
    a.addJob(func=standoff,func_kwarg={'wind_factor' : 0.35})
    
    restored = Scenario.load_json(
        file_path=os.path.join(TEST_DATA_DIR, 'test-scenario.json')
    ) 
    
    # a.email = 'user@mail.com'
    # a.api_key = 'DEMO_KEY'
    a.path, restored.path = None, None
    a.file, restored.file = None, None
    
    assert a == restored

def test_Scenario_run():
    a = Scenario.load_json(file_path=os.path.join(TEST_DATA_DIR, 'test-scenario.json'), email=EMAIL, api_key=API_KEY)
    a.run()
    
    res_df = a.results['test-module']["GLUSE"]
    known_df = pd.DataFrame(
        {'x': {0: 0.0},
        'T98_0': {0: 68.80867997141961},
        'T98_inf': {0: 46.362946615593664}
        } )

    pd.testing.assert_frame_equal(res_df, known_df, check_dtype=False)


# geospatial tests should only run if on hpc, ask martin about protocol. load meta csv and weather nc (for very small scenario?)

