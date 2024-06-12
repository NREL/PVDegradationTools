from pvdeg.scenario import Scenario
from pvdeg.standards import standoff
from pvdeg import TEST_DATA_DIR
import json
import pandas as pd 
import pytest
import os

# problems with scenario creating directory in test directory?
# should we have a dedicated testing api-key, docs key is limited to 50 requests per ip address per day

EMAIL = 'user@mail.com'
API_KEY = 'DEMO_KEY'

def test_Scenario_add():
    a = Scenario(name='test')
    a.restore_credentials(email=EMAIL, api_key=API_KEY)
    a.addLocation(lat_long=(40.63336, -73.99458))
    a.addModule(module_name='test-module')
    a.addJob(func=standoff,func_params={'wind_factor' : 0.35})
    a.file = None

    # restore scenario from file
    restored = Scenario() 
    restored.load_json(file_path=os.path.join(TEST_DATA_DIR, 'test-scenario.json')) # this will faill when load_json is moved to a classmethod
    restored.file = None 

    def first_odict_value(odict):
        """Get first value from an odict"""
        return next(iter(odict.values()))

    # test all items in belonging to the scenario instances 
    for key, value in a.__dict__.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(value, restored.__dict__[key])
        elif isinstance(value, (float, int)):
            pytest.approx(value, restored.__dict__[key])
        else:
            if key != "path":
                if key != "pipeline":
                    assert value == restored.__dict__[key]
                else:
                    pd.testing(first_odict_value(a.pipeline), first_odict_value(restored.pipeline)) # try this again later

def test_Scenario_run():
    # load from saved json
    a = Scenario(name='test-running')
    a.load_json(file_path=os.path.join(TEST_DATA_DIR, 'test-scenario.json'))

    # run saved json
    a.run()

    # compare results to some saved results
    res_df = a.results['test-module']["GLUSE"]
    known_df = pd.DataFrame(
        {'x': {0: 0.0},
        'T98_0': {0: 68.80867997141961},
        'T98_inf': {0: 46.362946615593664}
        } )

    pd.testing.assert_frame_equal(res_df, known_df, check_dtype=False)

# geospatial tests should only run if on hpc, ask martin about protocol. load meta csv and weather nc (for very small scenario?)
