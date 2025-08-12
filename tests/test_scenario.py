from pvdeg.scenario import Scenario
from pvdeg.standards import standoff
from pvdeg import TEST_DATA_DIR
from pvdeg import weather
import pandas as pd
import pytest
import os
import numpy as np

# problems with scenario creating directory in test directory?
EMAIL = "user@mail.com"
API_KEY = "DEMO_KEY"


def monkeypatch_addLocation(self, *args, **kwargs) -> None:
    """Mocker function to be monkey patched at runtime for Scenario.addLocation to avoid.

    psm3 api calls and use local weather files instead.
    """

    PSM_FILE = os.path.join(TEST_DATA_DIR, r"psm3_pytest.csv")
    weather_df, meta = weather.read(PSM_FILE, "psm")

    self.email, self.api_key = None, None

    self.lat_long = (-999, -999)
    self.weather_data = weather_df
    self.meta_data = meta
    self.gids = np.asanyarray([1245357])


def test_Scenario_add(monkeypatch):
    ### monkey patch to bypass psm3 api calls in addLocation ###
    monkeypatch.setattr(
        target=Scenario, name="addLocation", value=monkeypatch_addLocation
    )

    a = Scenario(name="test")

    EMAIL = ("placeholder@email.xxx",)
    API_KEY = "fake_key"

    a.clean()
    a.restore_credentials(email=EMAIL, api_key=API_KEY)
    a.addLocation(lat_long=(40.63336, -73.99458))
    a.addModule(module_name="test-module")
    a.addJob(func=standoff, func_kwarg={"wind_factor": 0.35})

    restored = Scenario.load_json(
        file_path=os.path.join(TEST_DATA_DIR, "test-scenario.json")
    )

    a.path, restored.path = None, None
    a.file, restored.file = None, None

    assert a == restored


def test_Scenario_run(monkeypatch):
    ### monkey patch to bypass psm3 api calls in addLocation called by load_json ###
    monkeypatch.setattr(
        target=Scenario, name="addLocation", value=monkeypatch_addLocation
    )

    a = Scenario.load_json(
        file_path=os.path.join(TEST_DATA_DIR, "test-scenario.json"),
        email=EMAIL,
        api_key=API_KEY,
    )
    a.run()

    res_df = a.results["test-module"]["GLUSE"]
    known_df = pd.DataFrame(
        {
            "x": {0: 2.008636},
            "T98_0": {0: 77.038644},
            "T98_inf": {0: 50.561112},
        }
    )

    pd.testing.assert_frame_equal(res_df, known_df, check_dtype=False)


# def test_clean():
#     a = Scenario(name='clean-a')
#     a.file = 'non-existent-file.json'
#     with pytest.raises(FileNotFoundError):
#         a.clean()

#     b = Scenario(name='clean-b')
#     with pytest.raises(ValueError):
#         b.clean()


def test_addLocation_pvgis():
    a = Scenario(name="location-test")
    with pytest.raises(ValueError):
        a.addLocation((40.63336, -73.99458), weather_db="PSM3")  # no api key


def test_addModule_badmat():
    a = Scenario.load_json(
        file_path=os.path.join(TEST_DATA_DIR, "test-scenario.json"),
        email=EMAIL,
        api_key=API_KEY,
    )

    with pytest.warns(UserWarning, match="Failed to add module"):
        a.addModule(module_name="fail", material="fake-material")


def test_addModule_existingmod():
    a = Scenario.load_json(
        file_path=os.path.join(TEST_DATA_DIR, "test-scenario.json"),
        email=EMAIL,
        api_key=API_KEY,
    )
    a.addModule(module_name='test-module')

    with pytest.warns(UserWarning, match="Module already found"):
        a.addModule(module_name='test-module')


def test_addJob_noncallable():
    a = Scenario(name="non-callable-pipeline-func")

    with pytest.raises(ValueError, match='FAILED: Requested function' \
    ' "str_not_callable" not found'):
        a.addJob(func="str_not_callable")


def monkeypatch_badjob_new_id_fail(*args, **kwargs):
    raise RuntimeError("invalid_job")


def test_addJob_bad(monkeypatch):
    a = Scenario(name="bad-job")
    monkeypatch.setattr("pvdeg.utilities.new_id", monkeypatch_badjob_new_id_fail)
    with pytest.warns(UserWarning, match="Failed to add job: invalid_job"):
        a.addJob(func=lambda: None, func_kwarg={})
