from pvdeg.scenario import Scenario
from pvdeg.standards import standoff
from pvdeg import TEST_DATA_DIR
import json
import pandas as pd
import pytest
import os

# problems with scenario creating directory in test directory?
EMAIL = "user@mail.com"
API_KEY = "DEMO_KEY"


def test_Scenario_add():
    a = Scenario(name="test")
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


def test_Scenario_run():
    a = Scenario.load_json(
        file_path=os.path.join(TEST_DATA_DIR, "test-scenario.json"),
        email=EMAIL,
        api_key=API_KEY,
    )
    a.run()

    res_df = a.results["test-module"]["GLUSE"]
    known_df = pd.DataFrame(
        {
            "x": {0: 0.0},
            "T98_0": {0: 68.80867997141961},
            "T98_inf": {0: 46.362946615593664},
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


def test_addModule_badmat(capsys):
    a = Scenario.load_json(
        file_path=os.path.join(TEST_DATA_DIR, "test-scenario.json"),
        email=EMAIL,
        api_key=API_KEY,
    )

    a.addModule(module_name="fail", material="fake-material")

    captured = capsys.readouterr()
    assert "Material Not Found - No module added to scenario." in captured.out
    assert "If you need to add a custom material, use .add_material()" in captured.out


# def test_addModule_existingmod(capsys):
#     b = Scenario.load_json(file_path=os.path.join(TEST_DATA_DIR, 'test-scenario.json'), email=EMAIL, api_key=API_KEY)

#     b.addModule(module_name='test-module')

#     captured = capsys.readouterr()
#     assert 'WARNING - Module already found by name "test-module"' in captured.out
#     assert "Module will be replaced with new instance." in captured.out

# just stdout not errout
# def test_addModule_seeadded(capsys):
#     c = Scenario(name='see-added-module')

#     c.addModule(module_name='works-see-added')

#     captured = capsys.readouterr()
#     assert 'Module "works-see-added" added.' in captured.out


def test_addJob_bad(capsys):
    a = Scenario(name="non-callable-pipeline-func")

    a.addJob(func="str_not_callable")

    captured = capsys.readouterr()
    assert 'FAILED: Requested function "str_not_callable" not found' in captured.out
    assert "Function has not been added to pipeline." in captured.out


# def test_addJob_seeadded():
#     a = Scenario(name='good-func-see-added')
#     func=standoff

#     with pytest.warns(UserWarning) as record:
#         a.addJob(func=func,see_added=True)

#     hex_addr = hex(id(func)).replace('0x', '').lstrip()
#     hex_addr = '0x' + hex_addr.lower()
#     message = f"{{'job': <function {func.__name__} at {hex_addr}>, 'params': {{}}}}"

#     assert len(record) == 1
#     assert record[0].category == UserWarning
#     assert str(record[0].message) == message


# geospatial tests should only run if on hpc, ask martin about protocol. load meta csv and weather nc (for very small scenario?)
