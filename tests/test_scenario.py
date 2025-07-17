from pvdeg.scenario import Scenario
from pvdeg.standards import standoff
from pvdeg import TEST_DATA_DIR
from pvdeg import weather
import json
import pandas as pd
import pytest
import os
import numpy as np
from pvdeg import utilities

# problems with scenario creating directory in test directory?
EMAIL = "user@mail.com"
API_KEY = "DEMO_KEY"


def monkeypatch_addLocation(self, *args, **kwargs) -> None:
    """
    mocker function to be monkey patched at runtime for Scenario.addLocation to avoid psm3 api calls and use local weather files instead.
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
        target=Scenario,
        name="addLocation",
        value=monkeypatch_addLocation
    )

    a = Scenario(name="test")

    EMAIL = "placeholder@email.xxx",
    API_KEY =  "fake_key"

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
        target=Scenario,
        name="addLocation",
        value=monkeypatch_addLocation
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


# temporarily disabling test in line with propsoed update from print errors to keyerrors
#
# def test_addModule_badmat(capsys, monkeypatch):

#     ### monkey patch to bypass psm3 api calls in addLocation called by load_json ###
#     monkeypatch.setattr(
#         target=Scenario,
#         name="addLocation",
#         value=monkeypatch_addLocation
#     )

#     a = Scenario.load_json(
#         file_path=os.path.join(TEST_DATA_DIR, "test-scenario.json"),
#         email=EMAIL,
#         api_key=API_KEY,
#     )

#     a.addModule(module_name="fail", material="fake-material")

#     captured = capsys.readouterr()
#     assert "Material Not Found - No module added to scenario." in captured.out
#     assert "If you need to add a custom material, use .add_material()" in captured.out


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


def test_addModule_string_material_valid():
    scenario = Scenario(name="test_scenario")
    scenario.addModule(
        module_name="test_module",
        materials="OX003",
        material_file="O2permeation",
        parameters=['Ead', 'Do', 'Eas', 'So', 'Eap', 'Po']
    )
    assert len(scenario.modules) == 1
    assert scenario.modules[0]["module_name"] == "test_module"
    assert scenario.modules[0]["material_params"] == {
        'Ead': 29.43112031,
        'Do': 0.129061678,
        'Eas': 16.6314948252219,
        'So': 0.136034525059804,
        'Eap': 49.1083457348515,
        'Po': 528718258.338532
        }


def test_addModule_string_material_invalid_name():
    scenario = Scenario(name="test_scenario")
    with pytest.raises(ValueError, match="Material 'invalid_name' not found in O2permeation"):
        scenario.addModule(
            module_name="test_module",
            materials="invalid_name",
            material_file="O2permeation",
            parameters=['Ead', 'Do', 'Eas', 'So', 'Eap', 'Po']
        )
    assert len(scenario.modules) == 0


def test_addModule_string_material_invalid_file():
    scenario = Scenario(name="test_scenario")
    with pytest.raises(ValueError, match="Material 'OX003' not found in invalid_file"):
        scenario.addModule(
            module_name="test_module",
            materials="OX003",
            material_file="invalid_file",
            parameters=['Ead', 'Do', 'Eas', 'So', 'Eap', 'Po']
        )
    assert len(scenario.modules) == 0


def test_addModule_dict_single_material_valid_name():
    materials_dict = {
        "encapsulant": {
            "material_file": "O2permeation",
            "material_name": "OX003"
        }
    }
    scenario = Scenario(name="test_scenario")
    scenario.addModule(
        module_name="test_module",
        materials=materials_dict,
        parameters=['Ead', 'Do', 'Eas', 'So', 'Eap', 'Po']
    )
    assert len(scenario.modules) == 1
    assert scenario.modules[0]["module_name"] == "test_module"
    assert scenario.modules[0]["material_params"] == {
        'encapsulant': {
            'Ead': 29.43112031,
            'Do': 0.129061678,
            'Eas': 16.6314948252219,
            'So': 0.136034525059804,
            'Eap': 49.1083457348515,
            'Po': 528718258.338532
        }
    }


def test_addModule_dict_single_material_invalid_name():
    materials_dict = {
        "encapsulant": {
            "material_file": "O2permeation",
            "material_name": "invalid_name"
        }
    }
    scenario = Scenario(name="test_scenario")
    with pytest.raises(ValueError, match="Material 'invalid_name' not found in O2permeation"):
        scenario.addModule(
            module_name="test_module",
            materials=materials_dict,
            parameters=['Ead', 'Do', 'Eas', 'So', 'Eap', 'Po']
        )
    assert len(scenario.modules) == 0


def test_addModule_dict_single_material_invalid_file():
    materials_dict = {
        "encapsulant": {
            "material_file": "invalid_file",
            "material_name": "OX003"
        }
    }

    scenario = Scenario(name="test_scenario")
    with pytest.raises(ValueError, match="Material 'OX003' not found in invalid_file"):
        scenario.addModule(
            module_name="test_module",
            materials=materials_dict,
            parameters=['Ead', 'Do', 'Eas', 'So', 'Eap', 'Po']
        )
    assert len(scenario.modules) == 0


def test_addModule_dict_multiple_material_valid():
    materials_dict = {
        "encapsulant": {
            "material_file": "O2permeation",
            "material_name": "OX003"
        },
        "backsheet": {
                "material_file": "H2Opermeation",
                "material_name": "W024"
        }
    }
    scenario = Scenario(name="test_scenario")
    scenario.addModule(
        module_name="test_module",
        materials=materials_dict,
        parameters=['Ead', 'Do', 'Eas', 'So', 'Eap', 'Po']
    )
    assert len(scenario.modules) == 1
    assert scenario.modules[0]["module_name"] == "test_module"
    assert scenario.modules[0]["material_params"] == {
        'encapsulant': {
            'Ead': 29.43112031,
            'Do': 0.129061678,
            'Eas': 16.6314948252219,
            'So': 0.136034525059804,
            'Eap': 49.1083457348515,
            'Po': 528718258.338532
        },
        'backsheet': {
            'Ead': 96.5385865449266,
            'Do': 4172967.14420414,
            'Eas': -12.3825598156611,
            'So': 0.000027596664527881,
            'Eap': 84.1560267292654,
            'Po': 994982178508.989
        }
    }


def test_add_single_custom_material():
    scenario = Scenario(name="test_scenario")
    scenario.materials = {}
    
    custom_params = {
        "Ead": 95.0,
        "Do": 40e5,
        "Eas": -10.0,
        "So": 20e-6,
        "Eap": 84.0,
        "Po": 99e9
    }
    materials_dict = {
        "custom_layer": {
            "material_name": "custom_material_name",
            "parameters": custom_params
        }
    }
    with pytest.raises(ValueError, match="Error adding custom material for layer 'custom_layer'"):
        scenario.add_material(materials_dict)


def test_add_multi_custom_material():
    scenario = Scenario(name="test_scenario")
    scenario.materials = {}
    
    custom_params_1 = {
        "Ead": 95.0,
        "Do": 40e5,
        "Eas": -10.0,
        "So": 20e-6,
        "Eap": 84.0,
        "Po": 99e9
    }
    custom_params_2 = {
        "Ead": 100.0,
        "Do": 30e5,
        "Eas": -20.0,
        "So": 30e-6,
        "Eap": 80.0,
        "Po": 90e9
    }
    materials_dict = {
        "custom_layer_1": {
            "material_name": "custom_material_name_1",
            "parameters": custom_params_1
        },
        "custom_layer_2": {
            "material_name": "custom_material_name_2",
            "parameters": custom_params_2
        }
    }
    with pytest.raises(ValueError, match="Error adding custom material for layer 'custom_layer_1'"):
        scenario.add_material(materials_dict)


def test_add_material_invalid_layer_spec():
    scenario = Scenario(name="test_scenario")
    materials_dict = {
        "encapsulant": "not_a_dict"  # Should be a dict
    }
    
    with pytest.raises(ValueError, match="Invalid material spec for layer 'encapsulant' - must be a dict"):
        scenario.add_material(materials_dict)


def test_add_material_mixed_valid_invalid():
    scenario = Scenario(name="test_scenario")
    materials_dict = {
        "valid_layer": {
            "material_file": "O2permeation",
            "material_name": "OX003"
        },
        "invalid_layer": {
            "material_file": "O2permeation"
            # Missing material_name
        }
    }
    
    with pytest.raises(ValueError, match="material_name is required for layer 'invalid_layer'"):
        scenario.add_material(materials_dict)


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
