"""DRAFT TESTS. Not for publication"""
"""DRAFT TESTS. Not for publication"""
"""DRAFT TESTS. Not for publication"""
"""DRAFT TESTS. Not for publication"""

import pytest
from pvdeg.scenario import Scenario
from pvdeg import utilities

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