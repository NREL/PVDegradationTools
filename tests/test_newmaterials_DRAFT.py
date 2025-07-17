"""DRAFT TESTS. Not for publication"""
"""DRAFT TESTS. Not for publication"""
"""DRAFT TESTS. Not for publication"""
"""DRAFT TESTS. Not for publication"""

import pytest
from unittest.mock import patch
from pvdeg.scenario import Scenario
from pvdeg import utilities

@patch('pvdeg.utilities.read_material')
def test_addModule_string_material_valid(mock_read_material):
    # Mock the read_material function to return expected data
    mock_read_material.return_value = {
        'Ead': 29.43112031,
        'Do': 0.129061678,
        'Eas': 16.6314948252219,
        'So': 0.136034525059804,
        'Eap': 49.1083457348515,
        'Po': 528718258.338532
    }
    
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
