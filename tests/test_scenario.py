from pvdeg.scenario import Scenario, _resolve_material_params
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
    """Mocker function to be monkey patched at runtime for Scenario.addLocation to avoid

    psm3 api calls and use local weather files instead.
    """

    PSM_FILE = os.path.join(TEST_DATA_DIR, r"psm3_pytest.csv")
    weather_df, meta = weather.read(PSM_FILE, "psm")

    self.email, self.api_key = None, None

    self.lat_long = (-999, -999)
    self.weather_data = weather_df
    self.meta_data = meta
    self.gids = np.asanyarray([1245357])


def test_Scenario_add(monkeypatch, tmp_path):
    # monkey patch to bypass psm3 api calls in addLocation
    monkeypatch.setattr(
        target=Scenario,
        name="addLocation",
        value=monkeypatch_addLocation,
    )

    a = Scenario(name="test")

    EMAIL = "placeholder@email.xxx"

    API_KEY = "fake_key"

    a.clean()
    a.restore_credentials(email=EMAIL, api_key=API_KEY)
    a.addLocation(lat_long=(40.63336, -73.99458))
    a.addModule(module_name="test-module")
    a.addJob(func=(standoff, "encapsulant"))

    restored = Scenario.load_json(
        file_path=os.path.join(TEST_DATA_DIR, "test-scenario.json")
    )

    a.name, restored.name = None, None
    a.path, restored.path = None, None
    a.file, restored.file = None, None

    assert a.name == restored.name
    assert len(a.modules) == len(restored.modules)
    assert len(a.pipeline) == len(restored.pipeline)


def test_Scenario_run(monkeypatch, tmp_path):
    # monkey patch to bypass psm3 api calls in addLocation called by load_json
    monkeypatch.setattr(
        target=Scenario, name="addLocation", value=monkeypatch_addLocation
    )

    a = Scenario.load_json(
        file_path=os.path.join(TEST_DATA_DIR, "test-scenario.json"),
        email=EMAIL,
        api_key=API_KEY,
    )
    a.path = tmp_path
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


def test_run_multilayer_material_injection(monkeypatch, tmp_path):
    """Test that run() handles nested (multi-layer) material_params correctly.

    Each job is bound to a different material layer. Material params that don't
    match the function signature should be silently filtered out. Results for
    both jobs should be produced.
    """
    monkeypatch.setattr(
        target=Scenario, name="addLocation", value=monkeypatch_addLocation
    )

    s = Scenario(name="multilayer-run-test", path=tmp_path)
    monkeypatch_addLocation(s)

    s.addModule(
        module_name="two-layer-module",
        materials={
            "encapsulant": {
                "material_file": "O2permeation",
                "material_name": "OX003",
            },
            "backsheet": {
                "material_file": "O2permeation",
                "material_name": "OX004",
            },
        },
    )

    s.addJob(func=(standoff, "encapsulant"))
    s.addJob(func=(standoff, {"wind_factor": 0.35}, "backsheet"))

    s.run()

    module_results = s.results["two-layer-module"]
    assert len(module_results) == 2

    for job_id, result in module_results.items():
        assert result is not None
        assert isinstance(result, pd.DataFrame)


# def test_clean():
#     a = Scenario(name='clean-a')
#     a.file = 'non-existent-file.json'
#     with pytest.raises(FileNotFoundError):
#         a.clean()

#     b = Scenario(name='clean-b')
#     with pytest.raises(ValueError):
#         b.clean()


def test_addLocation_pvgis(tmp_path):
    a = Scenario(
        name="location-test",
        path=tmp_path,
    )
    with pytest.raises(ValueError):
        a.addLocation((40.63336, -73.99458), weather_db="PSM3")  # no api key


def test_addModule_badkey():
    a = Scenario.load_json(
        file_path=os.path.join(TEST_DATA_DIR, "test-scenario.json"),
        email=EMAIL,
        api_key=API_KEY,
    )

    with pytest.raises(ValueError, match="Material 'invalid-key' not found"):
        a.addModule(module_name="test-invalid-key", materials="invalid-key")


def test_addModule_existingmod():
    a = Scenario.load_json(
        file_path=os.path.join(TEST_DATA_DIR, "test-scenario.json"),
        email=EMAIL,
        api_key=API_KEY,
    )
    a.addModule(module_name="test-module")

    with pytest.warns(UserWarning, match="Module already found"):
        a.addModule(module_name="test-module")


def monkeypatch_badjob_new_id_fail(*args, **kwargs):
    raise RuntimeError("invalid_job")


def test_addJob_bad(monkeypatch):
    a = Scenario(name="bad-job")
    monkeypatch.setattr("pvdeg.utilities.new_id", monkeypatch_badjob_new_id_fail)
    with pytest.warns(UserWarning, match="Failed to add job: invalid_job"):
        a.addJob(func=(standoff, "encapsulant"))


def test_addModule_string_material_valid():
    scenario = Scenario(name="test_scenario")
    scenario.addModule(
        module_name="test_module",
        materials="OX003",
        material_file="O2permeation",
        parameters=["Ead", "Do", "Eas", "So", "Eap", "Po"],
    )
    assert len(scenario.modules) == 1
    assert scenario.modules[0]["module_name"] == "test_module"
    assert scenario.modules[0]["material_params"] == {
        "Ead": 29.43112031,
        "Do": 0.129061678,
        "Eas": 16.6314948252219,
        "So": 0.136034525059804,
        "Eap": 49.1083457348515,
        "Po": 528718258.338532,
    }


def test_addModule_string_material_invalid_name():
    scenario = Scenario(name="test_scenario")
    with pytest.raises(
        ValueError, match="Material 'invalid_name' not found in O2permeation"
    ):
        scenario.addModule(
            module_name="test_module",
            materials="invalid_name",
            material_file="O2permeation",
            parameters=["Ead", "Do", "Eas", "So", "Eap", "Po"],
        )
    assert len(scenario.modules) == 0


def test_addModule_string_material_invalid_file():
    scenario = Scenario(name="test_scenario")
    with pytest.raises(ValueError, match="Material 'OX003' not found in invalid_file"):
        scenario.addModule(
            module_name="test_module",
            materials="OX003",
            material_file="invalid_file",
            parameters=["Ead", "Do", "Eas", "So", "Eap", "Po"],
        )
    assert len(scenario.modules) == 0


def test_addModule_dict_single_material_valid_name():
    materials_dict = {
        "encapsulant": {"material_file": "O2permeation", "material_name": "OX003"}
    }
    scenario = Scenario(name="test_scenario")
    scenario.addModule(
        module_name="test_module",
        materials=materials_dict,
        parameters=["Ead", "Do", "Eas", "So", "Eap", "Po"],
    )
    assert len(scenario.modules) == 1
    assert scenario.modules[0]["module_name"] == "test_module"
    assert scenario.modules[0]["material_params"] == {
        "encapsulant": {
            "Ead": 29.43112031,
            "Do": 0.129061678,
            "Eas": 16.6314948252219,
            "So": 0.136034525059804,
            "Eap": 49.1083457348515,
            "Po": 528718258.338532,
        }
    }


def test_addModule_dict_single_material_invalid_name():
    materials_dict = {
        "encapsulant": {
            "material_file": "O2permeation",
            "material_name": "invalid_name",
        }
    }
    scenario = Scenario(name="test_scenario")
    with pytest.raises(
        ValueError, match="Material 'invalid_name' not found in O2permeation"
    ):
        scenario.addModule(
            module_name="test_module",
            materials=materials_dict,
            parameters=["Ead", "Do", "Eas", "So", "Eap", "Po"],
        )
    assert len(scenario.modules) == 0


def test_addModule_dict_single_material_invalid_file():
    materials_dict = {
        "encapsulant": {"material_file": "invalid_file", "material_name": "OX003"}
    }

    scenario = Scenario(name="test_scenario")
    with pytest.raises(ValueError, match="Material 'OX003' not found in invalid_file"):
        scenario.addModule(
            module_name="test_module",
            materials=materials_dict,
            parameters=["Ead", "Do", "Eas", "So", "Eap", "Po"],
        )
    assert len(scenario.modules) == 0


def test_addModule_dict_multiple_material_valid():
    materials_dict = {
        "encapsulant": {"material_file": "O2permeation", "material_name": "OX003"},
        "backsheet": {"material_file": "H2Opermeation", "material_name": "W024"},
    }
    scenario = Scenario(name="test_scenario")
    scenario.addModule(
        module_name="test_module",
        materials=materials_dict,
        parameters=["Ead", "Do", "Eas", "So", "Eap", "Po"],
    )
    assert len(scenario.modules) == 1
    assert scenario.modules[0]["module_name"] == "test_module"
    assert scenario.modules[0]["material_params"] == {
        "encapsulant": {
            "Ead": 29.43112031,
            "Do": 0.129061678,
            "Eas": 16.6314948252219,
            "So": 0.136034525059804,
            "Eap": 49.1083457348515,
            "Po": 528718258.338532,
        },
        "backsheet": {
            "Ead": 96.5385865449266,
            "Do": 4172967.14420414,
            "Eas": -12.3825598156611,
            "So": 0.000027596664527881,
            "Eap": 84.1560267292654,
            "Po": 994982178508.989,
        },
    }


def test_add_single_custom_material():
    scenario = Scenario(name="test_scenario")

    custom_params = {
        "Ead": 95.0,
        "Do": 40e5,
        "Eas": -10.0,
        "So": 20e-6,
        "Eap": 84.0,
        "Po": 99e9,
    }
    materials_dict = {
        "custom_layer": {
            "material_name": "custom_material_name",
            "parameters": custom_params,
        }
    }
    with pytest.raises(
        ValueError, match="Error adding custom material for layer 'custom_layer'"
    ):
        scenario.add_material(materials_dict)


def test_add_multi_custom_material():
    scenario = Scenario(name="test_scenario")

    custom_params_1 = {
        "Ead": 95.0,
        "Do": 40e5,
        "Eas": -10.0,
        "So": 20e-6,
        "Eap": 84.0,
        "Po": 99e9,
    }
    custom_params_2 = {
        "Ead": 100.0,
        "Do": 30e5,
        "Eas": -20.0,
        "So": 30e-6,
        "Eap": 80.0,
        "Po": 90e9,
    }
    materials_dict = {
        "custom_layer_1": {
            "material_name": "custom_material_name_1",
            "parameters": custom_params_1,
        },
        "custom_layer_2": {
            "material_name": "custom_material_name_2",
            "parameters": custom_params_2,
        },
    }
    with pytest.raises(
        ValueError, match="Error adding custom material for layer 'custom_layer_1'"
    ):
        scenario.add_material(materials_dict)


def test_add_material_invalid_layer_spec():
    scenario = Scenario(name="test_scenario")
    materials_dict = {"encapsulant": "not_a_dict"}  # Should be a dict

    with pytest.raises(
        ValueError,
        match="Invalid material spec for layer 'encapsulant' - must be a dict",
    ):
        scenario.add_material(materials_dict)


def test_add_material_mixed_valid_invalid():
    scenario = Scenario(name="test_scenario")
    materials_dict = {
        "valid_layer": {"material_file": "O2permeation", "material_name": "OX003"},
        "invalid_layer": {
            "material_file": "O2permeation"
            # Missing material_name
        },
    }

    with pytest.raises(
        ValueError, match="material_name is required for layer 'invalid_layer'"
    ):
        scenario.add_material(materials_dict)


def test_addJob_tuple_wrong_length():
    s = Scenario(name="tuple-wrong-length")
    # Tuple with only one element
    with pytest.raises(ValueError, match="Job tuple must have 2 or 3 elements"):
        s.addJob(func=(standoff,))
    # Tuple with four elements (too many)
    with pytest.raises(ValueError, match="Job tuple must have 2 or 3 elements"):
        s.addJob(func=(standoff, {}, "encapsulant", "extra"))


def test_addJob_2tuple_first_not_callable():
    s = Scenario(name="2tuple-noncallable")
    # 2-tuple: first element not callable
    with pytest.raises(ValueError, match="First element of job tuple must be callable"):
        s.addJob(func=(123, "encapsulant"))


def test_addJob_2tuple_second_dict_no_layer():
    s = Scenario(name="2tuple-second-dict-no-layer")
    # 2-tuple: second element is a dict — backward compat, no layer injection
    s.addJob(func=(standoff, {"wind_factor": 0.35}))
    jobs = list(s.pipeline.values())
    assert len(jobs) == 1
    assert jobs[0]["job"] == standoff
    assert jobs[0]["params"] == {"wind_factor": 0.35}
    assert jobs[0]["material_layer"] is None


def test_addJob_2tuple_second_invalid_type():
    s = Scenario(name="2tuple-second-invalid-type")
    # 2-tuple: second element is a list — not str (layer) or dict (kwargs)
    with pytest.raises(
        ValueError, match="Second element of a 2-tuple must be a layer name"
    ):
        s.addJob(func=(standoff, [1, 2, 3]))


def test_addJob_3tuple_first_not_callable():
    s = Scenario(name="3tuple-noncallable")
    with pytest.raises(ValueError, match="First element of job tuple must be callable"):
        s.addJob(func=(123, {}, "encapsulant"))


def test_addJob_3tuple_second_not_dict():
    s = Scenario(name="3tuple-second-not-dict")
    with pytest.raises(ValueError, match="Second element of a 3-tuple must be a dict"):
        s.addJob(func=(standoff, "not_a_dict", "encapsulant"))


def test_addJob_3tuple_third_not_str():
    s = Scenario(name="3tuple-third-not-str")
    with pytest.raises(
        ValueError, match="Third element of a 3-tuple must be a layer name"
    ):
        s.addJob(func=(standoff, {}, 123))


def test_addJob_invalid_job_type():
    s = Scenario(name="invalid-job-type")
    # Non-callable, non-tuple (e.g. plain int) is not accepted
    with pytest.raises(ValueError, match="Each job must be a callable"):
        s.addJob(func=123)


def test_addJob_bare_callable_valid():
    s = Scenario(name="bare-callable-valid")
    # Bare callable — no layer, no extra kwargs
    s.addJob(func=standoff)
    jobs = list(s.pipeline.values())
    assert len(jobs) == 1
    assert jobs[0]["job"] == standoff
    assert jobs[0]["params"] == {}
    assert jobs[0]["material_layer"] is None


def test_addJob_2tuple_valid():
    s = Scenario(name="2tuple-valid")
    s.addJob(func=(standoff, "encapsulant"))
    jobs = list(s.pipeline.values())
    assert len(jobs) == 1
    assert jobs[0]["job"] == standoff
    assert jobs[0]["params"] == {}
    assert jobs[0]["material_layer"] == "encapsulant"


def test_addJob_3tuple_valid():
    s = Scenario(name="3tuple-valid")
    s.addJob(func=(standoff, {"wind_factor": 0.35}, "encapsulant"))
    jobs = list(s.pipeline.values())
    assert len(jobs) == 1
    assert jobs[0]["job"] == standoff
    assert jobs[0]["params"] == {"wind_factor": 0.35}
    assert jobs[0]["material_layer"] == "encapsulant"


def test_addJob_list_valid():
    s = Scenario(name="list-valid")
    s.addJob(
        func=[
            (standoff, "encapsulant"),
            (standoff, {"wind_factor": 0.5}, "backsheet"),
        ]
    )
    jobs = list(s.pipeline.values())
    assert len(jobs) == 2
    assert jobs[0]["material_layer"] == "encapsulant"
    assert jobs[1]["material_layer"] == "backsheet"
    assert jobs[1]["params"] == {"wind_factor": 0.5}


def test_resolve_material_params_missing_layer_warns():
    """Warning + empty dict when requested layer is absent from nested params."""
    nested = {"encapsulant": {"Ead": 30.0}, "backsheet": {"Ead": 25.0}}
    with pytest.warns(UserWarning, match="missing from material_params"):
        result = _resolve_material_params(nested, "frontsheet")
    assert result == {}


def test_resolve_material_params_flat_with_layer_warns():
    """Warning + empty dict when material_params is flat but a layer is requested."""
    flat = {"Ead": 29.4, "Do": 0.13}
    with pytest.warns(UserWarning, match="not structured as named layers"):
        result = _resolve_material_params(flat, "encapsulant")
    assert result == {}
