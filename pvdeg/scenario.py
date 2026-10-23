"""Scenario objects and methods for accelerated analysis."""

import pvdeg
from pvdeg import utilities

import matplotlib.pyplot as plt
from datetime import datetime as dt
import os
from shutil import rmtree
import json
from inspect import signature
import warnings
import pandas as pd
import xarray as xr
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from typing import List, Union, Optional, Tuple, Callable
import pprint
from IPython.display import display, HTML


def _resolve_material_params(material_params, material_layer):
    """Resolve which material parameters to inject for a given job.

    Parameters:
    -----------
    material_params : dict
        The module's material parameters. Either a flat dict of scalar
        values (single-material, from ``addModule(materials="OX003")``)
        or a nested dict of dicts (multi-layer, from
        ``addModule(materials={"encapsulant": {...}, ...})``).
    material_layer : str or None
        The layer name bound to the job via ``addJob``. If ``None``,
        no material parameters are injected.

    Returns:
    --------
    dict
        Flat dict of material parameters to inject into the function, or
        an empty dict if no injection is needed.
    """
    if material_layer is None:
        return {}

    # Detect nested (multi-layer) vs flat (single-material) structure.
    # Nested: all values are dicts, e.g. {"encapsulant": {"Ead": ...}}
    # Flat:   values are scalars,   e.g. {"Ead": 29.4, "Do": 0.13, ...}
    is_nested = material_params and all(
        isinstance(v, dict) for v in material_params.values()
    )

    if is_nested:
        if material_layer not in material_params:
            warnings.warn(
                f"Layer '{material_layer}' is missing from material_params. "
                "No material parameters will be injected for this job. "
                f"Available layers: {list(material_params.keys())}"
            )
            return {}
        return material_params[material_layer]
    else:
        # material_params is flat (single material) but a layer name was
        # requested — layer selection requires a nested material dict.
        warnings.warn(
            f"Layer '{material_layer}' was specified but material_params is not "
            "structured as named layers. No material parameters will be injected. "
            "Use a dict-of-dicts when calling addModule to enable layer selection."
        )
        return {}


class Scenario:
    """Scenario object, contains all parameters and criteria for a given scenario.

    Generally speaking, this will be information such as: Scenario Name, Path,
    Geographic Location, Module Type, Racking Type
    """

    def __init__(
        self,
        name: Optional[str] = None,
        path: Optional[str] = None,
        gids: Optional[Union[int, List[int], np.ndarray[int]]] = None,
        modules: Optional[list] = None,
        pipeline: Optional[OrderedDict] = None,
        file: Optional[str] = None,
        results=None,
        weather_data: Optional[pd.DataFrame] = None,  # df
        meta_data: Optional[dict] = None,  # dict
        email: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the degradation scenario object.

        Parameters:
        -----------
        name : (str)
            custom name for deg. scenario. If none given, will use datetime of
            initialization (DDMMYY_HHMMSS)
        path : (str, pathObj)
            File path to operate within and store results. If none given, new folder
            "name" will be created in the working directory.
        gids : (str, pathObj)
            Spatial area to perform calculation for. This can be Country or Country and
            State.
        modules : (list, str)
            List of module names to include in calculations.
        pipeline : (list, str)
            List of function names to run in job pipeline
        file : (path)
            Full file path to a pre-generated Scenario object. If specified, all other
            parameters
            will be ignored and taken from the .json file.
        results : (pd.Series)
            Full collection of outputs from pipeline execution. Populated by
            ``scenario.runPipeline()``
        """
        self.name = name
        self.path = path
        self.modules = modules if modules is not None else []
        self.gids = gids
        self.pipeline = pipeline if pipeline is not None else OrderedDict()
        self.results = results
        self.weather_data = weather_data
        self.meta_data = meta_data
        self.lat_long = None
        self.api_key = api_key
        self.email = email

        filedate = dt.now().strftime("%d%m%y_%H%M%S")

        if name is None:
            name = filedate
        self.name = name

        if path is None:
            from pvdeg import config as _cfg

            _base = (
                os.environ.get("PVDEG_JOB_PATH")
                or _cfg.SCENARIO_OUTPUT_PATH
                or os.getcwd()
            )
            self.path = os.path.join(_base, f"pvd_job_{self.name}")
        else:
            self.path = path

        os.makedirs(self.path, exist_ok=True)

        if file:
            self.load_json(file_path=file, email=email, api_key=api_key)

    def __eq__(self, other):
        """Define the behavior of the `==` operator between two Scenario instances.

        Does not check credentials.
        """
        if not isinstance(other, Scenario):
            print("wrong type")
            return False

        def compare_ordereddict_values(od1, od2):
            return list(od1.values()) == list(od2.values())

        return (
            self.name == other.name
            and self.path == other.path
            and np.array_equal(self.gids, other.gids)
            and self.modules == other.modules
            and compare_ordereddict_values(
                self.pipeline, other.pipeline
            )  # keys are random
            and self.file == other.file
            and self.results == other.results
            and (
                self.weather_data.equals(other.weather_data)
                if self.weather_data is not None and other.weather_data is not None
                else self.weather_data is other.weather_data
            )
            and self.meta_data == other.meta_data
            and self.email == other.email
            and self.api_key == other.api_key
        )

    def clean(self):
        """Wipe the Scenario object filetree.

        This is useful because the Scenario object
        stores its data in local files outside of the python script. This causes issues
        when two unique scenario instances are created in the same directory, they
        appear to be seperate instances to python but share the same data (if no path is
        provided). Changes made to one are reflected in both.

        Parameters:
        -----------
        None

        See Also:
        ---------
        `pvdeg.utilties.remove_scenario_filetree`
        to remove all pvd_job_* directories and children from a directory
        """
        if self.path:
            rmtree(path=self.path)  # error when file is not found
        else:
            raise ValueError(f"{self.name} does not have a path attribute")

    def addLocation(
        self,
        lat_long: tuple = None,
        weather_db: str = "PSM4",
    ):
        """Add a location to the scenario using a latitude-longitude pair.

        The scenario object instance must already be populated with
        credentials when making a call to the NSRBD. Provide credentials
        during class intialization or using `Scenario.restore_credentials`

        Parameters:
        -----------
        lat-long : tuple
            tuple of floats representing a latitude longitude coordinate.
            >>> (24.7136, 46.6753) #Riyadh, Saudi Arabia
        weather_db : str
            source of data for provided location.
            - For NSRDB data use `weather_db = 'PSM4'`
            - For PVGIS data use `weather_db = 'PVGIS'`
        """
        if isinstance(lat_long, list):  # is a list when reading from json
            lat_long = tuple(lat_long)

        if (
            isinstance(lat_long, tuple)
            and all(isinstance(item, (int, float)) for item in lat_long)
            and len(lat_long) == 2
        ):
            weather_id = lat_long
            self.lat_long = lat_long  # save coordinate
        else:
            raise ValueError(
                f"arg: lat_long is type = {type(lat_long)}, must be tuple(float)"
            )

        weather_arg = {}

        if weather_db == "PSM4":
            weather_arg = {"map_variables": True}

        if self.email is not None and self.api_key is not None and weather_db == "PSM4":
            credentials = {
                "api_key": self.api_key,
                "email": self.email,
            }
            weather_arg = weather_arg | credentials
        elif weather_db == "PVGIS":
            pass
        else:
            raise ValueError(
                f"""
                email : {self.email} \n api-key : {self.api_key}
                Must provide an email and api key during class initialization
                when using NDSRDB : {weather_db} == 'PSM4'
                """
            )

        try:
            point_weather, point_meta = pvdeg.weather.get(
                weather_db, id=weather_id, **weather_arg
            )

            if weather_db == "PSM4":
                gid = point_meta["Location ID"]
                self.gids = [int(gid)]

            self.meta_data = point_meta
            self.weather_data = point_weather

        except KeyError as e:
            warnings.warn(f"Metadata missing location ID: {e}")
        except Exception as e:
            warnings.warn(f"Failed to add location: {e}")

    def addModule(
        self,
        module_name: str = None,
        racking: str = "open_rack_glass_polymer",
        materials: Union[str, dict] = "OX003",
        material_file: str = "O2permeation",
        parameters: Optional[list] = None,
        temperature_model: str = "sapm",
        model_kwarg: dict = {},
        irradiance_kwarg: dict = {},
    ):
        """
        Add a module to the Scenario. Multiple modules can be added. Each module will
        be tested in the given scenario.

        Parameters
        -----------
        module_name : str
            unique name for the module. adding multiple modules of the same name will
            replace the existing entry.
        racking : str
            temperature model racking type as per PVLIB (see pvlib.temperature). Allowed
            entries:
            'open_rack_glass_glass', 'open_rack_glass_polymer',
            'close_mount_glass_glass', 'insulated_back_glass_polymer'
        materials : Union[str, dict]
            Materials specification. Can be either:

            - str: Single material key e.g., "OX003"
            - dict: Nested dictionary with structure PV layer, materials file, material
              key, and parameters if custom material is specifed. For example:

              .. code-block:: python

                  {
                      "encapsulant": {
                          "material_file": "O2permeation",
                          "material_name": "OX003"
                      },
                      "backsheet": {
                          "material_file": "H20permeation",
                          "material_name": "W024"
                      },
                      "custom_layer": {
                          "parameters": {
                              "Ead": 95,
                              "Do": 40e5,
                              "Eas": -10,
                              "So": 20e-6,
                              "Eap": 84,
                              "Po": 99e9
                          }
                      }
                  }

        material_file : str
            Material file used to access parameters if ``material_name`` exists in one
            of the local material json databases. Options:

            - ``"AApermeation"``, ``"H2Opermeation"``, ``"O2permeation"``
        parameters : list
            List of parameter names to retrieve from the material database. If None, all
            parameters are retrieved. This argument is passed to the ``parameters``
            argument of utilities._read_material.
        temperature_model : list[str]
            select pvlib temperature models. See ``pvdeg.temperature.temperature`` for
            more. Options : ``'sapm', 'pvsyst', 'faiman', 'faiman_rad', 'fuentes',
            'ross'``
        model_kwarg : dict, (optional)
            provide a dictionary of temperature model coefficents to be used
            instead of pvlib defaults. Some models will require additional
            arguments such as ``ross`` which requires nominal operating cell
            temperature (``noct``). This is where other values such as noct
            should be provided.
            pvlib-python temperature models:
            https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html  # noqa
        irradiance_kwarg : dict, (optional)
            provide keyword arguments for poa irradiance calculations.
            Options : ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``
        """
        if isinstance(materials, str):
            # Handle single material string format
            try:
                mat_params = utilities.read_material(
                    pvdeg_file=material_file, key=materials, parameters=parameters
                )
            except KeyError:
                raise ValueError(
                    f"Material '{materials}' not found in {material_file}"  # noqa
                )
        elif isinstance(materials, dict):
            # Handle multiple material dictionary format
            mat_params = {}
            for layer, material_spec in materials.items():
                if not isinstance(material_spec, dict):
                    raise ValueError(
                        f"Invalid material specification for layer '{layer}' - "
                        "must be a dict"
                    )

                material_file_layer = material_spec.get("material_file")
                material_name = material_spec.get("material_name")
                custom_params = material_spec.get("parameters")
                # returns None if no custom material specified

                if not material_file_layer:
                    raise ValueError(
                        f"Missing 'material_file' for layer '{layer}'"  # noqa
                    )

                if material_name:
                    # Use existing material from file
                    try:
                        material_parameters = utilities.read_material(
                            pvdeg_file=material_file_layer,
                            key=material_name,
                            parameters=parameters,
                        )
                        mat_params[layer] = material_parameters
                    except KeyError:
                        raise ValueError(
                            f"Material '{material_name}' not found in "  # noqa
                            f"{material_file_layer}"
                        )
                elif custom_params is not None:
                    # Use custom parameters directly
                    mat_params[layer] = custom_params
                else:
                    raise ValueError(
                        f"Layer '{layer}' must have either 'material_name' or "
                        "'parameters'"
                    )
        else:
            raise ValueError("Materials parameter must be either a string or dict")

        # Check for existing module and warn user
        old_modules = [mod["module_name"] for mod in self.modules]
        if module_name in old_modules:
            warnings.warn(f'WARNING - Module already found by name "{module_name}"')
            warnings.warn("Module will be replaced with new instance.")
            self.modules.pop(old_modules.index(module_name))

        self.modules.append(
            {
                "module_name": module_name,
                "racking": racking,
                "material_params": mat_params,
                "temp_model": temperature_model,
                "model_kwarg": model_kwarg,
                "irradiance_kwarg": irradiance_kwarg,
            }
        )

    def add_material(self, materials, see_added=False):
        """
        Add new material types to multiple layers/files.

        Parameters:
        -----------
        materials : dict
            Dictionary with layer names as keys, and material info including:

            - material_file: str - Name of the material file (e.g., "O2permeation")
              (required only for existing materials)
            - material_name: str - Name of the material to add
            - parameters: dict - Custom material parameters (for custom materials)

        Example:
        --------
        .. code-block:: python

            scenario.add_material({
                "encapsulant": {
                    "material_file": "O2permeation",
                    "material_name": "EVA_001"
                },
                "backsheet": {
                    "material_file": "H2Opermeation",
                    "material_name": "PET_001"
                },
                "custom_layer": {
                    "material_name": "CUSTOM_001",
                    "parameters": {
                        "Ead": 95,
                        "Do": 40e5,
                        "Eas": -10,
                        "So": 20e-6,
                        "Eap": 84,
                        "Po": 99e9
                    }
                }
            })
        """
        if not isinstance(materials, dict):
            raise ValueError(
                "Materials parameter must be a dict with layer names as keys"
            )

        for layer, material_spec in materials.items():
            if not isinstance(material_spec, dict):
                raise ValueError(
                    f"Invalid material spec for layer '{layer}' - must be a dict"
                )

            material_file = material_spec.get("material_file")
            material_name = material_spec.get("material_name")
            custom_params = material_spec.get("parameters")

            if material_name is None:
                raise ValueError(f"material_name is required for layer '{layer}'")

            if custom_params is not None and material_file is None:
                material_file = "custom_materials"
                try:
                    utilities._add_material(
                        name=material_name,
                        alias=custom_params.get("alias", layer),
                        Ead=custom_params["Ead"],
                        Eas=custom_params["Eas"],
                        So=custom_params["So"],
                        Do=custom_params.get("Do"),
                        Eap=custom_params.get("Eap"),
                        Po=custom_params.get("Po"),
                        fickian=custom_params.get("fickian", True),
                        fname=f"{material_file}.json",
                    )
                except Exception as e:
                    raise ValueError(
                        f"Error adding custom material for layer '{layer}': {e}"  # noqa
                    )

            elif material_file is not None:
                # Handle existing material from file - read and add to database
                try:
                    material_parameters = utilities.read_material(
                        pvdeg_file=material_file, key=material_name
                    )

                    # Add the existing material to the database
                    utilities._add_material(
                        name=material_name,
                        alias=material_parameters.get("alias", layer),
                        Ead=material_parameters["Ead"],
                        Eas=material_parameters["Eas"],
                        So=material_parameters["So"],
                        Do=material_parameters.get("Do"),
                        Eap=material_parameters.get("Eap"),
                        Po=material_parameters.get("Po"),
                        fickian=material_parameters.get("fickian", True),
                        fname=f"{material_file}.json",
                    )

                except KeyError:
                    raise ValueError(
                        f'Material "{material_name}" not found in {material_file}'  # noqa
                    )
                except Exception as e:
                    raise ValueError(
                        f"Error adding existing material for layer '{layer}': {e}"  # noqa
                    )

            else:
                raise ValueError(
                    f"Either 'material_file' or 'parameters' must be provided for "
                    f"layer '{layer}'"
                )

    def viewScenario(self):
        """Print all scenario information currently stored in the scenario instance.

        Does not implement ipython.display. If available, use this.
        """
        pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)

        if self.name:
            print(f"Name : {self.name}")

        if self.pipeline:
            print("Pipeline : ")

            # pipeline is a list of dictionaries, each list entry is one pipeline job
            df_pipeline = pd.json_normalize(self.pipeline)
            print(df_pipeline.to_string())
        else:
            print("Pipeline : no jobs in pipeline")

        print("Results : ", end="")
        try:
            print("Pipeline results : ")

            for result in self.results:
                if isinstance(result, pd.DataFrame):
                    print(result.to_string())
        except TypeError:
            print("Pipeline has not been run")

        # leave this to make sure the others work
        pp.pprint(f"gids : {self.gids}")
        pp.pprint("test modules :")
        for mod in self.modules:
            pp.pprint(mod)

        # can't check if dataframe is empty
        if isinstance(self.weather_data, (pd.DataFrame, xr.Dataset)):
            print(f"scenario weather : {self.weather_data}")

    def addJob(
        self,
        func,
        name: Optional[str] = None,
        depends_on: Optional[dict] = None,
    ):
        """Add pvdeg function(s) to the scenario pipeline.

        A material layer is optional. When specified, ``run()`` will inject
        the matching layer's material parameters into the function automatically.
        Functions that do not require material parameters (e.g. ``standoff``,
        ``solder_fatigue``) can be added as a bare callable or a
        ``(function, kwargs)`` 2-tuple without a layer name.

        Parameters:
        -----------
        func : callable, tuple, or list
            Specify job(s) to add. Each job must be one of:

            - A bare ``callable`` — add a function with no material layer
              binding (no material parameters are injected).
            - A 2-tuple ``(function, layer_name)`` — bind a function to a
              material layer. Material parameter values are injected
              automatically by ``run()``.
            - A 2-tuple ``(function, extra_kwargs)`` — add a function with
              extra kwargs but no material layer binding (backward compatible
              for functions that do not use material parameters).
            - A 3-tuple ``(function, extra_kwargs, layer_name)`` — as above
              but also supply extra non-material arguments (e.g. experimental
              conditions such as ``I_chamber``) *and* bind to a material layer.
            - A list of any of the above for adding multiple jobs at once.

            ``layer_name`` (str) must match a key in the module's
            ``material_params`` dict (e.g. ``"encapsulant"``,
            ``"backsheet"``).

        name : str or None
            Optional unique name for this job. Used for sequential analysis in the
            Scenario pipeline. Pass this name to a later job's ``depends_on`` variable
            to forward the output of this job as an input to the later job at run time.
        depends_on : dict or None
            ``{kwarg_name: source_job_name}`` mapping. The job from which to take the
            output as an input for this job. At run time, ``run()`` resolves each source
            (depends on) job's result and passes it as the named kwarg to this function.
            The source job must already exist in the pipeline (i.e. have been added with
            a matching ``name``).
        """
        # name/depends_on cannot be used with list form
        if isinstance(func, list) and (name is not None or depends_on is not None):
            raise ValueError(
                "'name' and 'depends_on' cannot be used with list-form addJob. "
                "Call addJob separately for each job you want to name."
            )

        # Normalize input to list of jobs
        if isinstance(func, list):
            jobs_to_add = func
        else:
            jobs_to_add = [func]

        # Process each job
        if len(jobs_to_add) == 1:
            return self._add_single_job(jobs_to_add[0], name, depends_on)
        else:
            return [self._add_single_job(job) for job in jobs_to_add]

    def _add_single_job(
        self,
        job_spec,
        name: Optional[str] = None,
        depends_on: Optional[dict] = None,
    ):
        """Parse a job spec and add a single job to the pipeline.

        Parameters:
        -----------
        job_spec : callable or tuple
            One of:
            - A bare callable — no material layer, no extra kwargs.
            - ``(func, layer_name)`` — bind to a material layer.
            - ``(func, extra_kwargs)`` — extra kwargs, no layer (backward compat).
            - ``(func, extra_kwargs, layer_name)`` — extra kwargs + layer binding.
        name : str or None
            Optional unique name for this job. Used with ``depends_on`` to
            reference this job's output from a later job.
        depends_on : dict or None
            Mapping of ``{kwarg_name: source_job_name}`` declaring that the
            result of a previously-named job should be forwarded as a keyword
            argument to this job at run time.
        """
        # Parse the spec into (func, func_kwarg, material_layer)
        if callable(job_spec):
            func, func_kwarg, material_layer = job_spec, {}, None
        elif isinstance(job_spec, tuple):
            if len(job_spec) == 2:
                job_func, second = job_spec
                if not callable(job_func):
                    raise ValueError(
                        "First element of job tuple must be callable, "
                        f"got {type(job_func)}"
                    )
                if isinstance(second, str):
                    func, func_kwarg, material_layer = job_func, {}, second
                elif isinstance(second, dict):
                    func, func_kwarg, material_layer = job_func, second, None
                else:
                    raise ValueError(
                        "Second element of a 2-tuple must be a layer name "
                        f"(str) or extra kwargs (dict), got {type(second)}"
                    )
            elif len(job_spec) == 3:
                job_func, job_kwargs, layer_name = job_spec
                if not callable(job_func):
                    raise ValueError(
                        "First element of job tuple must be callable, "
                        f"got {type(job_func)}"
                    )
                if not isinstance(job_kwargs, dict):
                    raise ValueError(
                        "Second element of a 3-tuple must be a dict of "
                        f"extra kwargs, got {type(job_kwargs)}"
                    )
                if not isinstance(layer_name, str):
                    raise ValueError(
                        "Third element of a 3-tuple must be a layer name "
                        f"(str), got {type(layer_name)}"
                    )
                func, func_kwarg, material_layer = job_func, job_kwargs, layer_name
            else:
                raise ValueError(
                    "Job tuple must have 2 or 3 elements, " f"got {len(job_spec)}"
                )
        else:
            raise ValueError(
                "Each job must be a callable, a 2-tuple, or a 3-tuple, "
                f"got {type(job_spec)}"
            )

        # Validate and insert into pipeline
        try:
            if name is not None:
                existing_names = {
                    j.get("name") for j in self.pipeline.values() if j.get("name")
                }
                if name in existing_names:
                    raise ValueError(
                        f"A job named '{name}' already exists in the pipeline."
                    )
            if depends_on:
                existing_names = {
                    j.get("name") for j in self.pipeline.values() if j.get("name")
                }
                for kwarg, source in depends_on.items():
                    if source not in existing_names:
                        raise ValueError(
                            f"depends_on references unknown job '{source}'. "
                            "Add the source job before the dependent job."
                        )
            job_id = utilities.new_id(self.pipeline)
            self.pipeline[job_id] = {
                "job": func,
                "params": func_kwarg,
                "material_layer": material_layer,
                "name": name,
                "depends_on": depends_on or {},
            }
            return job_id
        except Exception as e:
            warnings.warn(f"Failed to add job: {e}")

    def run(self):
        """
        Run all jobs in pipeline on scenario object for each module in the scenario.

        For jobs bound to a material layer (via ``addJob``), only the
        parameters for that layer are injected. For jobs with no layer
        binding, no material parameters are injected.

        Note: if a pipeline job contains a function not adhering to the
        package-wide pv parameter naming scheme, the job will raise a
        fatal error.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        results_series = pd.Series(dtype="object")
        results_dict = {}

        if self.modules:
            for module in self.modules:
                module_result = {}
                completed: dict = {}

                weather_dict = {
                    "weather_df": self.weather_data,
                    "meta": self.meta_data,
                }

                temperature_args = {
                    "temp_model": module["temp_model"],
                    "model_kwarg": module["model_kwarg"],
                    "irradiance_kwarg": module["irradiance_kwarg"],
                    "conf": module["racking"],
                    **module["irradiance_kwarg"],
                }

                for id, job in self.pipeline.items():
                    func = job["job"]
                    params = job["params"]
                    material_layer = job.get("material_layer", None)
                    job_name = job.get("name")
                    job_depends_on = job.get("depends_on", {})

                    # Resolve upstream job outputs into kwargs
                    upstream_kwargs = {}
                    for kwarg, source_name in job_depends_on.items():
                        if source_name in completed:
                            upstream_kwargs[kwarg] = completed[source_name]

                    # Resolve material params for this specific job's layer
                    job_material_params = _resolve_material_params(
                        module["material_params"], material_layer
                    )

                    combined = (
                        weather_dict
                        | temperature_args
                        | job_material_params
                        | upstream_kwargs
                    )

                    func_params = signature(func).parameters
                    func_args = {k: v for k, v in combined.items() if k in func_params}

                    res = func(**params, **func_args)
                    module_result[id] = res
                    if job_name:
                        completed[job_name] = res
                        module_result[job_name] = res

                results_dict[module["module_name"]] = module_result

            self.results = results_dict

            for module, pipeline_result in self.results.items():
                module_dir = os.path.join(
                    self.path, "pipeline_results", f"{module}_pipeline_results"
                )
                os.makedirs(module_dir, exist_ok=True)
                for function, result in pipeline_result.items():
                    if isinstance(result, (pd.Series, pd.DataFrame)):
                        result.to_csv(os.path.join(module_dir, f"{function}.csv"))
                    elif isinstance(result, (int, float)):
                        out_file = os.path.join(module_dir, f"{function}.csv")
                        with open(out_file, "w") as file:
                            file.write(f"{result}\n")

        elif not self.modules:
            pipeline_results = {}
            completed: dict = {}

            for id, job in self.pipeline.items():
                func, params = job["job"], job["params"]
                job_name = job.get("name")
                job_depends_on = job.get("depends_on", {})

                # Resolve upstream job outputs into kwargs
                upstream_kwargs = {}
                for kwarg, source_name in job_depends_on.items():
                    if source_name in completed:
                        upstream_kwargs[kwarg] = completed[source_name]

                # Arguments to pass to the function
                func_args = {
                    "weather_df": self.weather_data,
                    "meta": self.meta_data,
                }
                # Merge upstream results, then user-defined parameters
                func_args.update(upstream_kwargs)
                if params:
                    func_args.update(params)

                # Filter arguments to only those accepted by the function
                sig_params = signature(func).parameters
                final_args = {k: v for k, v in func_args.items() if k in sig_params}

                result = func(**final_args)

                results_dict[id] = result
                if job_name:
                    results_dict[job_name] = result
                    completed[job_name] = result
                pipeline_results = results_dict

            for key in pipeline_results.keys():
                if isinstance(results_dict[key], pd.DataFrame):
                    results_series[key] = results_dict[key]
                elif isinstance(results_dict[key], pd.Series):
                    results_series[key] = results_dict[key]
                elif isinstance(results_dict[key], (float, int)):
                    results_series[key] = pd.DataFrame(
                        [results_dict[key]],
                        columns=[key],
                    )
                else:
                    # Preserve any other return type (e.g. a dict of series/scalars)
                    results_series[key] = results_dict[key]

            self.results = results_series

    @classmethod
    def load_json(
        cls,
        file_path: str = None,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Import scenario dictionaries from existing 'scenario.json' file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        name = data["name"]
        path = data["path"]
        modules = data["modules"]
        gids = data["gids"]
        process_pipeline = OrderedDict(data["pipeline"])
        lat_long = data["lat_long"]

        for task in process_pipeline.values():
            utilities._update_pipeline_task(task=task)

        # Handle legacy scenario files with metadata in material_params
        # New files created by read_material(values_only=True) won't need this
        for mod in modules:
            if "material_params" in mod:
                # Check if any values contain metadata dictionaries
                has_metadata = any(
                    isinstance(v, dict) and "value" in v
                    for v in mod["material_params"].values()
                )

                if has_metadata:
                    # Legacy format: extract values from metadata
                    mod["material_params"] = {
                        k: v["value"] if isinstance(v, dict) and "value" in v else v
                        for k, v in mod["material_params"].items()
                    }

        instance = cls()
        instance.name = name
        instance.path = path
        instance.modules = modules
        instance.gids = gids
        instance.pipeline = process_pipeline
        instance.file = file_path

        try:
            instance.email = data["email"]
            instance.api_key = data["api_key"]
        except KeyError:
            print("credentials not in json file using arguments")
            instance.email = email
            instance.api_key = api_key

        instance.addLocation(lat_long=lat_long)
        return instance

    @classmethod
    def remove_scenario_filetrees(fp, pattern="pvd_job_*"):
        """Move `cwd` to fp and remove all scenario file trees from fp directory.

        Permanently deletes all scenario file trees. USE WITH CAUTION.

        Parameters:
        -----------
        fp : string
            file path to directory where all scenario files should be removed
        pattern : str
            pattern to search for using glob. Default value of `pvd_job_` is
            equvilent to `pvd_job_*` in bash.

        Returns
        -------
        None

        See Also
        --------
        `pvdeg.utilities.remove_scenario_filetrees`
        """
        utilities.remove_scenario_filetrees(fp=fp, pattern=pattern)
        return

    def _verify_function(func_name: str) -> Tuple[Callable, List]:
        """Check all classes in pvdeg for a function of the name "func_name".

        Returns a
        callable function and list of all function parameters with no default values.

        Parameters:
        -----------
        func_name : (str)
            Name of the desired function. Only returns for 1:1 matches

        Returns:
        --------
        _func : (func)
            callable instance of named function internal to pvdeg
        reqs : (list(str))
            list of minimum required paramters to run the requested funciton
        """
        from inspect import signature

        class_list = [c for c in dir(pvdeg) if not c.startswith("_")]
        for c in class_list:
            _class = getattr(pvdeg, c)
            if func_name in dir(_class):
                _func = getattr(_class, func_name)
        if _func is None:
            return (None, None)

        # check if necessary parameters given
        reqs_all = signature(_func).parameters
        reqs = []
        for param in reqs_all:
            if reqs_all[param].default == reqs_all[param].empty:
                reqs.append(param)

        return (_func, reqs)

    def _to_dict(self, api_key=False):
        # pipeline is special case, must remove 'job' function reference at every entry
        modified_pipeline = deepcopy(self.pipeline)

        def get_qualified(x):
            return f"{x.__module__}.{x.__name__}"

        for task in modified_pipeline.values():
            function_ref = task["job"]
            task["qualified_function"] = get_qualified(function_ref)
            task.pop("job")

        attributes = {
            "name": self.name,
            "path": self.path,
            "modules": self.modules,
            "gids": self.gids,
            "lat_long": self.lat_long,
            "pipeline": modified_pipeline,
        }

        if api_key:
            protected = {"email": self.email, "api_key": self.api_key}
            attributes.update(protected)

        return attributes

    def dump(self, api_key: bool = False, path: Optional[str] = None) -> None:
        """Serialize the scenario instance as a json.

        No dataframes will be saved but
        some attributes like weather_df and results will be stored in nested file trees
        as csvs.

        Parameters:
        -----------
        api_key : bool, default=``False``
            Save api credentials to json. Default False.
            Use with caution.
        path : str
            location to save. If no path provided save to scenario directory.
        """
        if path is None:
            path = self.path
        target = os.path.join(path, f"{self.name}.json")

        scenario_as_dict = self._to_dict(api_key)
        scenario_as_json = json.dumps(scenario_as_dict, indent=4)

        with open(target, "w") as f:
            f.write(scenario_as_json)

        return

    def restore_credentials(
        self,
        email: str,
        api_key: str,
    ) -> None:
        """Restore email and api key to scenario.

        Use after importing scenario if json
        does not contain email and api key.

        Parameters
        ----------
        email : str
            email associated with nsrdb developer account
        api_key : str
            api key associated with nsrdb developer account
        """
        if self.email is None and self.api_key is None:
            self.email = email
            self.api_key = api_key

    def extract(
        self,
        dim_target: Tuple[str, str],
        col_name: Optional[str] = None,
        tmy: bool = False,
        start_time: Optional[dt] = None,
        end_time: Optional[dt] = None,
    ) -> pd.DataFrame:
        """Extract scenario results along an axis.

        Note
        ----
        Only works if results are of the same shape.
        Ex) running 5 different temperature calculations on the same module.
        Counter Ex) running a standoff and tempeature calc on the same module.

        Ex: ('function' : 'AKWMC)

        Parameters
        ----------
        dim_target : tuple of str
            Define a tuple of `(dimension, name)` to select results.
            The dimension is either 'function' or 'module', and the name
            is the name of the function or module to grab results from.

            Note: Receives job ID, not function name in `dim_target`.

            Dimension options: `'function'`, `'module'`

            Examples:
            To grab 'standoff' result from all modules in the scenario:
            Determine the name of the standoff job using `display(Scenario)`.
            If the job is called `AJCWL`, the result would be:
            `dim_target = ('function', 'AJCWL')`

            To grab all results from a module named 'mod_a':
            `dim_target = ('module', 'mod_a')`

        col_name: Optional[str], default = None
            The column name to extract.
            Only use when results contain dataframes with multiple columns.
            Extranious if results are pd.Series or single numeric values.

        tmy: bool, default False
            Whether to use typical meteorological year data.

        start_time: Optional[dt.datetime], default None
            The start time for the data extraction.

        end_time: Optional[dt.datetime], default None
            The end time for the data extraction.
        """
        if self.results is None:
            raise ValueError("No scenario results. Run pipeline with ``.run()``")

        if not isinstance(dim_target, tuple):
            raise TypeError(f"dim_target is type: {type(dim_target)} must be tuple")
        if len(dim_target) != 2:
            raise ValueError(f"size dim_target={len(dim_target)} must be length 2")

        results = pd.DataFrame()

        if dim_target[0] == "module":
            sub_dict = self.results[dim_target[1]]

            for key, value in sub_dict.items():
                if isinstance(value, pd.Series):
                    results[key] = value
                elif isinstance(value, pd.DataFrame):
                    if col_name is not None:
                        results[key] = value[col_name]
                    else:
                        raise ValueError(
                            "col_name must be provided for DataFrame extraction"
                        )

        elif dim_target[0] == "function":
            for module, sub_dict in self.results.items():
                for function, function_result in sub_dict.items():
                    if dim_target[1] == function:
                        if isinstance(function_result, pd.Series):
                            results[module] = function_result
                        elif isinstance(function_result, pd.DataFrame):
                            if col_name is not None:
                                results[module] = function_result[col_name]
                            else:
                                raise ValueError(
                                    "col_name must be provided for DataFrame extraction"
                                )

        if tmy:

            def set_placeholder_year(dt):
                return dt.replace(year=1970)

            if isinstance(results.index, pd.DatetimeIndex):
                results.index = pd.DatetimeIndex(
                    [set_placeholder_year(dt) for dt in results.index]
                )
            else:
                # Convert to DatetimeIndex if it isn't already
                results.index = pd.to_datetime(results.index)
                results.index = pd.DatetimeIndex(
                    [set_placeholder_year(dt) for dt in results.index]
                )

            if start_time and end_time:
                results = utilities.strip_normalize_tmy(results, start_time, end_time)

        return results

    def plot(
        self,
        dim_target: Tuple[str, str],
        col_name: Optional[str] = None,
        tmy: bool = False,
        start_time: Optional[dt] = None,
        end_time: Optional[dt] = None,
        title: str = "",
    ) -> tuple:
        """Plot scenario results along an axis using `Scenario.extract`.

        Note:
        --------
        only works if results are of the same shape.
        Ex) running 5 different temperature calculations on the same module.
        Counter Ex) running a standoff and tempeature calc on the same module.

        Ex: ('function' : 'AKWMC)

        Parameters:
        -----------
        dim_target : tuple of str
            Define a tuple of `(dimension, name)` to select results.
            The dimension is either 'function' or 'module', and the name
            is the name of the function or module to grab results from.

            Note: Receives job ID, not function name in `dim_target`.

            Dimension options: `'function'`, `'module'`

            Examples:
            To grab 'standoff' result from all modules in the scenario:
            Determine the name of the standoff job using `display(Scenario)`.
            If the job is called `AJCWL`, the result would be:
            `dim_target = ('function', 'AJCWL')`

            To grab all results from a module named 'mod_a':
            `dim_target = ('module', 'mod_a')`

        col_name: Optional[str], default = None
            The column name to extract.
            Only use when results contain dataframes with multiple columns.
            Extranious if results are pd.Series or single numeric values.

        tmy: bool, default False
            Whether to use typical meteorological year data.

        start_time: Optional[dt.datetime], default None
            The start time for the data extraction.

        end_time: Optional[dt.datetime], default None
            The end time for the data extraction.

        title: Optional[str], default ''
            Name of the matplotlib plot

        Returns:
        --------
        fig, ax: tuple
            matplotlib figure and axis objects

        See Also:
        ---------
        `Scenario.extract`
        To have more control over a plot simply extract the data and then use
        more specific plotting logic
        """
        df = self.extract(
            dim_target=dim_target,
            col_name=col_name,
            tmy=tmy,
            start_time=start_time,
            end_time=end_time,
        )

        fig, ax = plt.subplots()
        df.plot(ax=ax)
        ax.set_title(f"{self.name} : {title}")
        plt.show()

        return fig, ax

    def _ipython_display_(self):
        file_url = "no file provided"
        if self.path:
            file_url = (
                f"file:///{os.path.abspath(self.path).replace(os.sep, '/')}"  # noqa
            )
        html_content = f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 3px; margin-top: 5px;">  # noqa
            <h2>self.name: {self.name}</h2>
            <p><strong>self.path:</strong> <a href="{file_url}" target="_blank">{self.path}</a></p>  # noqa
            <p><strong>self.gids:</strong> {self.gids}</p>
            <p><strong>self.email:</strong> {self.email}</p>
            <p><strong>self.api_key:</strong> {self.api_key}</p>
            <div>
                <h3>self.results</h3>
                {self.format_results() if self.results is not None else None}
            </div>
            <div>
                <h3>self.pipeline</h3>
                {self.format_pipeline()}
            </div>
            <div>
                <h3>self.modules</h3>
                {self.format_modules()}
            </div>
            <div>
                <h3>self.weather_data</h3>
                {self.format_weather()}
            </div>
            <div>
                <h3>self.meta_data</h3>
                {self.meta_data}
            </div>
        </div>
        <p><i>All attributes can be accessed by the names shown above.</i></p>
        <script>
            function toggleVisibility(id) {{
                var content = document.getElementById(id);
                var arrow = document.getElementById('arrow_' + id);
                if (content.style.display === 'none') {{
                    content.style.display = 'block';
                    arrow.innerHTML = '▼';
                }} else {{
                    content.style.display = 'none';
                    arrow.innerHTML = '►';
                }}
            }}
        </script>
        """
        display(HTML(html_content))

    def format_modules(self):
        modules_html = "<div>"
        for i, module in enumerate(self.modules):
            material_params_html = (
                f"<pre>{json.dumps(module['material_params'], indent=2)}</pre>"
            )
            model_kwarg_html = (
                f"<pre>{json.dumps(module['model_kwarg'], indent=2)}</pre>"
            )
            irradiance_kwarg_html = (
                f"<pre>{json.dumps(module['irradiance_kwarg'], indent=2)}</pre>"
            )

            module_content = f"""
            <div onclick="toggleVisibility('module_{i}')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">  # noqa
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_module_{i}" style="color: #E6E6FA;">►</span>
                    {module["module_name"]}
                </h4>
            </div>
            <div id="module_{i}" style="display: none; margin-left: 20px; padding: 5px; background-color: #f0f0f0; color: #000;">  # noqa
                <p><strong>Racking:</strong> {module["racking"]}</p>
                <p><strong>Temperature Model:</strong> {module["temp_model"]}</p>
                <p><strong>Material Parameters:</strong></p>
                <div style="margin-left: 20px;">
                    {material_params_html}
                </div>
                <p><strong>Model Arguments:</strong></p>
                <div style="margin-left: 20px;">
                    {model_kwarg_html}
                </div>
                <p><strong>Irradiance Arguments:</strong></p>
                <div style="margin-left: 20px;">
                    {irradiance_kwarg_html}
                </div>
            </div>
            """
            modules_html += module_content
        modules_html += "</div>"
        return modules_html

    def format_results(self):
        results_html = "<div>"
        for module_name, functions in sorted(self.results.items()):
            module_id = f"result_module_{module_name}"
            module_content = f"""
            <div onclick="toggleVisibility('{module_id}')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">  # noqa
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_{module_id}" style="color: #E6E6FA;">►</span>
                    {module_name}
                </h4>
            </div>
            <div id="{module_id}" style="display: none; margin-left: 20px; padding: 5px; background-color: #f0f0f0; color: #000;">  # noqa
            """
            for function_name, output in functions.items():
                function_id = f"{module_id}_{function_name}"
                formatted_output = self.format_output(output)
                module_content += f"""
                <div onclick="toggleVisibility('{function_id}')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">  # noqa
                    <h5 style="font-family: monospace; margin: 0;">
                        <span id="arrow_{function_id}" style="color: #E6E6FA;">►</span>
                        {function_name}
                    </h5>
                </div>
                <div id="{function_id}" style="display: none; margin-left: 20px; padding: 5px; background-color: #f0f0f0; color: #000;">  # noqa
                    {formatted_output}
                </div>
                """
            module_content += "</div>"
            results_html += module_content
        results_html += "</div>"
        return results_html

    def format_output(self, output):
        if isinstance(output, pd.Series):
            output = pd.DataFrame(
                output
            )  # convert Series to DataFrame for HTML display
        if isinstance(output, pd.DataFrame):
            head = output.head(10).to_html()
            tail = output.tail(10).to_html()
            return f"{head}<br>...<br>{tail}"
        else:
            return str(output)

    def format_weather(self):
        weather_data_html = ""
        if isinstance(self.weather_data, pd.DataFrame):
            if len(self.weather_data) > 10:
                first_five = self.weather_data.head(5)
                last_five = self.weather_data.tail(5)
                ellipsis_row = pd.DataFrame(
                    [["..."] * len(self.weather_data.columns)],
                    columns=self.weather_data.columns,
                )

                # Create the custom index with ellipses
                custom_index = np.concatenate(
                    [
                        np.arange(0, 5, dtype=object).astype(str),
                        ["..."],
                        np.arange(
                            len(self.weather_data) - 5,
                            len(self.weather_data),
                            dtype=object,
                        ).astype(str),
                    ]
                )

                # Concatenate the DataFrames
                display_data = pd.concat(
                    [first_five, ellipsis_row, last_five], ignore_index=True
                )
                display_data.index = custom_index
            else:
                display_data = self.weather_data

            weather_data_html = f"""
            <div id="weather_data" onclick="toggleVisibility('content_weather_data')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">  # noqa
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_content_weather_data" style="color: #E6E6FA;">►</span>  # noqa
                    Weather Data
                </h4>
            </div>
            <div id="content_weather_data" style="display: none; margin-left: 20px; padding: 5px; background-color: #f0f0f0; color: #000;">  # noqa
                {display_data.to_html()}
            </div>
            """
        return weather_data_html

    def format_pipeline(self):
        pipeline_html = "<div>"
        for step_name, step in self.pipeline.items():
            try:
                if isinstance(step["params"], pd.DataFrame):
                    params_html = "<pre>DataFrame (not displayed)</pre>"
                else:
                    params_html = f"<pre>{json.dumps(step['params'], indent=2)}</pre>"
            except TypeError:  # json dumps fails
                params_html = "<pre>Unserializable data type</pre>"

            step_content = f"""
            <div id="{step_name}" onclick="toggleVisibility('pipeline_{step_name}')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">  # noqa
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_pipeline_{step_name}" style="color: #b676c2;">►</span>  # noqa
                    {step["job"].__name__}, <span style="color: #b676c2;">#{step_name}</span>  # noqa
                </h4>
            </div>
            <div id="pipeline_{step_name}" style="display: none; margin-left: 20px; padding: 5px; background-color: #f0f0f0; color: #000;">  # noqa
                <p>Job: {step["job"].__name__}</p>
                <p>Parameters:</p>
                <div style="margin-left: 20px;">
                    {params_html}
                </div>
            </div>
            """
            pipeline_html += step_content
        pipeline_html += "</div>"
        return pipeline_html
