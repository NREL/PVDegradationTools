"""Class to define an analysis scenario."""

import pvdeg
from pvdeg import utilities

from datetime import date
from datetime import datetime as dt
import os
from shutil import rmtree
import json
from inspect import signature
import warnings
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from copy import deepcopy
from typing import List, Union, Optional, Tuple, Callable
from functools import partial
import pprint
from IPython.display import display, HTML
import cartopy.crs as ccrs
import cartopy.feature as cfeature

### premade scenario with locations of interest. Ask Mike?
# TODO: geospatial reset weather and addLocation from gids.


class Scenario:
    """
    The scenario object contains all necessary parameters and criteria for a given scenario.
    Generally speaking, this will be information such as:
    Scenario Name, Path, Geographic Location, Module Type, Racking Type
    """

    def __init__(
        self,
        name: Optional[str] = None,
        path: Optional[str] = None,
        gids: Optional[Union[int, List[int], np.ndarray[int]]] = None,
        modules: Optional[list] = [],
        pipeline=OrderedDict(),
        file: Optional[str] = None,
        results=None,
        weather_data: Optional[pd.DataFrame] = None,  # df
        meta_data: Optional[dict] = None,  # dict
        email: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the degradation scenario object.

        Parameters:
        -----------
        name : (str)
            custom name for deg. scenario. If none given, will use date of initialization (DDMMYY)
        path : (str, pathObj)
            File path to operate within and store results. If none given, new folder "name" will be
            created in the working directory.
        gids : (str, pathObj)
            Spatial area to perform calculation for. This can be Country or Country and State.
        modules : (list, str)
            List of module names to include in calculations.
        pipeline : (list, str)
            List of function names to run in job pipeline
        file : (path)
            Full file path to a pre-generated Scenario object. If specified, all other parameters
            will be ignored and taken from the .json file.
        results : (pd.Series)
            Full collection of outputs from pipeline execution. Populated by ``scenario.runPipeline()``
        """

        self.name = name
        self.path = path
        self.modules = modules
        self.gids = gids
        self.pipeline = pipeline
        self.results = results
        self.weather_data = weather_data
        self.meta_data = meta_data
        self.lat_long = None
        self.api_key = api_key
        self.email = email

        filedate = dt.strftime(date.today(), "%d%m%y")

        if name is None:
            name = filedate
        self.name = name

        if path is None:
            self.path = os.path.join(os.getcwd(), f"pvd_job_{self.name}")
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        os.chdir(self.path)

        if file:
            self.load_json(file_path=file, email=email, api_key=api_key)

    def __eq__(self, other):
        """
        Define the behavior of the `==` operator between two Scenario instances.
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
            and compare_ordereddict_values(self.pipeline, other.pipeline)
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
        """
        Wipe the Scenario object filetree. This is useful because the Scenario
        object stores its data in local files outside of the python script.
        This causes issues when two unique scenario instances are created in
        the same directory, they appear to be seperate instances to python but
        share the same data (if no path is provided).
        Changes made to one are reflected in both.

        Parameters:
        -----------
        None

        See Also:
        --------
        `pvdeg.utilties.remove_scenario_filetree`
        to remove all pvd_job_* directories and children from a directory
        """
        if self.path:
            os.chdir(os.pardir)
            rmtree(path=self.path)  # error when file is not found
        else:
            raise ValueError(f"{self.name} does not have a path attribute")

    def addLocation(
        self,
        lat_long: tuple = None,
        weather_db: str = "PSM3",
        see_added: bool = False,
    ):
        """
        Add a location to the scenario using a latitude-longitude pair.

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
            - For NSRDB data use `weather_db = 'PSM3'`
            - For PVGIS data use `weather_db = 'PVGIS'`
        see_added : bool
            flag true if you want to see a runtime notification for added location/gids
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

        if weather_db == "PSM3":
            weather_arg = {"names": "tmy", "attributes": [], "map_variables": True}

        if self.email is not None and self.api_key is not None and weather_db == "PSM3":
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
                when using NDSRDB : {weather_db} == 'PSM3'
                """
            )

        point_weather, point_meta = pvdeg.weather.get(
            weather_db, id=weather_id, **weather_arg
        )

        try:
            if weather_db == "PSM3":
                gid = point_meta["Location ID"]
                self.gids = [int(gid)]
        except KeyError:
            return UserWarning("metadata missing location ID")

        self.meta_data = point_meta
        self.weather_data = point_weather

        if see_added and weather_db == "PSM3":
            message = f"Gids Added - {self.gids}"
            warnings.warn(message, UserWarning)

    def addModule(
        self,
        module_name: str = None,
        racking: str = "open_rack_glass_polymer",
        material: str = "OX003",
        temperature_model: str = "sapm",
        model_kwarg: dict = {},
        irradiance_kwarg: dict = {},
        see_added: bool = False,
    ):
        """
        Add a module to the Scenario. Multiple modules can be added. Each module will be tested in
        the given scenario.

        Parameters:
        -----------
        module_name : (str)
            unique name for the module. adding multiple modules of the same name will replace the
            existing entry.
        racking : (str)
            temperature model racking type as per PVLIB (see pvlib.temperature). Allowed entries:
            'open_rack_glass_glass', 'open_rack_glass_polymer',
            'close_mount_glass_glass', 'insulated_back_glass_polymer'
        material : (str)
            Key of the material desired. For a complete list, see pvdeg/data/O2permeation.json
            or pvdeg/data/H2Opermedation.json or pvdeg/data/AApermeation.json.
            To add a custom material, see pvdeg.addMaterial (ex: EVA, Tedlar)
        temp_model : (str)
            select pvlib temperature models. See ``pvdeg.temperature.temperature`` for more.
            Options : ``'sapm', 'pvsyst', 'faiman', 'faiman_rad', 'fuentes', 'ross'``
        model_kwarg : (dict), optional
            provide a dictionary of temperature model coefficents to be used
            instead of pvlib defaults. Some models will require additional
            arguments such as ``ross`` which requires nominal operating cell
            temperature (``noct``). This is where other values such as noct
            should be provided.
            Pvlib temp models:
            https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html
        irradiance_kwarg : (dict), optional
            provide keyword arguments for poa irradiance calculations.
            Options : ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``
        see_added : (bool), optional
        """

        try:
            mat_params = utilities._read_material(name=material)
        except KeyError:
            print("Material Not Found - No module added to scenario.")
            print("If you need to add a custom material, use .add_material()")
            return

        old_modules = [mod["module_name"] for mod in self.modules]
        if module_name in old_modules:
            print(f'WARNING - Module already found by name "{module_name}"')
            print("Module will be replaced with new instance.")
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

        if see_added:
            print(f'Module "{module_name}" added.')

    # test this?
    def add_material(
        self, name, alias, Ead, Eas, So, Do=None, Eap=None, Po=None, fickian=True
    ):
        """
        add a new material type to main list
        """
        utilities._add_material(
            name=name,
            alias=alias,
            Ead=Ead,
            Eas=Eas,
            So=So,
            Do=Do,
            Eap=Eap,
            Po=Po,
            fickian=fickian,
        )
        print("Material has been added.")
        print("To add the material as a module in your current scene, run .addModule()")

    def viewScenario(self):
        """
        Print all scenario information currently stored in the scenario instance.
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
        func=None,
        func_kwarg={},
        see_added=False,
    ):
        """
        Add a pvdeg function to the scenario pipeline

        Parameters:
        -----------
        func : function
            pvdeg function to use for single point calculation.
            All regular pvdeg functions will work at a single point when ``Scenario.geospatial == False``
        func_params : dict
            job specific keyword argument dictionary to provide to the function
        see_added : bool
            set flag to get a userWarning notifying the user of the job added
           to the pipeline in method call. ``default = False``
        """

        if func is None or not callable(func):
            print(f'FAILED: Requested function "{func}" not found')
            print("Function has not been added to pipeline.")
            return

        job_id = utilities.new_id(self.pipeline)

        job_dict = {"job": func, "params": func_kwarg}
        self.pipeline[job_id] = job_dict

        if see_added:
            message = f"{func.__name__} added to pipeline as \n {job_dict}"
            warnings.warn(message, UserWarning)

    def run(self):
        """
        Run all jobs in pipeline on scenario object for each module in the scenario.
        Note: if a pipeline job contains a function not adhering to package
        wide pv parameter naming scheme, the job will raise a fatal error.

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

                for id, job in self.pipeline.items():
                    func, params = job["job"], job["params"]

                    weather_dict = {
                        "weather_df": self.weather_data,
                        "meta": self.meta_data,
                    }

                    temperature_args = {
                        "temp_model": module["temp_model"],
                        "model_kwarg": module["model_kwarg"],
                        "irradiance_kwarg": module["irradiance_kwarg"],
                        "conf": module["racking"],
                        **module[
                            "irradiance_kwarg"
                        ],  # overwrite existing irradiance kwarg
                    }

                    combined = (
                        weather_dict | temperature_args | module["material_params"]
                    )

                    func_params = signature(func).parameters
                    func_args = {
                        k: v for k, v in combined.items() if k in func_params.keys()
                    }  # downselect func args

                    res = func(**params, **func_args)

                    if id not in module_result.keys():
                        module_result[id] = res

                results_dict[module["module_name"]] = module_result

            self.results = results_dict  # 2d dictionary array

            for module, pipeline_result in self.results.items():
                module_dir = f"pipeline_results/{module}_pipeline_results"
                os.makedirs(module_dir, exist_ok=True)
                for function, result in pipeline_result.items():
                    if isinstance(result, (pd.Series, pd.DataFrame)):
                        result.to_csv(f"{module_dir}/{function}.csv")
                    elif isinstance(result, (int, float)):
                        with open(f"{module_dir}/{function}.csv", "w") as file:
                            file.write(f"{result}\n")

        elif not self.modules:
            pipeline_results = {}

            for id, job in self.pipeline.items():
                func, params = job["job"], job["params"]

                try:
                    func = partial(
                        func, weather_df=self.weather_data, meta=self.meta_data
                    )
                except:
                    pass

                result = func(**params) if params else func()

                # if id not in module_result.keys():
                # results_dict[id] = result
                # pipeline_results = results_dict
                # pipeline_results[id] = result

                results_dict[id] = result
                pipeline_results = results_dict  # this is weird

            for key in pipeline_results.keys():
                # print(f"results_dict dtype : {type(results_dict[key])}")
                # print(results_dict)

                if isinstance(results_dict[key], pd.DataFrame):
                    results_series[key] = results_dict[key]

                elif isinstance(results_dict[key], (float, int)):
                    results_series[key] = pd.DataFrame(
                        [results_dict[key]],  # convert the single numeric to a list
                        columns=[key],  # name the single column entry in list form
                    )

                self.results = results_series

    @classmethod
    def load_json(
        cls,
        file_path: str = None,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Import scenario dictionaries from an existing 'scenario.json' file
        """
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
        """
        Move `cwd` to fp and remove all scenario file trees from fp directory.
        Permanently deletes all scenario file trees. USE WITH CAUTION.

        Parameters:
        -----------
        fp : string
            file path to directory where all scenario files should be removed
        pattern : str
            pattern to search for using glob. Default value of `pvd_job_` is
            equvilent to `pvd_job_*` in bash.

        Returns:
        --------
        None

        See Also:
        ---------
        `pvdeg.utilities.remove_scenario_filetrees`
        """

        utilities.remove_scenario_filetrees(fp=fp, pattern=pattern)

        return

    def _verify_function(func_name: str) -> Tuple[Callable, List]:
        """
        Check all classes in pvdeg for a function of the name "func_name". Returns a callable function
        and list of all function parameters with no default values.

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

        # find the function in pvdeg
        class_list = [c for c in dir(pvdeg) if not c.startswith("_")]
        func_list = []
        for c in class_list:
            _class = getattr(pvdeg, c)
            if func_name in dir(_class):
                _func = getattr(_class, func_name)
        if _func == None:
            return (None, None)

        # check if necessary parameters given
        reqs_all = signature(_func).parameters
        reqs = []
        for param in reqs_all:
            if reqs_all[param].default == reqs_all[param].empty:
                reqs.append(param)

        return (_func, reqs)

    def _to_dict(self, api_key=False):
        # pipeline is a special case, we need to remove the 'job' function reference at every entry
        modified_pipeline = deepcopy(self.pipeline)
        for task in modified_pipeline.values():
            function_ref = task["job"]
            get_qualified = lambda x: f"{x.__module__}.{x.__name__}"
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
        """
        Serialize the scenario instance as a json. No dataframes will be saved
        but some attributes like weather_df and results will be stored in
        nested file trees as csvs.

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
        """
        Restore email and api key to scenario. Use after importing
        scenario if json does not contain email and api key.

        Parameters:
        -----------
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
        """
        Extract scenario results along an axis.

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
            results.index = results.index.map(
                lambda dt: dt.replace(year=1970)
            )  # placeholder year

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
    ) -> None:
        """
        Plot scenario results along an axis using `Scenario.extract`

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
        -------
        None

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

    def _ipython_display_(self):
        file_url = "no file provided" 
        if self.path:
            file_url = f"file:///{os.path.abspath(self.path).replace(os.sep, '/')}"

        html_content = f"""
        <div style="border:1px solid #ddd; border-radius: 5px; padding: 3px; margin-top: 5px;">
            <h2>{self.name}: Scenario Analysis</h2>
            <p><strong>Path:</strong> <a href="{file_url}" target="_blank">{self.path}</a></p>
            <p><strong>GIDs:</strong> {self.gids}</p>
            <p><strong>Email:</strong> {self.email}</p>
            <p><strong>API Key:</strong> {self.api_key}</p>
            <div>
                <h3>Results</h3>
                {self.format_results() if self.results else None}
            </div>
            <div>
                <h3>Pipeline</h3>
                {self.format_pipeline()}
            </div>
            <div>
                <h3>Modules</h3>
                {self.format_modules()}
            </div>
            <div>
                <h3>Weather Data</h3>
                {self.format_weather()}
            </div>
            <div>
                <h3>Meta Data</h3>
                {self.meta_data}
            </div>
        </div>
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
            <div onclick="toggleVisibility('module_{i}')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_module_{i}" style="color: #E6E6FA;">►</span>
                    {module['module_name']}
                </h4>
            </div>
            <div id="module_{i}" style="display:none; margin-left: 20px; padding: 5px; background-color: #f0f0f0; color: #000;">
                <p><strong>Racking:</strong> {module['racking']}</p>
                <p><strong>Temperature Model:</strong> {module['temp_model']}</p>
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
            <div onclick="toggleVisibility('{module_id}')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_{module_id}" style="color: #E6E6FA;">►</span>
                    {module_name}
                </h4>
            </div>
            <div id="{module_id}" style="display:none; margin-left: 20px; padding: 5px; background-color: #f0f0f0; color: #000;">
            """
            for function_name, output in functions.items():
                function_id = f"{module_id}_{function_name}"
                formatted_output = self.format_output(output)
                module_content += f"""
                <div onclick="toggleVisibility('{function_id}')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">
                    <h5 style="font-family: monospace; margin: 0;">
                        <span id="arrow_{function_id}" style="color: #E6E6FA;">►</span>
                        {function_name}
                    </h5>
                </div>
                <div id="{function_id}" style="display:none; margin-left: 20px; padding: 5px; background-color: #f0f0f0; color: #000;">
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
            <div id="weather_data" onclick="toggleVisibility('content_weather_data')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_content_weather_data" style="color: #E6E6FA;">►</span>
                    Weather Data
                </h4>
            </div>
            <div id="content_weather_data" style="display:none; margin-left: 20px; padding: 5px; background-color: #f0f0f0; color: #000;">
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
            <div id="{step_name}" onclick="toggleVisibility('pipeline_{step_name}')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_pipeline_{step_name}" style="color: #b676c2;">►</span>
                    {step['job'].__name__}, <span style="color: #b676c2;">#{step_name}</span>
                </h4>
            </div>
            <div id="pipeline_{step_name}" style="display:none; margin-left: 20px; padding: 5px; background-color: #f0f0f0; color: #000;">
                <p>Job: {step['job'].__name__}</p>
                <p>Parameters:</p>
                <div style="margin-left: 20px;">
                    {params_html}
                </div>
            </div>
            """
            pipeline_html += step_content
        pipeline_html += "</div>"
        return pipeline_html


class GeospatialScenario(Scenario):
    def __init__(
        self,
        name: str = None,
        path: str = None,
        gids: Optional[Union[list, np.ndarray]] = None,
        modules: list = [],
        pipeline: dict = {},
        file=None,
        results=None,
        hpc=False,
        geospatial=False,
        weather_data: xr.Dataset = None,
        meta_data: pd.DataFrame = None,
    ):
        super().__init__(
            name=name,
            path=path,
            gids=gids,
            modules=modules,
            pipeline=pipeline,
            file=file,
            results=results,
            weather_data=weather_data,
            meta_data=meta_data,
        )
        self.geospatial = geospatial
        self.hpc = hpc

    def __eq__(self, other):
        raise NotImplementedError("cannot directly compare geospatial scenario objects")

    # add restoring from gids functionality from nsrdb
    def addLocation(
        self,
        country: Optional[str] = None,
        state: Optional[str] = None,
        county: Optional[str] = None,
        year: int = 2022,
        satellite: str = "Americas",
        nsrdb_attributes: List[str] = [
            "air_temperature",
            "wind_speed",
            "dhi",
            "ghi",
            "dni",
            "relative_humidity",
        ],
        downsample_factor: int = 0,
        gids: Optional[Union[int, List[int], np.ndarray]] = None,
        bbox_kwarg: Optional[dict] = {},
        see_added: bool = False,
    ) -> None:
        """
        Add a location to the scenario. This can be done in three ways: Pass (region, region_col) for gid list.

        Parameters:
        -----------
        country : str
            country to include from NSRDB. Currently supports full string names only.
            Either single string form or list of strings form.
            Examples:
            - ``country='United States'``
            - ``country=['United States']``
            - ``country=['Mexico', 'Canada']``

        state : str
            combination of states or provinces to include from NSRDB.
            Supports two-letter codes for American states. Can mix two-letter
            codes with full length strings. Can take single string, or list of strings (len >= 1)
            Examples:
            - ``state='Washington'``
            - ``state=WA`` (state abbr is case insensitive)
            - ``state=['CO', 'British Columbia']``

        county : str
            county to include from NSRDB. If duplicate county exists in two
            states present in the ``state`` argument, both will be included.
            If no state is provided
        downsample_factor : int
            downsample the weather and metadata attached to the region you have selected. default(0), means no downsampling
        year : int
            year of data to use from NSRDB, default = ``2022``
        nsrdb_attributes : list(str)
            list of strings of weather attributes to grab from the NSRDB, must be valid NSRDB attributes (insert list of valid options here).\
            Default = ``['air_temperature', 'wind_speed', 'dhi', 'ghi', 'dni', 'relative_humidity']``
        see_added : bool
            flag true if you want to see a runtime notification for added location/gids
        """

        if self.gids is not None:
            print(
                "Scenario already has designated project points.\nNothing has been added."
            )
            print(self.gids)
            return

        weather_db = "NSRDB"
        weather_arg = {
            "satellite": satellite,
            "names": year,
            "NREL_HPC": True,
            "attributes": nsrdb_attributes,
        }

        # nsrdb_fp = r"/datasets/NSRDB" # kestrel directory
        geo_weather, geo_meta = pvdeg.weather.get(
            weather_db, geospatial=True, **weather_arg
        )

        if gids:
            geo_meta = geo_meta.loc[gids]

        if bbox_kwarg:
            bbox_gids = pvdeg.geospatial.apply_bounding_box(geo_meta, **bbox_kwarg)
            geo_meta = geo_meta.loc[bbox_gids]

        # string to list whole word list or keep list
        toList = lambda s: s if isinstance(s, list) else [s]

        if country:
            countries = toList(country)
            geo_meta = geo_meta[geo_meta["country"].isin(countries)]
        if state:
            states = toList(state)
            states = [
                pvdeg.utilities._get_state(entry) if len(entry) == 2 else entry
                for entry in states
            ]
            geo_meta = geo_meta[geo_meta["state"].isin(states)]
        if county:
            if isinstance(county, str):
                county = toList(county)

            geo_meta = geo_meta[geo_meta["county"].isin(county)]

        # we don't downsample weather data until this runs
        # because on NSRDB we are storing weather OUT of MEMORY with dask
        geo_meta, geo_gids = pvdeg.utilities.gid_downsampling(
            geo_meta, downsample_factor
        )

        self.weather_data = geo_weather
        self.meta_data = geo_meta
        self.gids = geo_gids

        if see_added:
            message = f"Gids Added - {self.gids}"
            warnings.warn(message, UserWarning)

        return

    def location_bounding_box(
        self,
        coord_1: Optional[tuple[float]] = None,
        coord_2: Optional[tuple[float]] = None,
        coords: Optional[np.ndarray[float]] = None,
    ) -> None:
        """

        Apply a latitude-longitude rectangular bounding box to
        geospatial scenario metadata.

        Parameters:
        -----------
        coord_1 : list, tuple
            Top left corner of bounding box as lat-long coordinate pair as list or
            tuple.
        coord_2 : list, tuple
            Bottom right corner of bounding box as lat-long coordinate pair in list
            or tuple.
        coords : np.array
            2d tall numpy array of [lat, long] pairs. Bounding box around the most
            extreme entries of the array. Alternative to providing top left and
            bottom right box corners. Could be used to select amongst a subset of
            data points. ex) Given all points for the planet, downselect based on
            the most extreme coordinates for the United States coastline information.

        Returns:
        --------
        None
        """
        bbox_gids = pvdeg.geospatial.apply_bounding_box(
            self.meta_data, coord_1, coord_2, coords
        )

        self.meta_data = self.meta_data.loc[bbox_gids]

    def classify_mountains_radii(
        self,
        kdtree,
        rad_1: Union[float, int] = 12,
        rad_2: Union[float, int] = 1,
        threshold_factor: Union[float, int] = 1.25,
        elevation_floor: Union[float, int] = 0,
        bbox_kwarg: Optional[dict] = {},
    ):
        """
        Find mountains from elevation metadata using sklearn kdtree for fast lookup.
        Compares a large area of points to a small area of points to find
        significant changes in elevation representing mountains. Tweak the radii
        to determine the sensitivity and noise. Bad radii cause the result to
        become unstable quickly. kdtree can be generated using
        ``pvdeg.geospatial.meta_KDTree``

        Parameters:
        -----------
        meta_df : pd.DataFrame
            Dataframe of metadata as generated by pvdeg.weather.get for geospatial
        kdtree : sklearn.neighbors.KDTree
            kdtree containing latitude-longitude pairs for quick lookups
            Generate using ``pvdeg.geospatial.meta_KDTree``
        rad_1 : float
            radius of the larger search area whose elevations are compared against
            the smaller search area. controls the kdtree query region.
        rad_2 : float
            radius of the smaller search area whose elevations are compared to the
            larger area. controls the kdtree query region.
        threshold_factor : float
            change the significance level of elevation difference between
            small and large regions. Higher means terrain must be more extreme to
            register as a mountain. Small changes result in large differences here.
            When the left side of the expression is greater, the datapoint is
            classified as a mountain.
            ``local mean elevation > broad mean elevation * threshold_factor``
        elevation_floor : int
            minimum inclusive elevation in meters. If a point has smaller location
            it will be clipped from result.

        Returns:
        --------
        None, strictly updates meta_data attribute of geospatial scenari instance.
        """

        gids = pvdeg.geospatial.identify_mountains_radii(
            meta_df=self.meta_data,
            kdtree=kdtree,
            rad_1=rad_1,
            rad_2=rad_2,
            threshold_factor=threshold_factor,
            elevation_floor=elevation_floor,
            bbox_kwarg=bbox_kwarg,
        )

        self.meta_data["mountain"] = (self.meta_data.index).isin(gids)
        return

    def classify_mountains_weights(
        self,
        kdtree,
        threshold: int = 0,
        percentile: int = 75,
        k_neighbors: int = 3,
        method: str = "mean",
        normalization: str = "linear",
    ):
        """
        Add a column to the scenario meta_data dataframe containing a boolean
        True or False value representing if the entry is a near a mountain.
        Calculated from weights assigned during stochastic downselection.

        Parameters:
        -----------
        kdtree : sklearn.neighbors.KDTree or str
            kdtree containing latitude-longitude pairs for quick lookups
            Generate using ``pvdeg.geospatial.meta_KDTree``. Can take a pickled
            kdtree as a path to the .pkl file.
        threshold : float
            minimum weight that a mountain can be identifed.
            value between `[0,1]` (inclusive)
        percentile : float, int, (default = 75)
            mountain classification sensitivity. Calculates percentile of values
            remaining after thresholding, weights above this percentile are
            classified as mountains. value between `[0, 100]` (inclusive)
        k_neighbors : int, (default = 3)
            number of neighbors to check for elevation data in nearest neighbors
        method : str, (default = 'mean')
            method to calculate elevation weights for each point.
            Options : `'mean'`, `'sum'`, `'median'`
        normalization : str, (default = 'linear')
            function to apply when normalizing weights. Logarithmic uses log_e/ln
            options : `'linear'`, `'logarithmic'`, '`exponential'`

        Returns:
        --------
        None, strictly updates meta_data attribute of scenario.

        See Also:
        ---------
        `pvdeg.geospatial.identify_mountains_weights`
        """
        gids = pvdeg.geospatial.identify_mountains_weights(
            meta_df=self.meta_data,
            kdtree=kdtree,
            threshold=threshold,
            percentile=percentile,
            k_neighbors=k_neighbors,
            method=method,
            normalization=normalization,
        )

        self.meta_data["mountain"] = (self.meta_data.index).isin(gids)
        return

    def classify_feature(
        self,
        kdtree=None,
        feature_name=None,
        resolution="10m",
        radius=None,
        bbox_kwarg={},
    ):
        """
        kdtree : sklearn.neighbors.KDTree or str
            kdtree containing latitude-longitude pairs for quick lookups
            Generate using ``pvdeg.geospatial.meta_KDTree``. Can take a pickled
            kdtree as a path to the .pkl file.
        feature_name : str
            cartopy.feature.NaturalEarthFeature feature key.
            Options: ``'lakes'``, ``'rivers_lake_centerlines'``, ``'coastline'``
        resolution : str
            cartopy.feature.NaturalEarthFeature resolution.
            Options: ``'10m'``, ``'50m'``, ``'110m'``
        radius : float
            Area around feature coordinates to include in the downsampled result.
            Bigger area means larger radius and more samples included.
            pass

        Returns:
        --------
        None, strictly updates meta_data attribute of scenario.

        See Also:
        ---------
        `pvdeg.geospatial.feature_downselect`
        """

        feature_gids = pvdeg.geospatial.feature_downselect(
            meta_df=self.meta_data,
            kdtree=kdtree,
            feature_name=feature_name,
            resolution=resolution,
            radius=radius,
            bbox_kwarg=bbox_kwarg,
        )

        self.meta_data[feature_name] = (self.meta_data.index).isin(feature_gids)
        return

    def downselect_elevation_stochastic(
        self,
        kdtree,
        downselect_prop,
        k_neighbors=3,
        method="mean",
        normalization="linear",
    ):
        """
        Prefenetially downselect data points based on elevation and update
        scenario metadata.

        Parameters:
        -----------
        kdtree : sklearn.neighbors.KDTree or str
            kdtree containing latitude-longitude pairs for quick lookups
            Generate using ``pvdeg.geospatial.meta_KDTree``. Can take a pickled
            kdtree as a path to the .pkl file.
        downselect_prop : float
            proportion of original datapoints to keep in output gids list
        k_neighbors : int, (default = 3)
            number of neighbors to check for elevation data in nearest neighbors
        method : str, (default = 'mean')
            method to calculate elevation weights for each point.
            Options : `'mean'`, `'sum'`, `'median'`
        normalization : str, (default = 'linear')
            function to apply when normalizing weights. Logarithmic uses log_e/ln
            options : `'linear'`, `'logarithmic'`, '`exponential'`

        Returns:
        --------
        None

        See Also:
        ---------
        `pvdeg.geospatial.elevation_stochastic_downselect` for more info/docs
        """
        gids = pvdeg.geospatial.elevation_stochastic_downselect(
            meta_df=self.meta_data,
            kdtree=kdtree,
            downselect_prop=downselect_prop,
            k_neighbors=k_neighbors,
            method=method,
            normalization=normalization,
        )

        self.meta_data = self.meta_data.iloc[gids]
        return

    def gid_downsample(self, downsample_factor: int) -> None:
        """
        Downsample the NSRDB GID grid by a factor of n

        Returns:
        --------
        None

        See Also:
        ---------
        `pvdeg.utilities.gid_downsample`
        """
        self.meta_data, _ = utilities.gid_downsampling(
            meta=self.meta_data, n=downsample_factor
        )

    def gids_tonumpy(self) -> np.array:
        """
        Convert the scenario's gids to a numpy array

        Returns:
        --------
        gids : np.array
            all nsrdb gids from the scenario's metadata
        """
        return self.meta_data.index

    def gids_tolist(self) -> np.array:
        """
        Convert the scenario's gids to a python list

        Returns:
        --------
        gids : np.array
            all nsrdb gids from the scenario's metadata
        """
        return list(self.meta_data.index)

    def coords_tonumpy(self) -> np.array:
        """
        Create a tall 2d numpy array of gids of the shape
        ```
        [
            [lat, long],
                ...
            [lat, long]
        ]
        ```
        Returns:
        --------
        coords : np.array
            tall numpy array of lat-long pairs
        """
        coords = np.column_stack(
            self.meta_data["latitude"], self.meta_data["longitude"]
        )

        return coords

    def geospatial_data(self) -> tuple[xr.Dataset, pd.DataFrame]:
        """
        Extract the geospatial weather dataset and metadata dataframe from the scenario object

        Example Use:
        >>> geo_weather, geo_meta = GeospatialScenario.geospatial_data()

        This gets us the result we would use in the traditional pvdeg geospatial approach.

        Parameters:
        -----------
        None

        Returns:
        --------
        (weather_data, meta_data): (xr.Dataset, pd.DataFrame)
            A tuple of weather data as an `xarray.Dataset` and the corresponding meta data as a dataframe.
        """
        # downsample here, not done already happens at pipeline runtime
        geo_weather_sub = self.weather_data.sel(gid=self.meta_data.index).chunk(
            chunks={"time": -1, "gid": 50}
        )
        return geo_weather_sub, self.meta_data

    def addJob(
        self,
        func: Callable = None,
        func_params: dict = {},
        see_added: bool = False,
    ):
        """
        Add a pvdeg function to the scenario pipeline

        Parameters:
        -----------
        func : function
            pvdeg function to use for geospatial analysis.
            *Note: geospatial analysis is only available with a limited subset of pvdeg
            functions*
            Current supported functions for geospatial analysis: ``pvdeg.standards.standoff``,
            ``pvdeg.humidity.module``, ``pvdeg.letid.calc_letid_outdoors``
        func_params : dict
            job specific keyword argument dictionary to provide to the function
        see_added : bool
            set flag to get a userWarning notifying the user of the job added
           to the pipeline in method call. ``default = False``
        """
        # check if we can do geospatial analyis on desired function
        try:
            pvdeg.geospatial.template_parameters(func)
        except ValueError:
            return ValueError(
                f"{func.__name__} does does not have a valid geospatial results template or does not exist"
            )

        geo_job_dict = {"geospatial_job": {"job": func, "params": func_params}}

        # # UNTESTED
        # if func_params:
        #     geo_job_dict.update(func_params)

        self.pipeline = geo_job_dict

        if see_added:
            message = f"{func.__name__} added to pipeline as \n {geo_job_dict}"
            warnings.warn(message, UserWarning)

    def run(self, hpc_worker_conf: Optional[dict] = None) -> None:
        """
        Run the function in the geospatial pipeline.
        GeospatialScenario only supports one geospatial pipeline job at a time
        unlike Scenario which supports unlimited conventional pipeline jobs.
        Results are stored in the `GeospatialScenario.results` attribute.

        Creates a dask cluster or client using the hpc_worker_conf parameter.

        Parameters:
        -----------
        hpc_worker_conf : dict
            Dictionary containing dask hpc settings (see examples below).

            Examples
            --------
            Local cluster:

            .. code-block:: python

                hpc = {'manager': 'local',
                    'n_workers': 1,
                    'threads_per_worker': 8,
                    'memory_limit': '10GB'}

            SLURM cluster:

            .. code-block:: python

                kestrel = {
                    'manager': 'slurm',
                    'n_jobs': 1,  # Max number of nodes used for parallel processing
                    'cores': 104,
                    'memory': '246GB',
                    'account': 'pvsoiling',
                    'walltime': '4:00:00',
                    'processes': 52,
                    'local_directory': '/tmp/scratch',
                    'job_extra_directives': ['-o ./logs/slurm-%j.out'],
                    'death_timeout': 600,}
        """
        client = pvdeg.geospatial.start_dask(hpc=hpc_worker_conf)

        geo_weather_sub = self.weather_data.sel(gid=self.meta_data.index)

        func = self.pipeline["geospatial_job"]["job"]

        if func == pvdeg.standards.standoff or func == pvdeg.humidity.module:
            geo = {
                "func": func,
                "weather_ds": geo_weather_sub,
                "meta_df": self.meta_data,
            }

            analysis_result = pvdeg.geospatial.analysis(**geo)

            self.results = analysis_result

        client.shutdown()

    def restore_result_gids(self):
        """
        Restore gids to result Dataset as datavariable from original metadata.
        Assumes results will be in the same order as input metadata rows.
        Otherwise will fail silently and restore incorrect gids
        """

        flattened = self.results.stack(points=("latitude", "longitude"))

        gids = self.meta_data.index.values

        # Create a DataArray with the gids and assign it to the Dataset
        gids_da = xr.DataArray(gids, coords=[flattened["points"]], name="gids")

        # Unstack the DataArray to match the original dimensions of the Dataset
        gids_da = gids_da.unstack("points")

        self.results = self.results.assign(gids=gids_da)

    def _get_geospatial_data(year: int):
        """
        Helper function. gets geospatial weather dataset and metadata dictionary.

        Parameters
        ----------
        Year : int
            select the year of data to take from the NSRDB

        Returns
        --------
        weather_ds : xarray.Dataset
            dataset with coordinates of gid and time and weather data as datavariables
        meta_df : pd.DataFrame
            dataframe with each row representing the metadata of each gid in the dataset
        """
        weather_db = "NSRDB"

        weather_arg = {
            "satellite": "Americas",
            "names": year,
            "NREL_HPC": True,
            # 'attributes': ['air_temperature', 'wind_speed', 'dhi', 'ghi', 'dni', 'relative_humidity']}
            "attributes": [],  # does having do atributes break anything, should we just pick one
        }

        weather_ds, meta_df = pvdeg.weather.get(
            weather_db, geospatial=True, **weather_arg
        )

        return weather_ds, meta_df

    def getValidRegions(
        self,
        country: Optional[str] = None,
        state: Optional[str] = None,
        county: Optional[str] = None,
        target_region: Optional[str] = None,
    ):
        """
        Gets all valid region names in the NSRDB. Only works on hpc

        Arguments
        ---------
        country : str, optional
        state : str, optional
        country : str, optional
        target_region : str
            Select return field. Options ``country``, ``state``, ``county``.

        Returns
        -------
        valid_regions : numpy.ndarray
            list of strings representing all unique region entries in the nsrdb.
        """

        if not self.geospatial:  # add hpc check
            return AttributeError(
                f"self.geospatial should be True. Current value = {self.geospatial}"
            )

        discard_weather, meta_df = Scenario._get_geospatial_data(year=2022)

        if country:
            meta_df = meta_df[meta_df["country"] == country]
        if state:
            meta_df = meta_df[meta_df["state"] == state]
        if county:
            meta_df = meta_df[meta_df["county"] == county]

        return meta_df[target_region].unique()

    def plot(self):
        """
        Not Usable in GeospatialScenario class instance, only in Scenario instance.
        python has no way to hide a parent class method in the child,
        so this only exists to prevent access
        """
        raise AttributeError(
            "The 'plot' method is not accessible in GeospatialScenario, only in Scenario"
        )

    def plot_coords(
        self,
        coord_1: Optional[tuple[float]] = None,
        coord_2: Optional[tuple[float]] = None,
        coords: Optional[np.ndarray[float]] = None,
        size: Union[int, float] = 1,
    ) -> None:
        """
        Plot lat-long coordinate pairs on blank map. Quickly view
        geospatial datapoints before your analysis.

        Parameters:
        -----------
        coord_1 : list, tuple
            Top left corner of bounding box as lat-long coordinate pair as list or
            tuple.
        coord_2 : list, tuple
            Bottom right corner of bounding box as lat-long coordinate pair in list
            or tuple.
        coords : np.array
            2d tall numpy array of [lat, long] pairs. Bounding box around the most
            extreme entries of the array. Alternative to providing top left and
            bottom right box corners. Could be used to select amongst a subset of
            data points. ex) Given all points for the planet, downselect based on
            the most extreme coordinates for the United States coastline information.
        size : float
            matplotlib scatter point size. Without any downsampling NSRDB
            points will siginficantly overlap.
        """
        fig = plt.figure(figsize=(15, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())

        if (coord_1 and coord_2) or (coords != None):
            utilities._plot_bbox_corners(
                ax=ax, coord_1=coord_1, coord_2=coord_2, coords=coords
            )

        utilities._add_cartopy_features(ax=ax)

        ax.scatter(
            self.meta_data["longitude"],
            self.meta_data["latitude"],
            color="black",
            s=size,
            transform=ccrs.PlateCarree(),
        )

        plt.title(f"Coordinate Pairs from '{self.name}' Meta Data")
        plt.show()

    def plot_meta_classification(
        self,
        col_name: str = None,
        coord_1: Optional[tuple[float]] = None,
        coord_2: Optional[tuple[float]] = None,
        coords: Optional[np.ndarray[float]] = None,
        size: Union[int, float] = 1,
    ):
        """
        Plot classified lat-long coordinate pairs on map. Quicly view
        geospatial datapoints with binary classification in a meta_data
        dataframe column before your analysis.

        Parameters:
        -----------
        col_name : str
            Column containing binary classification data. Ex: `mountain` after
            running ``downselect_mountains_weights``.
        coord_1 : list, tuple
            Top left corner of bounding box as lat-long coordinate pair as list or
            tuple.
        coord_2 : list, tuple
            Bottom right corner of bounding box as lat-long coordinate pair in list
            or tuple.
        coords : np.array
            2d tall numpy array of [lat, long] pairs. Bounding box around the most
            extreme entries of the array. Alternative to providing top left and
            bottom right box corners. Could be used to select amongst a subset of
            data points. ex) Given all points for the planet, downselect based on
            the most extreme coordinates for the United States coastline information.
        size : float
            matplotlib scatter point size. Without any downsampling NSRDB
            points will siginficantly overlap.
        """
        if not col_name:
            raise ValueError("col_name cannot be none")

        if col_name not in self.meta_data.columns:
            raise ValueError(
                f"{col_name} not in self.meta_data columns as follows {self.meta_data.columns}"
            )

        col_dtype = self.meta_data[col_name].dtype
        if col_dtype != bool:
            raise ValueError(
                f"meta_data column {col_name} expected dtype bool not {col_dtype}"
            )

        near = self.meta_data[self.meta_data[col_name] == True]
        not_near = self.meta_data[self.meta_data[col_name] == False]

        fig = plt.figure(figsize=(15, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())

        if (coord_1 and coord_2) or (coords != None):
            utilities._plot_bbox_corners(
                ax=ax, coord_1=coord_1, coord_2=coord_2, coords=coords
            )
        utilities._add_cartopy_features(ax=ax)

        ax.scatter(
            not_near["longitude"],
            not_near["latitude"],
            color="red",
            s=size,
            transform=ccrs.PlateCarree(),
            label=f"Not Near {col_name}",
        )
        ax.scatter(
            near["longitude"],
            near["latitude"],
            color="blue",
            s=size,
            transform=ccrs.PlateCarree(),
            label=f"Near {col_name}",
        )

        plt.title(f"Geographic Points with Proximity to {col_name} Highlighted")
        plt.legend()
        plt.show()

    def plot_world(
        self,
        data_variable: str,
        cmap: str = "viridis",
    ):
        da = (self.results)[data_variable]

        fig, ax = plt.subplots(
            figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()}
        )

        da.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap)
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.gridlines(draw_labels=True)

        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAKES, edgecolor="black")
        plt.show()

    # test this
    def plot_USA(
        self,
        data_from_result: str,
        fpath: str = None,
        cmap: str = "viridis",
        vmin: Union[int, float] = 0,
        vmax: Optional[Union[int, float]] = None,
    ):
        """
        Plot a vizualization of the geospatial scenario result.
        Only works on geospatial scenarios.

        Parameters
        ----------
        data_from_result : str
            select the datavariable to plot from the result xarray
        fpath : str
            path to save plot output on, saves to current directory if ``None``
        cmap : str
            colormap to use in plot
        vmin : int
            lower bound on values in linear color map
        vmax : int
            upper bound on values in linear color map
        """

        if not self.geospatial:
            return False

        fig, ax = pvdeg.geospatial.plot_USA(
            self.results[data_from_result],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title="add_dynamic_title",
            cb_title=f"dynamic title : {data_from_result}",
        )

        fpath if fpath else [f"os.getcwd/{self.name}-{self.results[data_from_result]}"]
        fig.savefig()

    def format_pipeline(self):
        pipeline_html = "<div>"
        if "geospatial_job" in self.pipeline:
            step_name = "geospatial_job"
            step = self.pipeline[step_name]
            params_html = f"<pre>{json.dumps(step['params'], indent=2)}</pre>"

            step_content = f"""
            <div id="{step_name}" onclick="toggleVisibility('pipeline_{step_name}')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_pipeline_{step_name}" style="color: #b676c2;">►</span>
                    {step['job'].__name__}, <span style="color: #b676c2;">#{step_name}</span>
                </h4>
            </div>
            <div id="pipeline_{step_name}" style="display:none; margin-left: 20px; padding: 5px; background-color: #f0f0f0; color: #000;">
                <p>Job: {step['job'].__name__}</p>
                <p>Parameters:</p>
                <div style="margin-left: 20px;">
                    {params_html}
                </div>
            </div>
            """
            pipeline_html += step_content
        pipeline_html += "</div>"
        return pipeline_html

    def _ipython_display_(self):
        file_url = f"file:///{os.path.abspath(self.path).replace(os.sep, '/')}"
        html_content = f"""
        <div style="border:1px solid #ddd; border-radius: 5px; padding: 3px; margin-top: 5px;">
            <h2>{self.name}: Scenario Analysis</h2>
            <p><strong>Path:</strong> <a href="{file_url}" target="_blank">{self.path}</a></p>
            <p><strong>HPC Configuration:</strong> {self.hpc}</p>
            <p><strong>GIDs:</strong> {self.gids}</p>
            <div>
                <h3>Results</h3>
                {self.format_results() if self.results else None}
            </div>
            <div>
                <h3>Pipeline</h3>
                {self.format_pipeline()}
            </div>
            <div>
                <h3>Modules</h3>
                {super().format_modules()}
            </div>
            <div>
                <h3>Weather Dataset</h3>
                {self.format_weather()}
            </div>
            <div>
                <h3>Meta Dataframe</h3>
                {self.format_meta()}
            </div>
        </div>
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

    def format_results(self):
        results_html = "<div>"
        if "geospatial_job" in self.results:
            result = self.results["geospatial_job"]
            result_id = "geospatial_result"
            formatted_output = self.format_output(result)
            result_content = f"""
            <div id="{result_id}" onclick="toggleVisibility('content_{result_id}')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_content_{result_id}" style="color: #b676c2;">►</span>
                    Geospatial Result
                </h4>
            </div>
            <div id="content_{result_id}" style="display:none; margin-left: 20px; padding: 5px; background-color: #f0f0f0; color: #000;">
                {formatted_output}
            </div>
            """
            results_html += result_content
        results_html += "</div>"
        return results_html

    def format_meta(self):
        meta_data_html = ""
        if isinstance(self.meta_data, pd.DataFrame):
            if len(self.meta_data) > 10:
                first_five = self.meta_data.head(5)
                last_five = self.meta_data.tail(5)
                ellipsis_row = pd.DataFrame(
                    [["..."] * len(self.meta_data.columns)],
                    columns=self.meta_data.columns,
                )

                # Create the custom index with ellipses
                custom_index = np.concatenate(
                    [
                        np.arange(0, 5, dtype=object).astype(str),
                        ["..."],
                        np.arange(
                            len(self.meta_data) - 5, len(self.meta_data), dtype=object
                        ).astype(str),
                    ]
                )

                # Concatenate the DataFrames
                display_data = pd.concat(
                    [first_five, ellipsis_row, last_five], ignore_index=True
                )
                display_data.index = custom_index
            else:
                display_data = self.meta_data

            meta_data_html = f"""
            <div id="meta_data" onclick="toggleVisibility('content_meta_data')" style="cursor: pointer; background-color: #000000; color: #FFFFFF; padding: 5px; border-radius: 3px; margin-bottom: 1px;">
                <h4 style="font-family: monospace; margin: 0;">
                    <span id="arrow_content_meta_data" style="color: #b676c2;">►</span>
                    Meta Data
                </h4>
            </div>
            <div id="content_meta_data" style="display:none; margin-left: 20px; padding: 5px; background-color: #f0f0f0; color: #000;">
                {display_data.to_html()}
            </div>
            """
        return meta_data_html
