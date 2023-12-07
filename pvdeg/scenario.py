"""Class to define an analysis scenario.
"""
from datetime import date
from datetime import datetime as dt
import os
from pvdeg import utilities as utils
import pvdeg
import json

# TODO: add functions...


class Scenario:
    """
    The scenario object contains all necessary parameters and criteria for a given scenario.
    Generally speaking, this will be information such as:
    Scenario Name, Path, Geographic Location, Module Type, Racking Type
    """

    def __init__(
        self,
        name=None,
        path=None,
        gids=None,
        modules=[],
        pipeline=[],
        hpc=False,
        file=None,
    ) -> None:
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
        """

        if file is not None:
            with open(file, "r") as f:
                data = json.load()
            name = data["name"]
            path = data["path"]
            modules = data["modules"]
            gids = data["gids"]
            pipeline = data["pipeline"]

        self.name = name
        self.path = path
        self.modules = modules
        self.gids = gids
        self.pipeline = pipeline

        filedate = dt.strftime(date.today(), "%d%m%y")

        if name is None:
            name = filedate
        self.name = name

        if path is None:
            self.path = os.path.join(os.getcwd(), f"pvd_job_{self.name}")
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        os.chdir(self.path)

    def addLocation(
        self, weather_fp=None, region=None, region_col="state", lat_long=None, gids=None
    ):
        """
        Add a location to the scenario. Generates "gids.csv" and saves the file path within
        Scenario dictionary. This can be done in three ways: Pass (region, region_col) for gid list,
        pass (gid) for a single location, pass (lat, long) for a single location.

        Parameters:
        -----------
        weather_fp : (str, path_obj)
            File path to the source dataframe for weather and spatial data. Default should be NSRDB
        region : (str)
            Region or state to iterate over
        region_col : (str)
            Region column name within h5 file (example "State")
        lat : (tuple - float)
            latitute and longitude of a single location
        """

        if self.gids is not None:
            print(
                "Scenario already has designated project points.\nNothing has been added."
            )
            print(self.gids)
            return

        if not weather_fp:
            weather_fp = r"/datasets/NSRDB/current/nsrdb_tmy-2021.h5"

        file_name = f"gids_{self.name}"
        gids_path = utils.write_gids(
            weather_fp,
            region=region,
            region_col=region_col,
            lat_long=lat_long,
            gids=gids,
            out_fn=file_name,
        )

        self.gids = gids_path
        print(f"Location Added - {self.gids}")

    def addModule(
        self,
        module_name,
        racking="open_rack_glass_polymer",  # move ?? split RACKING_CONSTRUCTION
        material="EVA",
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
            Name of the material desired. For a complete list, see data/materials.json.
            To add a custom material, see pvdeg.addMaterial (ex: EVA, Tedlar)
        """

        # fetch material parameters (Eas, Ead, So, etc)
        try:
            mat_params = utils._read_material(name=material)
        except:
            print("Material Not Found - No module added to scenario.")
            print("If you need to add a custom material, use .add_material()")
            return

        # remove module if found in instance list
        for i in range(self.modules.__len__()):
            if self.modules[i]["module_name"] == module_name:
                print(f'WARNING - Module already found by name "{module_name}"')
                print("Module will be replaced with new instance.")
                self.modules.pop(i)

        # generate temperature model params
        # TODO: move to temperature based functions
        # temp_params = TEMPERATURE_MODEL_PARAMETERS[model][racking]

        # add the module and parameters
        self.modules.append({"module_name": module_name, "material_params": mat_params})
        print(f'Module "{module_name}" added.')

    def add_material(
        self, name, alias, Ead, Eas, So, Do=None, Eap=None, Po=None, fickian=True
    ):
        """
        add a new material type to master list
        """
        utils._add_material(
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
        Print all scenario information currently stored
        """

        import pprint

        pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
        print(f"Name : {self.name}")
        print(f"pipeline: {self.pipeline}")
        print(f"gid file : {self.gids}")
        print("test modules :")
        for mod in self.modules:
            pp.pprint(mod)
        return

    def addFunction(self, func_name=None, func_params=None):
        """
        Add a pvdeg function to the scenario pipeline

        TODO: list public functions if no func_name given or bad func_name given

        Parameters:
        -----------
        func_name : (str)
            The name of the requested pvdeg function. Do not include the class.
        func_params : (dict)
            The required parameters to run the requested pvdeg function

        Returns:
        --------
        func_name : (str)
            the name of the pvdeg function requested
        """

        _func, reqs = pvdeg.Scenario._verify_function(func_name)

        if _func == None:
            print(f'FAILED: Requested function "{func_name}" not found')
            print("Function has not been added to pipeline.")
            return None

        if not all(x in func_params for x in reqs):
            print(
                f"FAILED: Requestion function {func_name} did not receive enough parameters"
            )
            print(f"Requestion function: \n {_func} \n ---")
            print(f"Required Parameters: \n {reqs} \n ---")
            print("Function has not been added to pipeline.")
            return None

        # add the function and arguments to pipeline
        job_dict = {"job": func_name, "params": func_params}

        self.pipeline.append(job_dict)
        return func_name

    def runJob(self, job=None):
        """
        Run a named function on the scenario object

        TODO: overhaul with futures/slurm
              capture results
              standardize result format for all of pvdeg

        Parameters:
        -----------
        job : (str, default=None)
        """
        if self.hpc:
            # do something else
            pass

        for job in self.pipeline:
            args = job["parameters"]
            _func = pvdeg.Scenario._verify_function(job["job"], args)[0]
            result = _func(**args)

    def exportScenario(self, file_path=None):
        """
        Export the scenario dictionaries to a json configuration file

        TODO exporting functions as name string within pipeline. cannot .json dump <pvdeg.func>
             Need to make sure name is verified > stored > export > import > re-verify > converted.
             This could get messy. Need to streamline the process or make it bullet proof

        Parameters:
        -----------
        file_path : (str, default = None)
            Desired file path to save the scenario.json file
        """

        if not file_path:
            file_path = self.path
        file_name = f"config_{self.name}.json"
        out_file = os.path.join(file_path, file_name)

        scene_dict = {
            "name": self.name,
            "path": self.path,
            "pipeline": self.pipeline,
            "gid_file": self.gids,
            "test_modules": self.modules,
        }

        with open(out_file, "w") as f:
            json.dump(scene_dict, f, indent=4)
        print(f"{file_name} exported")

    def importScenario(self, file_path=None):
        """
        Import scenario dictionaries from an existing 'scenario.json' file
        """

        with open(file_path, "r") as f:
            data = json.load()
        name = data["name"]
        path = data["path"]
        modules = data["modules"]
        gids = data["gids"]
        pipeline = data["pipeline"]

        self.name = name
        self.path = path
        self.modules = modules
        self.gids = gids
        self.pipeline = pipeline

    def _verify_function(func_name):
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
