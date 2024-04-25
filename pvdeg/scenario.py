"""Class to define an analysis scenario."""
from datetime import date
from datetime import datetime as dt
import os
from pvdeg import utilities as utils
import pvdeg
import json
from inspect import signature
from typing import Callable
import warnings
import pandas as pd
from pvlib.location import Location

# TODO: 
# fix .clear(), currently breaks?
# test adding geospatial data to the secenario
# combine getValidCounty, getValidState, getValidCounty into one function
# rename see_added_function, see_added_location flags to just flag or see_added

# SOLVE LATER:
# resolve spillage between class instances
# how do we save the weather at locations (project points)

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
        file=None,
        results=None,
        # are these valuable
        hpc=False,
        geospatial=False,
        weather_data=None, # xarray ds when geospatial
        meta_data = None, # dataframe when geospatial

        email = None, # only for single location
        api_key = None,
        
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
        results : (pd.Series)
            Full collection of outputs from pipeline execution. Populated by ``scenario.runPipeline()``
        """

        if file is not None:
            with open(file, "r") as f:
                data = json.load()
            name = data["name"]
            path = data["path"]
            modules = data["modules"]
            gids = data["gids"]
            pipeline = data["pipeline"]
            # add results to file

        self.name = name
        self.path = path
        self.modules = modules
        self.gids = gids
        self.pipeline = pipeline
        self.results = results
        self.hpc = hpc
        self.geospatial = geospatial
        self.weather_data = weather_data
        self.meta_data = meta_data

        if geospatial and api_key:
            raise ValueError("you cannot use an api key on hpc, NSRDB files exist locally")
        # only if not geospatial
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

    # this could be renamed ``reset`` or ``wipe``?
    # not functional currently (close though)
    def clear(self):
        """
        Wipe the Scenario object. This is useful because the Scenario object stores its data in local files outside of the python script.
        This causes issues when two unique scenario instances are created in the same directory, they appear to be seperate instances
        to python but share the same data (if no path is provided). Changes made to one are reflected in both.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """

        if self:
            # add recursive search check?
            if self.file is not None:
                # may not work properly due to nested nature, need to test
                # only deletes file, not nested directory structure
                os.remove(self.file)
            
            # update attribute
            self.file = None

            # blank scenario object
            self = Scenario

        else:
            raise ValueError(f"cannot clear scenario object: {self}")

    
# add list of strings for county, state, country input so we can have multiples
# add two letter state support, theres a mapping function for these already
# for non geospatial, just use lat_long, ignore weather arg
# add error checks for arg types?
    def addLocation(self, 
        # region=None,
        # region_col="state", 

        # geospatial parameters
        country=None,
        state=None,
        county=None,
        year=2022,
        downsample_factor=0,
        # gids=None, 

        lat_long=None, 
        see_added_location = False
    ):
        """
        Add a location to the scenario. Generates "gids.csv" and saves the file path within
        Scenario dictionary. This can be done in three ways: Pass (region, region_col) for gid list,
        pass (gid) for a single location, pass (lat, long) for a single location.

        Parameters:
        -----------
        country : str
            country to include from NSRDB (using pvdeg.weather.get() geospatial call)
        state : str
            state or providence to include from NSRDB (using pvdeg.weather.get() geospatial call)
        county : str
            county to include from NSRDB (using pvdeg.weather.get() geospatial call)
        downsample_factor : int
            downsample the weather and metadata attached to the region you have selected. default(0), means no downsampling
        year : int
            year of data to use from NSRDB, default = 2022

        lat_long : (tuple - float)
            latitute and longitude of a single location

        see_added_location : bool
            flag true if you want to see a runtime notification for added location/gids

        weather_fp : (str, path_obj, pd.Dataframe)  
            File path to the source dataframe for weather and spatial data. Default should be NSRDB. 
            Additionally, takes weather dataframe generated by ``pvdeg.weather.get()`` for a single location.
        """

        if self.gids is not None:
            print(
                "Scenario already has designated project points.\nNothing has been added."
            )
            print(self.gids)
            return

        # untested
        if not self.geospatial:
            weather_db = 'PSM3' # wrong type, move this to func arg
            # check if lat long is a tuple of 2 floats, may not be a nessecary check
            if isinstance(lat_long, tuple) and all(isinstance(item, float) for item in lat_long) and len(lat_long) == 2:
                weather_id = lat_long
            else:
                return ValueError(f"arg: lat_long is type = {type(lat_long)}, must be tuple(float)")

            # change email and api key???
            weather_arg = {'api_key': 'DEMO_KEY',
                        'email': 'user@mail.com',
                        'names': 'tmy',
                        'attributes': [],
                        'map_variables': True}

            point_weather, point_meta = pvdeg.weather.get(weather_db, weather_id, **weather_arg)

            try:
                # gid type = str when reading from meta
                gid = point_meta['Location ID']            
            except KeyError:
                return UserWarning(f"metadata missing location ID")

            # added location, for single point, this may be misleading
            self.gids = gid
            self.weather_data[int(gid)] == point_weather
            
        # untested
        if self.geospatial:
            # nsrdb_fp = r"/datasets/NSRDB" # kestrel directory, need to update this

            # Get weather data
            weather_db = 'NSRDB'

            weather_arg = {'satellite': 'Americas',
                        'names': year,
                        'NREL_HPC': True,
                        'attributes': ['air_temperature', 'wind_speed', 'dhi', 'ghi', 'dni', 'relative_humidity']}

            geo_weather, geo_meta = pvdeg.weather.get(weather_db, geospatial=True, **weather_arg)

            # downselect 
            if country:
                geo_meta = geo_meta[geo_meta['country'] == country]
            if state:
                geo_meta = geo_meta[geo_meta['state'] == state]
            if county:
                geo_meta = geo_meta[geo_meta['county'] == county]
            
            # if downsample factor is 0, 
            # no downsampling happens but gid_downsampling() generates gids
            geo_meta, geo_gids = pvdeg.utilities.gid_downsampling(geo_meta, downsample_factor) 

            # apply downsampling to weather data
            geo_weather_sub = geo_weather.sel(gid=geo_meta.index)

            self.weather_data = geo_weather_sub
            self.meta_data = geo_meta # already downselected, bad naming?
            self.gids = geo_gids

        if see_added_location:
            message = f"Gids Added - {self.gids}"
            warnings.warn(message, UserWarning)

        # redo writing gids logic
        file_name = f"gids_{self.name}"

        # gids_path = utils.write_gids(
        #     # nsrdb_fp,
        #     region=region,
        #     region_col=region_col,
        #     lat_long=lat_long,
        #     gids=gids,
        #     out_fn=file_name,
        # )


    def addModule(
        self,
        module_name,
        racking="open_rack_glass_polymer",  # move ?? split RACKING_CONSTRUCTION
        material="EVA",
        temperature_model='sapm'
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
        temp_model : (str)
            select pvlib temperature models. Options : ``'sapm', 'pvsyst', 'faiman', 'faiman_rad', 'fuentes', 'ross'``
            
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

    # this is kind of a mess, redo using ironpython special printing like for xarray dataset
    def viewScenario(self):
        """
        Print all scenario information currently stored in the scenario instance
        """

        import pprint

        pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)

        if self.name:
            print(f"Name : {self.name}")

        if self.geospatial:
            print("\033[1;32mGEOSPATIAL = True\033[0m")

        if self.pipeline:
            print('Pipeline : ')

            # pipeline is a list of dictionaries, each list entry is one pipeline job
            df_pipeline = pd.json_normalize(self.pipeline)
            print(df_pipeline.to_string()) # should this be display?
        else:
            print("Pipeline : no jobs in pipeline")

        print("Results : ", end='')
        try: 
            # if this throws an error we have not run the pipeline yet
            results = self.results.empty

            print(f"Pipeline results : ")

            for result in self.results:
                if isinstance(result, pd.DataFrame):
                    print(result.to_string())
        except:
            print("Pipeline has not been run")

        # leave this to make sure the others work
        pp.pprint(f"gids : {self.gids}")
        pp.pprint("test modules :")
        for mod in self.modules:
            pp.pprint(mod)
        
        if self.weather_data:
            print(f"scenario weather : {self.weather_data}")
            # pp.pprint(f"sceneario weather : {self.weather_data}")

        return

    def updatePipeline(
        self, 

        func=None, 
        func_params=None,
        see_added_function=False,
        ):
        """
        Add a pvdeg function to the scenario pipeline

        Parameters:
        -----------
        func : function
            pvdeg function to use for single point calculation or geospatial analysis.
            All regular pvdeg functions will work at a single point when ``Scenario.geospatial == False``  
            *Note: geospatial analysis is only available with a limited subset of pvdeg 
            functions*   
            Current supported functions for geospatial analysis: ``pvdeg.standards.standoff``, 
            ``pvdeg.humidity.module``, ``pvdeg.letid.calc_letid_outdoors``
        func_params : dict  
            TODO
        see_added_function : bool
            set flag to get a userWarning notifying the user of the job added  
            to the pipeline in method call. ``default = False``
        """

        if not self.geospatial:
            if func is None or not callable(func):
                print(f'FAILED: Requested function "{func}" not found')
                print("Function has not been added to pipeline.")
                return None

            params_all = dict(signature(func).parameters)

            # this is a bad way of doing it 
            # some values with NONE are still optional
            # causing if statement below to work improperly, 
            reqs = {name: param for name, param in params_all.items() if param.default is None}
            optional = {name: param for name, param in params_all.items() if name not in reqs}

            ### this should be SUPERSET not subset ###
            # this will force it to work BUT may cause some parameters to be missed #
            if not set(func_params.keys()).issubset(set(reqs.keys())):
                print(func_params.keys())
                print(reqs.keys())
                print(f"FAILED: Requestion function {func} did not receive enough parameters")
                print(f"Requestion function: \n {func} \n ---")
                print(f"Required Parameters: \n {reqs} \n ---")
                print(f"Optional Parameters: {optional}")
                print("Function has not been added to pipeline.")
                return None
            
            job_dict = {"job": func, "params": func_params}
            self.pipeline.append(job_dict)
            
            if see_added_function:
                message = f"{func.__name__} added to pipeline as \n {job_dict}"
                warnings.warn(message, UserWarning)

        if self.geospatial:

            # check if we can do geospatial analyis on desired function
            try: 
               pvdeg.geospatial.template_parameters(func)
            except ValueError: 
                return ValueError(f"{func.__name__} does does not have a valid geospatial results template or does not exist")

            # standards.standoff only needs weather, meta, and func
            geo_job_dict = {f"geospatial_job" : func} 

            # UNTESTED
            if func_params:
                geo_job_dict.update(func_params)

            self.pipeline.append(geo_job_dict)

            if see_added_function:
                message = f"{func.__name__} added to pipeline as \n {geo_job_dict}"
                warnings.warn(message, UserWarning)

    # TODO: run pipeline on each module added (if releveant)
    # may only work with one geospatial job in the pipeline
    def runPipeline(self):
        """
        Runs entire pipeline on scenario object
        """

        # maybe roundabout approach, be careful of type of nested dict value, may be df or other, 
        results_series = pd.Series(dtype='object')

        if not self.geospatial:
            results_dict = {}

            for job in self.pipeline:
                _func = job['job']
                _params = job['params']
                result = _func(**_params)

                results_dict[job['job'].__name__] = result

                # move standard results to single dictionary
                pipeline_results = results_dict
        
            # all job results -> pd.Series[pd.DataFrame]
            for key in pipeline_results.keys():
                print(f"results_dict dtype : {type(results_dict[key])}")
                print(results_dict)

                # properly store dataframe in results series
                if isinstance(results_dict[key], pd.DataFrame):
                    results_series[key] = results_dict[key]

                # implement instance check and convert to df for other return types
                # should all be stored as dataframes in results
                self.results = results_series

        if self.geospatial:
            
            for job in self.pipeline:
                func = job['geospatial_job']

                # shared case, letid may be identical
                if func == pvdeg.standards.standoff or func == pvdeg.humidity.module:
                    geo = {
                        'func': func,
                        'weather_ds': self.weather_data,
                        'meta_df': self.meta_data
                        }
               
                    # update geo dictionary with kwargs from job
                    # geo.update(job[''])

                    analysis_result = pvdeg.geospatial.analysis(**geo)

                    # store xr.ds in pd.Series (might be goofy)
                    # only works for one geospatial analysis per scenario instance
                    self.results = analysis_result

           
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

    def _get_geospatial_data(year : int):
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
        weather_db = 'NSRDB'

        weather_arg = {'satellite': 'Americas',
                    'names': year,
                    'NREL_HPC': True,
                    # 'attributes': ['air_temperature', 'wind_speed', 'dhi', 'ghi', 'dni', 'relative_humidity']}
                    'attributes' : [], # does having do atributes break anything, should we just pick one
                    }

        weather_ds, meta_df = pvdeg.weather.get(weather_db, geospatial=True, **weather_arg)

        return weather_ds, meta_df

    def getValidCountries():
        """
        Gets all valid of country names in the NSRDB in the year 2022

        Returns
        -------
        valid_countries : numpy.ndarray
            list of strings representing all unique country entries in the NSRDB
        """
        discard_weather, meta_df = Scenario._get_geospatial_data(year=2022)

        return meta_df['country'].unique()

    def getValidStates(
        country : str = None
        ):
        """
        Gets all valid of state/province names in the NSRDB in the year 2022

        Parameters
        ----------
        country : str
            select target country to view states/provices in

        Returns
        -------
        valid_states : numpy.ndarray
            list of strings representing all unique state entries in the NSRDB
        """
        discard_weather, meta_df = Scenario._get_geospatial_data(year=2022)

        # probably a cleaner one liner to do this
        if country:
            meta_df=meta_df[meta_df['country'] == country]
        
        return meta_df['state'].unique()

    def getValidCounties(
        country : str = None,
        state : str = None
        ):
        """
        Gets all valid of county names in the NSRDB in the year 2022

        Parameters
        ----------
        country : str
            target country to view states/provices in
        state : str
            target state to view counties in 

        Returns
        -------
        valid_states : numpy.ndarray
            list of strings representing all unique county entries in the NSRDB
        """
        discard_weather, meta_df = Scenario._get_geospatial_data(year=2022)

        # probably a cleaner one liner to do this
        if country:
            meta_df=meta_df[meta_df['country'] == country]
        if state:
            meta_df=meta_df[meta_df['state'] == state]

        return meta_df['county'].unique()

    def dump(
        self,
        attribute,
        ):
        """
        Dump desired class attributes to a file (what type???) previously using json, netcdf?
        Parameters
        ----------
        attribute : str or list[str]
            attribute names to save to csv file
        """

        pass

