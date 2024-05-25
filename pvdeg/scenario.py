"""Class to define an analysis scenario."""
from datetime import date
from datetime import datetime as dt
import os
from shutil import rmtree
from pvdeg import utilities as utils
import pvdeg
import json
from inspect import signature
import warnings
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

import pprint
from IPython.display import display, HTML


# TODO: 
### UPDATE SAVING SINGLE LOCATION SCENARIO TO FILES, single pipeline job is not correct, we cant save weather dataframe to csv in the parameters column of csv
### RESTORE SCENARIO FROM pvd_job DIR in constructor ###
### dynamic plotting function
### need to use package wide uniform naming scheme for scenario analysis to work with all funcs

# @ martin
# how should we handle multiple of the same function, currently names dict key with the address added but this isnt clear and will become a problem quickly
### ANS: use random strings or some other identifier like a memory address

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

        hpc=False,
        geospatial=False,
        weather_data=None, # df when single, xr.ds when geospatial
        meta_data = None, # dict when single, df when geospatial

        email = None, 
        api_key = None, 
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

        # if file is not None:
        #     with open(file, "r") as f:
        #         data = json.load()
        #     name = data["name"]
        #     path = data["path"]
        #     modules = data["modules"]
        #     gids = data["gids"]
        #     pipeline = data["pipeline"]
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
        self.lat_long = None

        if geospatial and api_key:
            raise ValueError("Cannot use an api key on hpc, NSRDB files exist locally")
        self.api_key = api_key # only if not geospatial
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
            self.importScenario(file_path=file)
        

    def clean(self):
        """
        Wipe the Scenario object filetree. This is useful because the Scenario object stores its data in local files outside of the python script.
        This causes issues when two unique scenario instances are created in the same directory, they appear to be seperate instances
        to python but share the same data (if no path is provided). Changes made to one are reflected in both.

        Parameters:
        -----------
        None

       """
        if self.path:
            try:
                os.chdir(os.pardir) 
                rmtree(path=self.path) 
            except:
                raise FileNotFoundError(f"cannot remove {self.name} directory")
        else:
            raise ValueError(f"{self.name} does not have a path attribute")
   
    def addLocation(self, 
        # geospatial parameters
        country=None,
        state=None,
        county=None,
        year=2022,
        downsample_factor=0,
        nsrdb_attributes=['air_temperature', 'wind_speed', 'dhi', 'ghi', 'dni', 'relative_humidity'],
        # gids=None, 

        lat_long=None, 
        see_added = False,
    ):
        """
        Add a location to the scenario. Generates "gids.csv" and saves the file path within
        Scenario dictionary. This can be done in three ways: Pass (region, region_col) for gid list,
        pass (gid) for a single location, pass (lat, long) for a single location.

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
        lat_long : (tuple - float)
            latitute and longitude of a single location

        see_added : bool
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
            weather_db = 'PSM3' # should this be PSM3

            if isinstance(lat_long, tuple) and all(isinstance(item, float) for item in lat_long) and len(lat_long) == 2:
                weather_id = lat_long
                self.lat_long = lat_long # save coordinate
            else:
                return ValueError(f"arg: lat_long is type = {type(lat_long)}, must be tuple(float)")

            try:
                weather_arg = {
                    'api_key': self.api_key,
                    'email': self.email,
                    'names': 'tmy',
                    'attributes': [],
                    'map_variables': True}
            except:
                raise ValueError(f"email : {self.email} \n api-key : {self.api_key} \n Must provide an email and api key during class initialization")

            point_weather, point_meta = pvdeg.weather.get(weather_db, id=weather_id, **weather_arg)

            try:
                # gid type of str when reading from meta
                gid = point_meta['Location ID']            
            except KeyError:
                return UserWarning(f"metadata missing location ID")

            # TODO: calculate gid using rex.NSRDBX.lat_long_gis, will only work on hpc
            if self.hpc:
                pass

            # gid for a single location, may be misleading, should confirm psm3 location id vs nsrdb gid 
            self.gids = [ int(gid) ]
            self.meta_data = point_meta
            self.weather_data = point_weather # just save as a dataframe, give xarray option?
            
        if self.geospatial:
            # nsrdb_fp = r"/datasets/NSRDB" # kestrel directory

            # Get weather data
            weather_db = 'NSRDB'
            weather_arg = {'satellite': 'Americas',
                        'names': year,
                        'NREL_HPC': True,
                        'attributes': nsrdb_attributes}

            geo_weather, geo_meta = pvdeg.weather.get(weather_db, geospatial=True, **weather_arg)

            # string to list whole word list or keep list
            toList = lambda s : s if isinstance(s, list)  else [s]
            
            # downselect 
            if country:
                countries = toList(country)
                geo_meta = geo_meta[geo_meta['country'].isin( countries )]
            if state:
                states = toList(state)
                # convert 2-letter codes to full state names
                states = [pvdeg.utilities._get_state(entry) if len(entry) == 2
                        else entry 
                        for entry in states]
                geo_meta = geo_meta[geo_meta['state'].isin( states )]
            if county:
                counties = toList(county)
                geo_meta = geo_meta[geo_meta['county'].isin( counties )]
            
            # if factor is 0 generate gids anyway 
            # no downsampling happens but gid_downsampling() generates gids
            geo_meta, geo_gids = pvdeg.utilities.gid_downsampling(geo_meta, downsample_factor) 

            # apply downsampling to weather data
            geo_weather_sub = geo_weather.sel(gid=geo_meta.index)

            self.weather_data = geo_weather_sub
            self.meta_data = geo_meta # already downselected, bad naming?
            self.gids = geo_gids

        if see_added:
            message = f"Gids Added - {self.gids}"
            warnings.warn(message, UserWarning)

        # redo writing gids logic
        # copied from utils.write_gids (not sure if the current function implementation is what we want)
        file_name = f"gids_{self.name}.csv"
        df_gids = pd.DataFrame(self.gids, columns=["gid"])
        df_gids.to_csv(file_name, index=False)

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
        module_name:str = None,
        racking:str="open_rack_glass_polymer",  
        material:str="EVA",
        temperature_model:str='sapm',
        model_kwarg:dict={},
        irradiance_kwarg:dict={},
        see_added:bool=False,
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
            select pvlib temperature models. See ``pvdeg.temperature.temperature`` for more.
            Options : ``'sapm', 'pvsyst', 'faiman', 'faiman_rad', 'fuentes', 'ross'``
        model_kwarg : (dict), optional
            provide a dictionary of temperature model coefficents to be used 
            instead of pvlib defaults. Some models will require additional \n
            arguments such as ``ross`` which requires nominal operating cell \n
            temperature (``noct``). This is where other values such as noct \n
            should be provided.
            Pvlib temp models: 
            https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html
        irradiance_kwarg : (dict), optional
            provide keyword arguments for poa irradiance calculations.
            Options : ``sol_position``, ``tilt``, ``azimuth``, ``sky_model``
        see_added : (bool), optional
        """

        # fetch material parameters (Eas, Ead, So, etc)
        try:
            mat_params = utils._read_material(name=material)
        except:
            print("Material Not Found - No module added to scenario.")
            print("If you need to add a custom material, use .add_material()")
            return

        # remove module if found in instance list
        old_modules = [mod['module_name'] for mod in self.modules]
        if module_name in old_modules:
            print(f'WARNING - Module already found by name "{module_name}"')
            print("Module will be replaced with new instance.")
            self.modules.pop(old_modules.index(module_name))
 
        # add the module and parameters
        self.modules.append({
            "module_name": module_name, 
            "racking" : racking,
            "material_params": mat_params, 
            "temp_model" : temperature_model,
            "model_kwarg" : model_kwarg,
            "irradiance_kwarg" : irradiance_kwarg,
            })

        if see_added:
            print(f'Module "{module_name}" added.')

    # test this?
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
        Print all scenario information currently stored in the scenario instance. 
        Does not implement ipython.display. If available, use this.
        """
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
        
        # can't check if dataframe is empty
        if isinstance(self.weather_data, (pd.DataFrame, xr.Dataset)):
            print(f"scenario weather : {self.weather_data}")

    def addJob(
        self, 
        func=None, 
        func_params={},
        see_added=False,
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
        see_added : bool
            set flag to get a userWarning notifying the user of the job added  
            to the pipeline in method call. ``default = False``
        """

        if not self.geospatial:
            if func is None or not callable(func):
                print(f'FAILED: Requested function "{func}" not found')
                print("Function has not been added to pipeline.")
                return 

            params_all = dict(signature(func).parameters)

            # cant check parameters needed by function now bc they come from multiple sources when the pipeline is run.

            job_dict = {"job": func, "params": func_params}
            self.pipeline.append(job_dict)
           
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

        if see_added:
            if self.geospatial:
                message = f"{func.__name__} added to pipeline as \n {geo_job_dict}"
                warnings.warn(message, UserWarning)

            elif not self.geospatial:
                message = f"{func.__name__} added to pipeline as \n {job_dict}"
                warnings.warn(message, UserWarning)
        
        # grabs fully qualified function name from function address
        get_qualified = lambda x : f"{x.__module__}.{x.__name__}"

        df_pipeline = pd.DataFrame(self.pipeline)
        # update the first column of the dataframe from func address to function name 
        df_pipeline.iloc[:,0] = df_pipeline.iloc[:,0].map(get_qualified)
        
        file_name = f"pipeline_{self.name}.csv"
        df_pipeline.to_csv(file_name, index=False)

    def run(self, hpc=None):
        """
        Run all jobs in pipeline on scenario object. If single location,
        run multiple jobs on multiple pannels or any combination. 
        If geospatial, can only run one job from the pipeline. Having more
        than one geospatial job in the pipeline may throw an error.

        Parameters:
        -----------
        hpc : (dict), optional, default=None
            Only for geospatial analysis.
            dictionary of parameters for dask client intialization. 
            See ``pvdeg.geospatial.start_dask`` for more information.
        """
        results_series = pd.Series(dtype='object')

        if not self.geospatial:
            results_dict = {}

            # we need do the pipeline for every module available
            if self.modules:

                for module in self.modules: 
                    module_result = {}
                    
                    for job in self.pipeline:
                        func, params = job['job'], job['params']

                        weather_dict = {'weather_df' : self.weather_data, 'meta' : self.meta_data} # move outside? doesn't need to be in here, cleaner though?

                        temperature_args = {
                            'temp_model' : module['temp_model'],
                            'model_kwarg' : module['model_kwarg'], # if func parameters are named incorrecly, ex model_kwargs instead of model_kwarg, this will not work properly
                            'irradiance_kwarg' : module['irradiance_kwarg'], # same for irradiance kwarg, is there a better way to do this then?
                            'conf' : module['racking'],
                            **module['irradiance_kwarg'] # some functions dont want irradiance_kwargs arg instead they just want the irradiance_kwargs values as kwargs
                        }

                        combined = {**weather_dict, **temperature_args, **module['material_params']} # maybe should not have material params like this, idk what functions need them which changes where they should be implemented

                        func_params = signature(func).parameters
                        func_args = {k:v for k,v in combined.items() if k in func_params.keys()} # downselect, only keep arguments that the function will take
                        # downselecting too many things here? we seem to be removing information we want like irradiance kwargs, this will happen if function doesnt take irradiance kwargs

                        # loosing the irradiance kwargs and probably model kwargs here?
                        res = func(**params, **func_args) # provide user args and module specific args

                        if func.__name__ not in module_result.keys(): # this is terrible
                            module_result[func.__name__] = res
                        else:
                            module_result[f"{func.__name__}_{id(func)}"] = res

                   # move standard results to single dictionary
                    results_dict[module['module_name']] = module_result

                self.results = results_dict # 2d dictionary array

                for module, pipeline_result in self.results.items():
                    module_dir = f"pipeline_results/{module}_pipeline_results"
                    os.makedirs(module_dir, exist_ok=True)
                    for function, result in pipeline_result.items():
                        if isinstance(result, (pd.Series, pd.DataFrame)):
                            result.to_csv(f"{module_dir}/{function}.csv")                            
                        elif isinstance(result, (int, float)):
                            with open(f"{module_dir}/{function}.csv", 'w') as file:
                                file.write(f"{result}\n")

            # REFACTOR??? this is really bad 
            # no modules case, all funcs will use default values from functions for module information
            elif not self.modules:
                for job in self.pipeline:
                    _func, _params = job['job'], job['params']

                    # if we do this we will need to enforce parameter naming scheme repo wide
                    try:
                        # try to populate with weather and meta
                        # could fail if function signature has wrong names or if we
                        # haven't added a location, can provide weather and meta in
                        # a kwargs argument

                        # if 'weather_df' not in _params.keys(): # make sure they havent provided weather in the job arguments
                        _func = partial(
                            _func, 
                            weather_df=self.weather_data, 
                            meta=self.meta_data
                            )
                    except:
                        pass

                    result = _func(**_params) if _params else _func()

                    if job['job'].__name__ not in results_dict.keys():
                        results_dict[job['job'].__name__] = result
                    else:
                        results_dict[f"{job['job'].__name__}_{id(job['job'])}"] = result # this is spagheti

                    # move standard results to single dictionary
                    pipeline_results = results_dict
            
                # TEST THIS NOW? AND REDO result series logic?
                # save all results to dataframes and store in a series
                for key in pipeline_results.keys():
                    print(f"results_dict dtype : {type(results_dict[key])}")
                    print(results_dict)

                    if isinstance(results_dict[key], pd.DataFrame):
                        results_series[key] = results_dict[key]

                    elif isinstance(results_dict[key], (float, int)):
                        results_series[key] = pd.DataFrame(
                            [ results_dict[key] ], # convert the single numeric to a list
                            columns=[key] # name the single column entry in list form
                            )

                    self.results = results_series

        if self.geospatial:

            # TODO : 
            # move dask client intialization? 
            # add check to see if there is already a dask client running
            pvdeg.geospatial.start_dask(hpc=hpc)   

            for job in self.pipeline:
                func = job['geospatial_job']

                # remove check??? the fuction cant be added to the pipeline if it doesn't have a valid output template
                if func == pvdeg.standards.standoff or func == pvdeg.humidity.module:
                    geo = {
                        'func': func,
                        'weather_ds': self.weather_data,
                        'meta_df': self.meta_data
                        }
               
                    # update geo dictionary with kwargs from job
                    # geo.update(job[''])

                    analysis_result = pvdeg.geospatial.analysis(**geo)

                    # bypass pd.Series and store result in result attribute,
                    # only lets us do one geospatial analysis per scenario
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

    def importScenario(self, file_path=None, email=None, api_key=None):
        """
        Import scenario dictionaries from an existing 'scenario.json' file
        """

        with open(file_path, "r") as f:
            data = json.load(f)
        name = data["name"]
        path = data["path"]
        hpc = data['hpc']        
        geospatial = data['geospatial']
        modules = data["modules"]
        gids = data["gids"]
        pipeline = data["pipeline"]
        lat_long = data["lat_long"]

        self.name = name
        self.path = path
        self.hpc = hpc
        self.geospatial = geospatial
        self.modules = modules
        self.gids = gids
        self.pipeline = pipeline
        self.file = file_path
        

        if not geospatial: # test this
            try:
                self.email = data['email']
                self.api_key = data['api_key']
            except KeyError:
                print(f"credentials not in json file using arguments")
                self.email = email
                self.api_key = api_key

            self.addLocation(lat_long=lat_long)
        elif geospatial:
            # self.addLocation(gids=???) 
            pass

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

    def getValidRegions(
        self,
        country : str = None,
        state : str = None,
        county : str = None,
        target_region : str = None,
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
 
        if not self.geospatial: # add hpc check
            return AttributeError(f"self.geospatial should be True. Current value = {self.geospatial}")

        # use rex instead
        discard_weather, meta_df = Scenario._get_geospatial_data(year=2022)

        if country:
            meta_df=meta_df[meta_df['country'] == country]
        if state:
            meta_df=meta_df[meta_df['state'] == state]
        if county:
            meta_df=meta_df[meta_df['county'] == county]
        
        return meta_df[target_region].unique() 

    def _to_dict(self, api_key=False): 
        attributes = {
            'name': self.name,
            'hpc' : self.hpc,
            'geospatial' : self.geospatial,
            'path': self.path,
            'modules': self.modules, # problem? list of dictionaries
            'gids': self.gids,
            'lat_long' : self.lat_long, 
            'pipeline' : self.pipeline,
        }

        if api_key:
            protected = {
                'email' : self.email,
                'api_key' : self.api_key
            }
            
            attributes.update(protected)

        return attributes

    def dump(self, api_key=False, path=None):
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
        target = os.path.join(path, f'{self.name}.json') 

        scenario_as_dict = self._to_dict(api_key)
        scenario_as_json = json.dumps(scenario_as_dict, indent=4) # what is this indent for 

        with open(target, 'w') as f:
            f.write(scenario_as_json)

        return
    
    def restore_credentials(self, email, api_key):
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
        if not self.email and self.api_key:
            self.email = email
            self.api_key = api_key

    def extract(
        self, 
        dim_target, 
        col_name=None, 
        tmy=False, 
        start_time=None, 
        end_time=None
        ):
        if self.results is None:
            raise ValueError(f"No scenario results. Run pipeline with ``.run()``")
        if self.geospatial:
            raise ValueError(f"self.geospatial must be false. only single point plotting currently implemented")

        if not isinstance(dim_target, tuple):
            raise TypeError(f"dim_target is type: {type(dim_target)} must be tuple")
        if len(dim_target) != 2:
            raise ValueError(f"size dim_target={len(dim_target)} must be length 2")

        results = pd.DataFrame()

        if dim_target[0] == 'module':
            sub_dict = self.results[dim_target[1]] 

            for key, value in sub_dict.items():
                if isinstance(value, pd.Series): 
                    results[key] = value
                elif isinstance(value, pd.DataFrame):
                    if col_name is not None:
                        results[key] = value[col_name]
                    else:
                        raise ValueError(f"col_name must be provided for DataFrame extraction")
                
        elif dim_target[0] == 'function':
            for module, sub_dict in self.results.items():
                for function, function_result in sub_dict.items():
                    if dim_target[1] == function:
                        if isinstance(function_result, pd.Series): 
                            results[module] = function_result     
                        elif isinstance(function_result, pd.DataFrame):
                            if col_name is not None:
                                results[module] = function_result[col_name]
                            else:
                                raise ValueError(f"col_name must be provided for DataFrame extraction")

        if tmy:
            results.index = results.index.map(lambda dt: dt.replace(year=1970)) # placeholder year

            if start_time and end_time:
                results = utils.strip_normalize_tmy(results, start_time, end_time)
        
        return results

    def plot(
        self, 
        dim_target, 
        col_name=None, 
        tmy=False, 
        start_time=None, 
        end_time=None,
        title='',
        ):
        """
        create plots of scenario data against a specific dimension quickly. 
        When complete this will be able to plot single location and geospatial
        data.
        Parameters:
        -----------
        see extract
        """

        df = self.extract(dim_target=dim_target,col_name=col_name,tmy=tmy,start_time=start_time,end_time=end_time)

        fig, ax = plt.subplots()
        df.plot(ax=ax)
        ax.set_title(f"{self.name} : {title}")
        plt.show()


    def plot_USA(
        self, 
        data_from_result : str, 
        fpath : str = None, 
        cmap = 'viridis',
        vmin = 0,
        vmax = None,
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

        fig, ax = pvdeg.geospatial.plot_USA(self.results[data_from_result], 
            cmap=cmap, vmin=vmin, vmax=vmax, 
            title='add_dynamic_title', 
            cb_title=f'dynamic title : {data_from_result}'
            )

        fpath if fpath else [f"os.getcwd/{self.name}-{self.results[data_from_result]}"]
        fig.savefig()

    # test on geospatial?
    def _ipython_display_(self):
        file_url = f"file:///{os.path.abspath(self.path).replace(os.sep, '/')}"
        html_content = f"""
        <div style="border:1px solid #ddd; border-radius: 5px; padding: 3px; margin-top: 5px;">
            <h2>{self.name}: Scenario Analysis</h2>
            <p><strong>Path:</strong> <a href="{file_url}" target="_blank">{self.path}</a></p>
            <p><strong>Geospatial Data:</strong> {self.geospatial}</p>
            <p><strong>HPC Configuration:</strong> {self.hpc}</p>
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
        modules_html = '<div>'
        for i, module in enumerate(self.modules):
            material_params_html = f"<pre>{json.dumps(module['material_params'], indent=2)}</pre>"
            model_kwarg_html = f"<pre>{json.dumps(module['model_kwarg'], indent=2)}</pre>"
            irradiance_kwarg_html = f"<pre>{json.dumps(module['irradiance_kwarg'], indent=2)}</pre>"

            module_content = f"""
            <div onclick="toggleVisibility('module_{i}')" style="cursor: pointer; background-color: #7a736c; padding: 1px; border-radius: 1px; margin-bottom: 1px;">
                <h4><span id="arrow_module_{i}">►</span> {module['module_name']}</h4>
            </div>
            <div id="module_{i}" style="display:none; margin-left: 20px; padding: 5px;">
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
        modules_html += '</div>'
        return modules_html

    def format_results(self):
        results_html = '<div>'
        for module_name, functions in sorted(self.results.items()):
            module_id = f"result_module_{module_name}" # 5, 3, 2 ebfore
            module_content = f"""
            <div onclick="toggleVisibility('{module_id}')" style="cursor: pointer; background-color: #a0a88c; padding: 1px; border-radius: 1px; margin-bottom: 1px;">
                <h4><span id="arrow_{module_id}">►</span> {module_name}</h4>
            </div>
            <div id="{module_id}" style="display:none; margin-left: 20px; padding: 5px;">
            """
            for function_name, output in functions.items():
                function_id = f"{module_id}_{function_name}"
                formatted_output = self.format_output(output)
                module_content += f"""
                <div onclick="toggleVisibility('{function_id}')" style="cursor: pointer; background-color: #849654; padding: 1px; border-radius: 1px; margin-bottom: 1px;">
                    <h5><span id="arrow_{function_id}">►</span> {function_name}</h5>
                </div>
                <div id="{function_id}" style="display:none; margin-left: 20px; padding: 5px;">
                    {formatted_output}
                </div>
                """
            module_content += '</div>'
            results_html += module_content
        results_html += '</div>'
        return results_html

    def format_output(self, output):
        if isinstance(output, pd.Series):
            output = pd.DataFrame(output) # cant display series as html so go to dataframe first
            
        if isinstance(output, pd.DataFrame):
            # Display first 10 and last 10 entries
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
                ellipsis_row = pd.DataFrame(["..."] * len(self.weather_data.columns)).T
                ellipsis_row.columns = self.weather_data.columns
                display_data = pd.concat([first_five, ellipsis_row, last_five], ignore_index=True)
            else:
                display_data = self.weather_data

            weather_data_html = f"""
            <div onclick="toggleVisibility('weather_data')" style="cursor: pointer; background-color: #7a736c; padding: 1px; border-radius: 1px; margin-bottom: 1px;">
                <h4><span id="arrow_weather_data">►</span> Weather Data</h4>
            </div>
           <div id="weather_data" style="display:none; margin-left: 20px; padding: 5px;">
                {display_data.to_html()}
            </div>
            """
        
        return weather_data_html

    def format_pipeline(self):
        pipeline_html = '<div>'
        for i, step in enumerate(self.pipeline):
            params_html = f"<pre>{json.dumps(step['params'], indent=2)}</pre>"

            step_content = f"""
            <div onclick="toggleVisibility('pipeline_{i}')" style="cursor: pointer; background-color: #cc512f; padding: 1px; border-radius: 1px; margin-bottom: 1px;">
                <h4><span id="arrow_pipeline_{i}">►</span> {step['job'].__name__}</h4>
            </div>
            <div id="pipeline_{i}" style="display:none; margin-left: 20px; padding: 5px;">
                <p>Job: {step['job'].__name__}</p>
                <p>Parameters:</p>
                <div style="margin-left: 20px;">
                    {params_html}
                </div>
            </div>
            """
            pipeline_html += step_content
        pipeline_html += '</div>'
        return pipeline_html