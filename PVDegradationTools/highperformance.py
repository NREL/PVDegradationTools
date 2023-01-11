"""
PVDegradationTools.HighPerformance
-----------------------------------

DESCRIPTION
------------
submit SLURM job requests using functions from PVD Tools
This module will only work on the HPC
Currently the method can only support a single function, so operations such as:
StressFactors.rh_backsheet() are not accepted as they require too many substeps

TODO:
 [x] output file name and path
 [x] output file structure
 [ ] shell/slurm integration
 [ ] test with ideal_installation_distance
 [ ] test with k, water_vapor_pressure, etc
"""
import os
from datetime import date
from datetime import datetime as dt
from configparser import ConfigParser as CP
import pandas as pd
from rex import NSRDBX
from dask.distributed import Client
import PVDegradationTools as PVD

class HighPerformance:
    '''
    The HighPerformance class is a container for all parameters necessary to iterate over spacial
    arguments and perform a single calculation

    Examples:
        - calculate installation distance for a dozen locations in the US.
          PVD.HighPerformance.submit('ideal_installation_distance', pvd_job.ini)
    '''

    def _read_ini(file):
        '''
        Reads the .ini file containing job iterables and parameters. See the provided
        template "pvd_job.ini" for the specific requirements when writing this file.

        Parameters
        -----------
        file: string
            file path to the .ini

        Returns
        -----------
        param_dict : dictionary
            parsed multi-level dictionary of function requested, HPC user, iterables,
            and function parameters.
        '''

        config = CP()
        config.read(file)

        if 'job' not in config.sections():
            print('Invalid configuration passed')
            return None

        param_dict = {}
        for s in config.sections():
            param_dict[s] = {}
            for o in config.options(s):
                param_dict[s][o] = config.get(s,o)

        return param_dict


    def _gen_weather(iterables):
        '''
        Generate the weather dataframe for simulations
        
        Parameters
        -----------
        iterables : dictionary
            python dictionary containing the spatial iteration parameters.
            Currently, this is restricted to Country and State.
        
        Returns
        -----------
        weather_df : data frame
            pandas dataframe with requested columns (see below)
        meta : dictionary
            python dictionary containing weather meta data required for calculation
            Currently includes (latitude, longitude)
        '''

        weather_file = r'/datasets/NSRDB/current/nsrdb_tmy-2021.h5'
        weather_params = ['air_temperature', 'wind_speed', 'dhi', 'ghi', 'dni']

        with NSRDBX(weather_file, hsds=False) as f:
            meta = f.meta

        meta_country = meta[meta['country'] == iterables['country']]
        region_col = 'state'

        for state in meta_country['state'].unique():
            region = state

            with NSRDBX(weather_file, hsds=False) as f:
                times = f.time_index
                gids = f.region_gids(region=region, region_col=region_col)
                meta = f.meta[f.meta.index.isin(gids)]

            data = []
            with NSRDBX(weather_file, hsds=False) as f:
                for p in weather_params:
                    data.append(f.get_gid_df(p, gids))

            columns = pd.MultiIndex.from_product([weather_params, gids], names=["par", "gid"])
            df_weather = pd.concat(data, axis=1)
            df_weather.columns = columns
            df_weather = df_weather.swaplevel(axis=1).sort_index(axis=1)
        
        return df_weather, meta


    def _run_one(function, params, weather_df, meta):
        '''
        Runs a single function call from PVD Tools
        If multiple function names given, panic.
        '''

        lat = meta['latitude']
        lon = meta['longitude']

        class_list = [ c for c in dir(PVD) if not c.startswith('_') ]
        for c in class_list:
            _class = getattr(PVD,c)
            if function in dir(_class):
                _func = getattr(_class,function)

        params = {**params,'weather_df':weather_df,'metadata':meta}
        calc = _func(**params)

        results = [lat, lon, calc]

        return results

    def _save_results(results_df, usr):
        '''
        Save results dataframe as .csv (pickle?)
        
        Parameters
        -----------
        results_df : data frame
            pandas data frame with all calculated values and location-index
        usr : string
            HPC username do determine file saving location
        '''

        filedate = dt.strftime(date.today(), "%d%m%y")
        out_path = rf'/scratch/{usr}/pvd_jobs/run_{filedate}'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_file = out_path + r'pvd_results.csv'
        
        results_df.to_csv(out_file)

    def submit(ini_file=None):
        '''
        Submit a job to the SLURM queue

        Parameters
        -----------
        ini_file: file path to pvd_job.ini
            configuration file to run the PVD job.
            If one is not supplied, will search current directory for 'pvd_job.ini'

        Returns
        -------
        output : dataframe
            dataframe containing all the requested results
            saved as .csv file also
        '''

        # -- Verify Environment is HPC
        if os.name != 'posix':
            print('HPC Environment not detected, job will not be queued')
            return None

        # -- Function Parameters and Iterables
        param_dict = HighPerformance._read_ini(ini_file)
        iterables = param_dict['iterables']
        function_params = param_dict['parameters']
        usr = param_dict['job']['user']
        job = param_dict['job']['function']

        # -- Weather Data Processing
        df_weather, meta = HighPerformance._gen_weather(iterables)

        # -- SLURM management
        scheduler_file = rf'/scratch/{usr}/scheduler.json'
        client = Client(scheduler_file=scheduler_file)
        futures = []
        for gid, row in meta.iterros():
            meta_dict = row.loc[['latitude','longitude']].to_dict()
            df_weather_filtered = df_weather.loc[:,gid]
            futures.append(client.submit(HighPerformance._run_one,
                                         job, function_params, df_weather_filtered, meta_dict))

        # -- Result Management
        res = client.gather(futures)
        results = pd.DataFrame(res, columns=('latitude','longitude',job))
