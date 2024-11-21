""" 
Pysam Integration for pvdeg, supports single site and geospatial calculations.
Produced to support Inspire Agrivoltaics: https://openei.org/wiki/InSPIRE
"""

import dask.dataframe as dd
import pandas as pd
import pickle
import json
import sys
import os

import PySAM
import PySAM.Pvsamv1 as pv1
import PySAM.Pvwattsv8 as pv8
import PySAM.Grid as Grid
import PySAM.Utilityrate5 as UtilityRate
import PySAM.Cashloan as Cashloan

from pvdeg import (
    weather,
    utilities,
    DATA_DIR
)

def vertical_POA(
    weather_df,
    meta,
    jsonfolder='/projects/pvsoiling/pvdeg/analysis/northern_lat/jsons',
    samjsonname='vertical',
    weather_kwarg=None,
):
    """
    Run a SAM

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data for a single location.
    meta : pd.DataFrame
        Meta data for a single location.
    weather_kwarg : dict
        other variables needed to access a particular weather dataset.
    jsonfolder : string
        Location and base name for the json files

    Returns
    -------
    annual_gh : float [Wh/m2/y]
        Annual GHI
    annual_energy : float [kWh]
        Annual AC energy
    lcoa_nom : float [cents/kWh]
        LCOE Levelized cost of energy nominal
    """

   
    parameters = ["temp_air", "wind_speed", "dhi", "ghi", "dni"]
    #print("weather_df KEYs", weather_df.keys())
    #print("meta KEYs", meta.keys())

    
    if isinstance(weather_df, dd.DataFrame):
        weather_df = weather_df[parameters].compute()
        weather_df.set_index("time", inplace=True)
    elif isinstance(weather_df, pd.DataFrame):
        weather_df = weather_df[parameters]
    elif weather_df is None:
        weather_df, meta = weather.get(**weather_kwarg)


    file_names = ["pvsamv1", "grid", "utilityrate5", "cashloan"]
    pv4 = PV.new()  # also tried PVWattsSingleOwner
    grid4 = Grid.from_existing(pv4)
    ur4 = UtilityRate.from_existing(pv4)
    so4 = Cashloan.from_existing(grid4, 'FlatPlatePVCommercial')

    # LOAD Values
    for count, module in enumerate([pv4, grid4, ur4, so4]):
        filetitle= samjsonname + '_' + file_names[count] + ".json"
        with open(os.path.join(jsonfolder,filetitle), 'r') as file:
            data = json.load(file)
            for k, v in data.items():
                if k == 'number_inputs':
                    continue
                try:
                    #if sys.version.split(' ')[0] == '3.11.7':  
                        # Bypassing this as it's not working with my pandas. the printouts
                        # like "<Pvsamv1 object at 0x7f0f01339cf0> dc_adjust_periods [[0, 0, 0]]" 
                        # means it is going to the except. !!!! 
                        
                    
                        # Check needed for python 3.10.7 and perhaps other releases above 3.10.4.
                        # This prevents the failure "UnicodeDecodeError: 'utf-8' codec can't decode byte... 
                        # This bug will be fixed on a newer version of pysam (currently not working on 5.1.0)
                    if 'adjust_' in k:  # This check is needed for Python 3.10.7 and some others. Not needed for 3.7.4
                        #print(k)
                        k = k.split('adjust_')[1]
                    module.value(k, v)
                except AttributeError:
                    # there is an error is setting the value for ppa_escalation
                    print(module, k, v)

    pv4.unassign('solar_resource_file')
                
    #print("Type meta ", meta)

    if 'tz' not in meta:
        meta['tz'] = '+0'
        
    #if meta.get('tz') == None: 
        #meta.loc['tz'] = '+0'

    if "albedo" not in weather_df.columns:
        #weather_df['albedo'] = 0.2 # new pandas cries with this. dumb.
        weather_df.loc[:,'albedo'] = 0.2

    data = {'dn':list(weather_df.dni),
           'df':list(weather_df.dhi),
            'gh':list(weather_df.ghi),
           'tdry':list(weather_df.temp_air),
           'wspd':list(weather_df.wind_speed),
           'lat':meta['latitude'],
           'lon':meta['longitude'],
           'tz':meta['tz'],
           'elev':meta['altitude'],
           'year':list(weather_df.index.year),
           'month':list(weather_df.index.month),
           'day':list(weather_df.index.day),
           'hour':list(weather_df.index.hour),
           'minute':list(weather_df.index.minute),
           'alb':list(weather_df.albedo)}

    pv4.value('solar_resource_data', data)

    pv4.execute()
    grid4.execute()
    ur4.execute()
    so4.execute()
    
    # SAVE RESULTS|
    results = pv4.Outputs.export()
    economicresults = so4.Outputs.export()
    
    annual_gh = results['annual_gh']
    annual_energy = results['annual_ac_gross']
    lcoe_nom = economicresults['lcoe_nom']
    
    res = {"annual_gh": annual_gh, "annual_energy": annual_energy, "lcoe_nom": lcoe_nom}
    df_res = pd.DataFrame.from_dict(res, orient="index").T

    return df_res

def pysam(
    weather_df: pd.DataFrame,
    meta: dict,
    pv_model: str,
    pv_model_default: str,
    # grid_default: str,
    # cashloan_default: str,
    # utilityrate_default: str,
    config_files: dict[str: str] = None,
    results: list[str] = None,
) -> dict:
    """
    Run pySam simulation. Only works with pysam weather.

    Parameters
    -----------
    weather_df: pd.DataFrame
        DataFrame of weather data. As returned by ``pvdeg.weather.get``
    meta: dict
        Dictionary of metadata for the weather data. As returned by ``pvdeg.weather.get``
    pv_model: str
        choose pySam photovoltaic system model. 
        Some models are less thorough and run faster. 
        pvwatts8 is ~50x faster than pysamv1 but only calculates 46 parameters while pysamv1 calculates 195.

            options: ``pvwatts8``, ``pysamv1``, etc.

    pv_model_default: str
        pysam config for pv model. [Pysam Modules](https://nrel-pysam.readthedocs.io/en/main/ssc-modules.html)

        On the docs some modules have availabile defaults listed. 

        For example:  
        [Pvwattsv8](https://nrel-pysam.readthedocs.io/en/main/modules/Pvwattsv8.html)
        - "FuelCellCommercial"
        - "FuelCellSingleOwner"
        - "GenericPVWattsWindFuelCellBatteryHybridHostDeveloper"
        - "GenericPVWattsWindFuelCellBatteryHybridSingleOwner"
        - "PVWattsBatteryCommercial"
        - "PVWattsBatteryHostDeveloper"
        - "PVWattsBatteryResidential"
        - "PVWattsBatteryThirdParty"
        - "PVWattsWindBatteryHybridHostDeveloper"
        - "PVWattsWindBatteryHybridSingleOwner"
        - "PVWattsWindFuelCellBatteryHybridHostDeveloper"
        - "PVWattsWindFuelCellBatteryHybridSingleOwner"
        - "PVWattsAllEquityPartnershipFlip"
        - "PVWattsCommercial"
        - "PVWattsCommunitySolar"
        - "PVWattsHostDeveloper"
        - "PVWattsLCOECalculator"
        - "PVWattsLeveragedPartnershipFlip"
        - "PVWattsMerchantPlant"
        - "PVWattsNone"
        - "PVWattsResidential"
        - "PVWattsSaleLeaseback"
        - "PVWattsSingleOwner"
        - "PVWattsThirdParty"

        [Pvsamv1](https://nrel-pysam.readthedocs.io/en/main/modules/Pvsamv1.html)
        - "FlatPlatePVAllEquityPartnershipFlip"
        - "FlatPlatePVCommercial"
        - "FlatPlatePVHostDeveloper"
        - "FlatPlatePVLCOECalculator"
        - "FlatPlatePVLeveragedPartnershipFlip"
        - "FlatPlatePVMerchantPlant"
        - "FlatPlatePVNone"
        - "FlatPlatePVResidential"
        - "FlatPlatePVSaleLeaseback"
        - "FlatPlatePVSingleOwner"
        - "FlatPlatePVThirdParty"
        - "PVBatteryAllEquityPartnershipFlip"
        - "PVBatteryCommercial"
        - "PVBatteryHostDeveloper"
        - "PVBatteryLeveragedPartnershipFlip"
        - "PVBatteryMerchantPlant"
        - "PVBatteryResidential"
        - "PVBatterySaleLeaseback"
        - "PVBatterySingleOwner"
        - "PVBatteryThirdParty"
        - "PhotovoltaicWindBatteryHybridHostDeveloper"
        - "PhotovoltaicWindBatteryHybridSingleOwner"

    grid_default: str

        pysam default config for grid model. [Grid Defaults](https://nrel-pysam.readthedocs.io/en/main/modules/Grid.html)

    cashloan_default: str

        pysam default config for cashloan model. [Cashloan Defaults](https://nrel-pysam.readthedocs.io/en/main/modules/Cashloan.html)  
        - "FlatPlatePVCommercial"  
        - "FlatPlatePVResidential"  
        - "PVBatteryCommercial"  
        - "PVBatteryResidential"  
        - "PVWattsBatteryCommercial"  
        - "PVWattsBatteryResidential"  
        - "PVWattsCommercial"  
        - "PVWattsResidential"  

    utiltityrate_default: str

        pysam default config for utilityrate5 model. [Utilityrate5 Defaults](https://nrel-pysam.readthedocs.io/en/main/modules/Utilityrate5.html())

    config_files: dict
        SAM configuration files. A dictionary containing a mapping to filepaths.

        Keys must be `'pv', 'grid', 'utilityrate', 'cashloan'`. Each key should contain a value as a string representing the file path to a SAM config file. 

        ```
        files = {
            'pv' : 'example/path/1/pv-file.json'
            'grid' : 'example/path/1/grid-file.json'
            'utilityrate' : 'example/path/1/utilityrate-file.json'
            'cashloan' : 'example/path/1/cashloan-file.json'
        }
        ```

    results: list[str]
        list of strings corresponding to pysam outputs to return.
        Pysam models such as `Pvwatts8` and `Pvsamv1` return hundreds of results.
        So we can chose to take only the specified results while throwing away the others.

        To grab only 'annual_energy' and 'ac' from the model results. 

        >>> results = ['annual_energy', 'ac']

        This may cause some undesired behavior with geospatial calculations if the lengths of the results within the list are different. 

    Returns
    -------
    pysam_res: dict
        dictionary of outputs. Keys are result name and value is the corresponding result. 
        If `results` is not specified, the dictionary will contain every calculation from the model.
    """

    sr = solar_resource_dict(weather_df=weather_df, meta=meta)

    # weather_df = utilities.add_time_columns_tmy(weather_df=weather_df)

    # solar_resource = {
    #     'lat': meta['latitude'],
    #     'lon': meta['longitude'],
    #     'tz': meta['tz'] if 'tz' in meta.keys() else 0,
    #     'elev': meta['altitude'],
    #     'year': list(weather_df['Year']),
    #     'month': list(weather_df['Month']),
    #     'day': list(weather_df['Day']),
    #     'hour': list(weather_df['Hour']),
    #     'minute': list(weather_df['Minute']),
    #     'dn': list(weather_df['dni']),
    #     'df': list(weather_df['dhi']),
    #     'wspd': list(weather_df['wind_speed']),
    #     'tdry': list(weather_df['temp_air']),
    #     'alb' : weather_df['albedo'] if 'albedo' in weather_df.columns.values else [0.2]*len(weather_df)
    # }

    # https://nrel-pysam.readthedocs.io/en/main/modules/Pvwattsv8.html
    # https://nrel-pysam.readthedocs.io/en/main/modules/Pvsamv1.html
    if pv_model == "pvwatts8": 
        pysam_model = pv8.default(pv_model_default) # PVWattsCommercial
    elif pv_model == "pysamv1":
        pysam_model = pv1.default(pv_model_default) # FlatPlatePVCommercial

    pysam_model.unassign('solar_resource_file') # unassign file

    # grid = Grid.from_existing(pv_model)
    # utility_rate = UtilityRate.from_existing(pv_model)
    # cashloan = Cashloan.from_existing(grid, 'FlatPlatePVCommercial')


    # Duplicate Columns in the dataframe seem to cause this issue
    # Error (-4) converting nested tuple 0 into row in matrix.
    pysam_model.SolarResource.solar_resource_data = sr 
    pysam_model.execute()
    outputs = pysam_model.Outputs.export()

    if not results:
        return outputs

    pysam_res = {}
    for key in results:
        pysam_res[key] = outputs[key]
    return pysam_res

def pysam_hourly_trivial(weather_df, meta):

    sr = solar_resource_dict(weather_df=weather_df, meta=meta)
    # weather_df = weather_df.reset_index(drop=True)
    # weather_df = utilities.add_time_columns_tmy(weather_df) # only supports hourly data
    
    # sr = {
    #     'lat': meta['latitude'],
    #     'lon': meta['longitude'],
    #     'tz': meta['tz'] if 'tz' in meta.keys() else 0,
    #     'elev': meta['altitude'],
    #     'year': list(weather_df['Year']),
    #     'month': list(weather_df['Month']),
    #     'day': list(weather_df['Day']),
    #     'hour': list(weather_df['Hour']),
    #     'minute': list(weather_df['Minute']),
    #     'dn': list(weather_df['dni']),
    #     'df': list(weather_df['dhi']),
    #     'wspd': list(weather_df['wind_speed']),
    #     'tdry': list(weather_df['temp_air']),
    #     'alb' : weather_df['albedo'] if 'albedo' in weather_df.columns.values else [0.2]*len(weather_df)
    # }
    
    # model = pv8.default("PVWattsCommercial")
    model = pv1.default("FlatPlatePVCommercial")
    model.SolarResource.solar_resource_data = sr
    model.execute()
    outputs = model.Outputs.export()

    return outputs

# TODO: add slots
class inspirePysamReturn():
    """simple struct to facilitate handling weirdly shaped pysam simulation return values"""

    # removes __dict__ atribute and breaks pickle
    # __slots__ = ("annual_poa", "ground_irradiance", "timeseries_index")

    def __init__(self, annual_poa, ground_irradiance, timeseries_index):
        self.annual_poa = annual_poa
        self.ground_irradiance = ground_irradiance
        self.timeseries_index = timeseries_index


# rename?
import xarray as xr
def _handle_pysam_return(pysam_res : inspirePysamReturn) -> xr.Dataset:
    """Handle a pysam return object and transform it to an xarray"""

    import dask.array as da
    import numpy as np

    # redo this using numba?
    ground_irradiance_values = da.from_array([row[1:] for row in pysam_res.ground_irradiance[1:]])

    single_location_ds = xr.Dataset(
        data_vars={
            "annual_poa" : pysam_res.annual_poa, # scalar variable
            "ground_irradiance" : (("time", "distance"), ground_irradiance_values)
        },
        coords={
            # "time" : np.arange(17520), # this will probably break because of the values assigned here, should be a pd.datetimeindex instead
            "time" : pysam_res.timeseries_index,
            # "distance" : np.array(pysam_res.ground_irradiance[0][1:])
            "distance" : np.arange(10), # this matches the dimension axis of the output_temlate dataset
        }
    ) 

    return single_location_ds


# annual_poa_nom, annual_poa_front, annual_poa_rear, poa_nom, poa_front, or poa_rear
# TODO: add config file, multiple config files.
def inspire_ground_irradiance(weather_df, meta):
    """
    Get ground irradiance array and annual poa irradiance for a given point using pvsamv1

    Returns
    --------
    result : inspirePysamReturn
        returns an custom class object so we can unpack it later.
    """

    sr = solar_resource_dict(weather_df=weather_df, meta=meta)

    model = pv1.default("FlatPlatePVCommercial")
    model.SolarResource.solar_resource_data = sr
    model.execute()
    outputs = model.Outputs.export()

    outputs = pysam(
        weather_df = weather_df,
        meta = meta,
        pv_model = "pysamv1",
        pv_model_default = "FlatPlatePVCommercial", # should use config file instead
        results = ["subarray1_ground_rear_spatial", "annual_poa_front"],
    )

    result = inspirePysamReturn(
            ground_irradiance = outputs["subarray1_ground_rear_spatial"],
            annual_poa = outputs["annual_poa_front"],
            timeseries_index=weather_df.index,
        )

    return result

def solar_resource_dict(weather_df, meta):
    """
    Create a solar resource dict mapping from weather and metadata.

    Works on PVGIS and appears to work on NSRDB (NOT PSM3).
    """

    weather_df = weather_df.reset_index(drop=True) # Probably dont need to do this
    weather_df = utilities.add_time_columns_tmy(weather_df) # only supports hourly data
    
    sr = {
        'lat': meta['latitude'],
        'lon': meta['longitude'],
        'tz': meta['tz'] if 'tz' in meta.keys() else 0,
        'elev': meta['altitude'],
        'year': list(weather_df['Year']),
        'month': list(weather_df['Month']),
        'day': list(weather_df['Day']),
        'hour': list(weather_df['Hour']),
        'minute': list(weather_df['Minute']),
        'dn': list(weather_df['dni']),
        'df': list(weather_df['dhi']),
        'wspd': list(weather_df['wind_speed']),
        'tdry': list(weather_df['temp_air']),
        'alb' : weather_df['albedo'] if 'albedo' in weather_df.columns.values else [0.2]*len(weather_df)
    }

    return sr 

def sample_pysam_result(weather_df, meta): # throw weather, meta away
    """returns a sample inspirePysamReturn"""
    with open(os.path.join(DATA_DIR,"inspireInstance.pkl"), "rb") as file:
        inspireInstance = pickle.load(file)

    return inspireInstance