""" 
Pysam Integration for pvdeg, supports single site and geospatial calculations.
Produced to support Inspire Agrivoltaics: https://openei.org/wiki/InSPIRE
"""

import dask.dataframe as dd
import dask.array as da
import pandas as pd
import xarray as xr
import numpy as np
import pickle
import json
import sys
import os

from pvdeg import (
    weather,
    utilities,
    decorators,
    DATA_DIR
)

@decorators.deprecated("unverified")
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

    try:
        import PySAM
        import PySAM.Pvsamv1 as pv1
        import PySAM.Pvwattsv8 as pv8
        import PySAM.Grid as Grid
        import PySAM.Utilityrate5 as UtilityRate
        import PySAM.Cashloan as Cashloan
    except ModuleNotFoundError:
        print("pysam not found. run `pip install pvdeg[sam]` to install the NREL-PySAM dependency")
        return

   
    parameters = ["temp_air", "wind_speed", "dhi", "ghi", "dni"]
    
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
                        k = k.split('adjust_')[1]
                    module.value(k, v)
                except AttributeError:
                    # there is an error is setting the value for ppa_escalation
                    print(module, k, v)

    pv4.unassign('solar_resource_file')
                
    if 'tz' not in meta:
        meta['tz'] = '+0'
        
    if "albedo" not in weather_df.columns:
        print("using placeholder albedo of 0.2 for all timesteps")
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

# TODO: add grid_default, cashloan_default, utilityrate_defaults for expanded pysam simulation capabilities
def pysam(
    weather_df: pd.DataFrame,
    meta: dict,
    pv_model: str,
    pv_model_default: str = None,
    config_files: dict[str: str] = None,
    results: list[str] = None,
    # pitch_override: float = None,
    # tilt_override: float = None,
    inspire_practical_pitch_tilt: bool = False,
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

        Keys must be `'pv', 'grid', 'utilityrate', 'cashloan'`. Each key should contain a value as a string representing the file path to a SAM config file. Cannot deal with the entire SAM config json.

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

    pitch_override: float
        override defined pitch from pv config file (fixed tilt systems only)
    tilt_override: float
        override defined tilt from pv config file (fixed tilt systems only)

    inspire_practical_tilt_pitch: bool
        use inspire practical considerations to limit/override defined pitch and tilt from SAM configs.
        

    Returns
    -------
    pysam_res: dict
        dictionary of outputs. Keys are result name and value is the corresponding result. 
        If `results` is not specified, the dictionary will contain every calculation from the model.
    """

    try:
        import PySAM
        import PySAM.Pvsamv1 as pv1
        import PySAM.Pvwattsv8 as pv8
        import PySAM.Grid as Grid
        import PySAM.Utilityrate5 as UtilityRate
        import PySAM.Cashloan as Cashloan
    except ModuleNotFoundError:
        print("pysam not found. run `pip install pvdeg[sam]` to install the NREL-PySAM dependency")
        return

    sr = solar_resource_dict(weather_df=weather_df, meta=meta)

    # https://nrel-pysam.readthedocs.io/en/main/modules/Pvwattsv8.html
    # https://nrel-pysam.readthedocs.io/en/main/modules/Pvsamv1.html
    model_map = {
        "pvwatts8"  : pv8,
        "pysamv1"   : pv1,
    }

    model_module = model_map[pv_model]
    if pv_model_default:
        pysam_model = model_module.default(pv_model_default)
    elif pv_model_default is None:
        pysam_model = model_module.new()

        with open( config_files['pv'], 'r') as f:
            pv_inputs = json.load( f )

        # these break the model when being loaded using InSpire doubleday configs
        # this is NREL-PySAM version dependent, these are problematic on 5.1.0
        bad_parameters = {
            'adjust_constant', 
            'adjust_en_timeindex', 
            'adjust_en_periods', 
            'adjust_timeindex', 
            'adjust_periods', 
            'dc_adjust_constant', 
            'dc_adjust_en_timeindex', 
            'dc_adjust_en_periods', 
            'dc_adjust_timeindex', 
            'dc_adjust_periods'
        }

        subarrays = set()

        for k, v in pv_inputs.items():
            if k not in ({'number_inputs', 'solar_resource_file'} | bad_parameters):
                pysam_model.value(k, v)

                # get all subarrays being used
                if k.startswith("subarray"):
                    subarrays.add(k.split("_")[0])

    if inspire_practical_pitch_tilt == True:
        print("overriding pitch -- debug (remove)")
        print("subarrays --debug (remove)")
        print(subarrays)

        # Create lists of parameter names
        # need to check all subarrays and update all subarrays
        param_latitude_tilt = [f"{subarray}_tilt_eq_lat" for subarray in subarrays]
        param_tracker_mode = [f"{subarray}_track_mode" for subarray in subarrays]
        param_tilt = [f"{subarray}_tilt" for subarray in subarrays]
        param_gcr = [f"{subarray}_gcr" for subarray in subarrays]

        if any(pysam_model.value(name) != 0 for name in param_latitude_tilt):
            print('latitude tilt defined for one of the subarrays, overriding to disable latitude tilt')
            for name in param_latitude_tilt:
                pysam_model.value(name, 0) 
        
        if any(pysam_model.value(name) != 0 for name in param_tracker_mode):
            raise ValueError("Inspire Practical Pitch, Tilt Consideration Failed: at least one subarray is using tracking")

        if any(pysam_model.value(name) == 90 for name in param_tilt):
            raise ValueError("Inspire Practical Pitch, Tilt Consideration Failed: at least one subarray is vertical fixed tilt (tilt = 90 deg)")

        # collector width of 2m for the inspire scenarios
        tilt_prac, pitch_prac, gcr_prac = inspire_practical_pitch(latitude=meta["latitude"], cw=2)

        # if the above passed, then we want to run
        for name in param_tilt:
            pysam_model.value(name, tilt_prac) 

        for name in param_gcr:
            pysam_model.value(name, gcr_prac) 

    pysam_model.unassign('solar_resource_file') # unassign file

    # Duplicate Columns in the dataframe seem to cause this issue
    # Error (-4) converting nested tuple 0 into row in matrix.
    pysam_model.SolarResource.solar_resource_data = sr 
    pysam_model.execute()
    outputs = pysam_model.Outputs.export()

    if not results:
        return outputs

    print("gcr used")
    print(pysam_model.value("subarray1_gcr") )

    pysam_res = {key: outputs[key] for key in results}
    return pysam_res

def _handle_pysam_return(pysam_res_dict : dict, weather_df: pd.DataFrame, tilt: float, pitch: float) -> xr.Dataset:
    """Handle a pysam return object and transform it to an xarray"""

    ground_irradiance = pysam_res_dict["subarray1_ground_rear_spatial"]

    annual_poa = pysam_res_dict["annual_poa_front"]
    annual_energy = pysam_res_dict["annual_energy"]

    # if we have made the considerations then use 
    subarray1_poa_front = pysam_res_dict["subarray1_poa_front"][:8760]
    subarray1_poa_rear = pysam_res_dict["subarray1_poa_rear"][:8760]
    subarray1_celltemp = pysam_res_dict["subarray1_celltemp"][:8760]
    subarray1_dc_gross = pysam_res_dict["subarray1_dc_gross"][:8760]

    timeseries_index = weather_df.index

    # redo this using numba?
    distances = ground_irradiance[0][1:]
    ground_irradiance_values = da.from_array([row[1:] for row in ground_irradiance[1:]])

    single_location_ds = xr.Dataset(
        data_vars={
            # SCALARS
            # we will calculate these for some configs.
            # these are calculated in inspire_ground_irradiance using inspire_practical_pitch
            "tilt": float(tilt),
            "pitch": float(pitch),

            "annual_poa" : annual_poa,
            "annual_energy" : annual_energy,

            # TIMESERIES (model outputs)
            "subarray1_poa_front" : (("time", ), da.array(subarray1_poa_front)),
            "subarray1_poa_rear" : (("time", ), da.array(subarray1_poa_rear)),
            "subarray1_celltemp" : (("time", ), da.array(subarray1_celltemp)),
            "subarray1_dc_gross" : (("time", ), da.array(subarray1_dc_gross)),

            # TIMESERIES (weather inputs)
            "temp_air":(("time", ), da.array(weather_df["temp_air"].values)),
            "wind_speed":(("time", ), da.array(weather_df["wind_speed"].values)),
            "wind_direction":(("time", ), da.array(weather_df["wind_direction"].values)),
            "dhi":(("time", ), da.array(weather_df["dhi"].values)),
            "ghi":(("time", ), da.array(weather_df["ghi"].values)),
            "dni":(("time", ), da.array(weather_df["dni"].values)),
            "relative_humidity":(("time", ), da.array(weather_df["relative_humidity"].values)),
            "albedo":(("time", ), da.array(weather_df["albedo"].values)),

            # SPATIO-TEMPORAL (model outputs)
            "ground_irradiance" : (("time", "distance"), ground_irradiance_values),
        },
        coords={
            "time" : timeseries_index,
            # distances vary for config and locations (on fixed tilt configs)
            # so we need to use a distance "index" that is not spatially meaningful
            "distance" : np.arange(10), # convient way to match the distances in the template
        }
    ) 

    return single_location_ds


INSPIRE_NSRDB_ATTRIBUTES = [
    "air_temperature",
    "wind_speed",
    "wind_direction",
    "dhi",
    "ghi",
    "dni",
    "relative_humidity",
    "surface_albedo",
]

scalar = ("gid",)
temporal = ("gid", "time")
spatio_temporal = ("gid", "time", "distance")

INSPIRE_GEOSPATIAL_TEMPLATE_SHAPES = {
    "tilt":scalar,
    "pitch":scalar,
    "annual_poa": scalar,
    "annual_energy": scalar,

    "dhi": temporal,
    "ghi": temporal,
    "dni": temporal,
    "albedo": temporal,
    "temp_air": temporal,
    "wind_speed": temporal,
    "wind_direction": temporal,
    "relative_humidity": temporal,
    "subarray1_poa_front" : temporal,
    "subarray1_poa_rear" : temporal,
    "subarray1_celltemp" : temporal,
    "subarray1_dc_gross" : temporal,

    "ground_irradiance": spatio_temporal,
}

# TODO: should this gcr 
# TODO: should this be in standards or design (or other)?
# TODO: should this contain all of the parameters for optimal gcr and let us choose
def optimal_gcr_pitch(latitude: float, cw: float = 2) -> tuple[float, float]:
    """
    determine optimal gcr and pitch for fixed tilt systems according to latitude and optimal GCR parameters for fixed tilt bifacial systems.

    .. math::

        GCR = \frac{P}{1 + e^{-k(\alpha - \alpha_0)}} + GCR_0

    Inter-row energy yield loss 5% Bifacial Parameters:

    +-----------+--------+-----------+
    | Parameter | Value  | Units     |
    +===========+========+===========+
    | P         | 0.560  | unitless  |
    | K         | 0.133  | 1/°       |
    | α₀        | 40.2   | °         |
    | GCR₀      | 0.70   | unitless  |
    +-----------+--------+-----------+

    Parameters
    ------------
    latitude: float
        latitude [deg]
    cw: float
        collector width [m]

    Returns
    --------
    gcr: float
        optimal ground coverage ratio [unitless]
    pitch: float
        optimal pitch [m]

    References
    -----------
    Erin M. Tonita, Annie C.J. Russell, Christopher E. Valdivia, Karin Hinzer,
    Optimal ground coverage ratios for tracked, fixed-tilt, and vertical photovoltaic systems for latitudes up to 75°N,
    Solar Energy,
    Volume 258,
    2023,
    Pages 8-15,
    ISSN 0038-092X,
    https://doi.org/10.1016/j.solener.2023.04.038.
    (https://www.sciencedirect.com/science/article/pii/S0038092X23002682)

    Optimal GCR from Equation 4 
    Parameters from Table 1
    """

    p = -0.560 
    k = 0.133 
    alpha_0 = 40.2 
    gcr_0 = 0.70 

    # optimal gcr
    gcr = ((p) / (1 + np.exp(-k * (latitude - alpha_0)) )) + gcr_0

    pitch = cw / gcr
    return gcr, pitch

def inspire_practical_pitch(latitude: float, cw: float) -> tuple[float, float, float]:
    """
    Calculate pitch for fixed tilt systems for InSPIRE Agrivoltaics Irradiance Dataset.

    We cannot use the optimal pitch due to certain real world restrictions so we will apply some constraints.

    We are using latitude tilt but we cannot use tilts > 40 deg, due to racking constraints, cap at 40 deg for latitudes above 40 deg.

    pitch minimum: 3.8 m 
    pitch maximum:  12 m

    tilt max: 40 deg (latitude tilt)

    Parameters
    ----------
    latitude: float
        latitude [deg]
    cw: float
        collector width [m]

    Returns
    -------
    tilt: float
        tilt for a fixed tilt system with practical considerations [deg]
    pitch: float
        pitch for a fixed tilt system with practical consideration [m] 
    gcr: float
        gcr for a fixed tilt system with practical considerations [unitless]
    """

    gcr_optimal, pitch_optimal = optimal_gcr_pitch(latitude=latitude, cw=cw)

    pitch_ceil = min(pitch_optimal, 12)    # 12 m pitch ceiling
    pitch_practical = max(pitch_ceil, 3.8) # 3.8m pitch floor

    if not (3.8 <= pitch_practical <= 12):
        raise ValueError("calculated practical pitch is outside range [3.8m, 12m]")

    tilt_practical = min(latitude, 40)

    # practical gcr from practical pitch
    gcr_practical = cw / pitch_optimal

    return float(tilt_practical), float(pitch_practical), float(gcr_practical)

def load_gcr_from_config(config_files:dict):
    """
    dictionary containg 'pv' key
    """

    import json

    with open(config_files["pv"], 'r') as fp:
        data = json.load(fp)

    return data["subarray1_gcr"]

def inspire_ground_irradiance(weather_df, meta, config_files):
    """
    Get ground irradiance array and annual poa irradiance for a given point using pvsamv1

    REQUIRES: input weather data time index in UTC time.

    Parameters
    ----------
    weather_df : pd.DataFrame
        weather dataframe
    meta : dict
        meta data
    config_files : dict[str]
        see pvdeg.pysam.pysam
        # config_files={'pv' : <stringpathtofile>},

    Returns
    --------
    result : inspirePysamReturn
        returns an custom class object so we can unpack it later.
    """

    if (not isinstance(weather_df, pd.DataFrame) or not isinstance(meta, dict)):
        raise ValueError(f"""
            weather_df must be pandas DataFrame, meta must be dict.
            weather_df type : {type(weather_df)}
            meta type : {type(meta)}
        """)

    #### there is no pitch/gcr output from the model so we might have to do other checks to see that this is being applied correctly
    #### verify that our equations are correct. plot the world, view to see if practical applications have been applied.

    # force localize utc from tmy to local time by moving rows
    weather_df = weather.roll_tmy(weather_df, meta)

    tracking_setups = ["01", "02", "03", "04", "05"] 
    pratical_considerations_setups = ["06", "07", "08", "09"] # fixed tilt setups want to calculate pitch/gcr as a function of latitude capped at 40 degrees
    # vertical tilt (fixed spacing) 10

    print(f"config file string: {config_files['pv']} -- debug")
    
    cw = 2 # collector width 2 [m]
    pratical_consideration = False
    if any(setup in config_files["pv"] for setup in pratical_considerations_setups):
        print("setup with practical consieration detected, using pysam inspire_practical_consideration_pitch_tilt=True")
        pratical_consideration=True
        tilt_used, pitch_used, gcr_used = inspire_practical_pitch(latitude=meta['latitude'], cw=cw)
    
    # why would this be not in
    # this should check in "10" is in the config files
    elif "10" in config_files["pv"]:
        print("using config 10 with vertical fixed tilt.")
        gcr_used = load_gcr_from_config(config_files=config_files)
        pitch_used = gcr_used / cw
        tilt_used=90.0
    
    elif any(setup in config_files["pv"] for setup in tracking_setups): # conf 01- 05 using tracking, default gcr from pysam config
        gcr_used = load_gcr_from_config(config_files=config_files)
        pitch_used = gcr_used / cw
        tilt_used = -999.0 # tracking doesnt have fixed tilt (use placeholder instead)
    
    else:
        # this is not portable but we don't care. this can be custom for this calculation.
        raise ValueError("Valid config not found, config name must contain setup name from 01-10")

    outputs = pysam(
        weather_df = weather_df,
        meta = meta,
        pv_model = "pysamv1",
        config_files=config_files,
        inspire_practical_pitch_tilt=pratical_consideration, # tell model to calculate practical tilt, pitch, gcr again inside function, these will be the same results.
    )

    ds_result = _handle_pysam_return(pysam_res_dict=outputs, weather_df=weather_df, tilt=tilt_used, pitch=pitch_used)

    return ds_result

def solar_resource_dict(weather_df, meta):
    """
    Create a solar resource dict mapping from weather and metadata.

    Works on PVGIS and appears to work on NSRDB (NOT PSM3).
    """

    # weather_df = weather_df.reset_index(drop=True) # Probably dont need to do this
    weather_df = utilities.add_time_columns_tmy(weather_df) # only supports hourly data

    # enforce tmy scheme
    times = pd.date_range(start="2001-01-01", periods=8760, freq="1h")
    
    # all options
    # lat,lon,tz,elev,year,month,hour,minute,gh,dn,df,poa,tdry,twet,tdew,rhum,pres,snow,alb,aod,wspd,wdir
    sr = {
        'lat': meta['latitude'],
        'lon': meta['longitude'],
        'tz': meta['tz'] if 'tz' in meta.keys() else 0,
        'elev': meta['altitude'],
        'year': list(times.year), #list(weather_df['Year']),
        'month': list(times.month),
        'day': list(times.day),
        'hour': list(times.hour),
        'minute': list(times.minute),
        'gh': list(weather_df['ghi']),
        'dn': list(weather_df['dni']),
        'df': list(weather_df['dhi']),
        'wspd': list(weather_df['wind_speed']),
        'tdry': list(weather_df['temp_air']),
        'alb' : list(weather_df['albedo']) if 'albedo' in weather_df.columns.values else [0.2] * len(weather_df)
    }

    # if we have wind direction then add it
    if 'wind_direction' in weather_df.columns.values:
        sr['wdir'] = list(weather_df['wind_direction']) 

    return sr 

@decorators.deprecated("outdated shapes, original shapes for dataset")
def sample_inspire_result(weather_df, meta): # throw weather, meta away
    """
    returns a sample inspire_ground_irradiance xarray. Dataset for geospatial testing. 
    Weather_df and meta exist to provide a homogenous arugment structure for geospatial calculations but are not used.
    
    Parameters
    ----------
    weather: pd.Dataframe
        weather dataframe, is thrown away
    meta: dict
        metadata dictionary, is thrown away

    Returns
    -------
    inspire_ground_irradiance: xr.Dataset
        returns an xarray dataset of the same shape generated by inpspire_ground_irradiance()
    """

    return xr.Dataset(
        data_vars={
        'poa_rear': (('time',), np.zeros((8760,))), 
        'ground_irradiance': (('time', 'distance'), np.zeros((8760, 10))),
        'annual_energy': ((), 0), 
        'annual_poa': ((), 0),
        'subarray1_poa_front': (('time',), np.zeros((8760,))), 
        'subarray1_poa_rear': (('time',), np.zeros((8760,))), 
        'poa_front': (('time',), np.zeros((8760,))),   
        },
        coords={
            'time':pd.date_range(start="2001-01-01 00:30:00", periods=8760, freq='h'),
            'distance':np.arange(10),
        }
    )


def ground_irradiance_monthly(inspire_res_ds : xr.Dataset) -> xr.Dataset:
    """
    Many rows are not populated because the model only calculates ground irradiance when certain measurements are met.
    Drop the rows and calculate the monthly average irradiance at each distance.
    """

    nonzero_mask = (inspire_res_ds["ground_irradiance"] != 0).any(dim="distance")
    filtered_data = inspire_res_ds["ground_irradiance"].where(nonzero_mask, drop=True)

    monthly_avg_ground_irradiance = filtered_data.groupby(filtered_data.time.dt.month).mean()
    return monthly_avg_ground_irradiance 


