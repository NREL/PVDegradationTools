"""Pysam Integration for pvdeg, supports single site and geospatial calculations.

Produced to support Inspire Agrivoltaics: https://openei.org/wiki/InSPIRE
"""

import dask.array as da
import pandas as pd
from pvdeg.utilities import _load_gcr_from_config, inspire_practical_pitch
import xarray as xr
import numpy as np
import json

from pvdeg import (
    weather,
    utilities,
)


def pysam(
    weather_df: pd.DataFrame,
    meta: dict,
    pv_model: str,
    pv_model_default: str = None,
    config_files: dict[str:str] = None,
    results: list[str] = None,
    practical_pitch_tilt_considerations: bool = False,
) -> dict:
    """
    Run SAM solar simulation.

    Parameters
    -----------
    weather_df: pd.DataFrame
        DataFrame of weather data. As returned by ``pvdeg.weather.get``
    meta: dict
        Dictionary of metadata for the weather data. As returned by
        ``pvdeg.weather.get``
    pv_model: str
        choose pySam photovoltaic system model.
        Some models are less thorough and run faster.
        pvwatts8 is ~50x faster than pysamv1 but only calculates 46 parameters while
        pysamv1 calculates 195.

            options: ``pvwatts8``, ``pvsamv1``

        Documentation Links
        - [pvwatts8](https://nrel-pysam.readthedocs.io/en/main/modules/Pvwattsv8.html)
        - [pvsamv1](https://nrel-pysam.readthedocs.io/en/main/modules/Pvsamv1.html)

    pv_model_default: str
        pysam config for pv model.
        [Pysam Modules](https://nrel-pysam.readthedocs.io/en/main/ssc-modules.html)

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

        pysam default config for grid model.
        [Grid Defaults](https://nrel-pysam.readthedocs.io/en/main/modules/Grid.html)

    cashloan_default: str

        pysam default config for cashloan model.
        [Cashloan Defaults](https://nrel-pysam.readthedocs.io/en/main/modules/Cashloan.html)  # noqa
        - "FlatPlatePVCommercial"
        - "FlatPlatePVResidential"
        - "PVBatteryCommercial"
        - "PVBatteryResidential"
        - "PVWattsBatteryCommercial"
        - "PVWattsBatteryResidential"
        - "PVWattsCommercial"
        - "PVWattsResidential"

    utiltityrate_default: str

        pysam default config for utilityrate5 model.
        [Utilityrate5 Defaults](https://nrel-pysam.readthedocs.io/en/main/modules/Utilityrate5.html())  # noqa

    config_files: dict
        SAM configuration files. A dictionary containing a mapping to filepaths.

        Keys must be `'pv', 'grid', 'utilityrate', 'cashloan'`.
        Each key should contain a value as a string representing the file path to a SAM
        config file. Cannot deal with the entire SAM config json.

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
        So we can chose to take only the specified results while throwing away the
        others.

        To grab only 'annual_energy' and 'ac' from the model results.

        >>> results = ['annual_energy', 'ac']

        This may cause some undesired behavior with geospatial calculations if the
        lengths of the results within the list are different.

    practical_pitch_tilt_considerations: bool
        Use inspire practical considerations to limit/override defined pitch and tilt from SAM configs.

        Calculates optimal GCR using `pvdeg.utilities.optimal_gcr_pitch` for fixed tilt bifacial systems.
        Imposes a minimum pitch of 3.8m and maximum pitch of 12m.

    Returns
    -------
    pysam_res: dict
        dictionary of outputs. Keys are result name and value is the corresponding
        result.
        If `results` is not specified, the dictionary will contain every calculation
        from the model.
    """
    try:
        import PySAM.Pvsamv1 as pv1
        import PySAM.Pvwattsv8 as pv8
    except ModuleNotFoundError:
        print(
            "pysam not found. run `pip install pvdeg[sam]` to install the NREL-PySAM \
            dependency"
        )
        return

    sr = solar_resource_dict(weather_df=weather_df, meta=meta)

    model_map = {
        "pvwatts8"  : pv8,
        "pvsamv1"   : pv1,
    }

    model_module = model_map[pv_model]
    if pv_model_default:
        pysam_model = model_module.default(pv_model_default)
    elif pv_model_default is None:
        pysam_model = model_module.new()

        with open(config_files["pv"], "r") as f:
            pv_inputs = json.load(f)

        # these break the model when being loaded using InSpire doubleday configs
        # this is NREL-PySAM version dependent, these are problematic on 5.1.0
        bad_parameters = {
            "adjust_constant",
            "adjust_en_timeindex",
            "adjust_en_periods",
            "adjust_timeindex",
            "adjust_periods",
            "dc_adjust_constant",
            "dc_adjust_en_timeindex",
            "dc_adjust_en_periods",
            "dc_adjust_timeindex",
            "dc_adjust_periods",
        }

        subarrays = set()

        for k, v in pv_inputs.items():
            if k not in ({"number_inputs", "solar_resource_file"} | bad_parameters):
                pysam_model.value(k, v)

                # get all subarrays being used
                if k.startswith("subarray"):
                    subarrays.add(k.split("_")[0])

    if practical_pitch_tilt_considerations is True:
        _apply_practical_pitch_tilt(
            pysam_model=pysam_model,
            meta=meta,
            subarrays=subarrays
        )

    pysam_model.unassign('solar_resource_file')

    # Duplicate Columns in the dataframe seem to cause this issue
    # Error (-4) converting nested tuple 0 into row in matrix.
    pysam_model.SolarResource.solar_resource_data = sr
    pysam_model.execute()
    outputs = pysam_model.Outputs.export()

    if not results:
        return outputs

    print("gcr used")
    print(pysam_model.value("subarray1_gcr"))

    pysam_res = {key: outputs[key] for key in results}
    return pysam_res

def _apply_practical_pitch_tilt(pysam_model, meta: dict, subarrays: set[str]) -> None:
    """
    Apply practical pitch/tilt constraints to all subarrays on the model.
    Mutates `pysam_model` in-place. Raises the same errors as the inlined code.
    """
    print("overriding pitch with practical considerations")
    print(f"subarrays {subarrays}")

    # Build parameter name lists for all discovered subarrays
    param_latitude_tilt = [f"{s}_tilt_eq_lat" for s in subarrays]
    param_tracker_mode = [f"{s}_track_mode" for s in subarrays]
    param_tilt         = [f"{s}_tilt" for s in subarrays]
    param_gcr          = [f"{s}_gcr" for s in subarrays]

    # Disable latitude-equals-tilt if set anywhere
    if any(pysam_model.value(name) != 0 for name in param_latitude_tilt):
        print(
            'config defined latitude tilt defined for one of the subarrays, '
            'disabling config latitude tilt (will be set later using practical consideration)'
        )
        for name in param_latitude_tilt:
            pysam_model.value(name, 0)

    # Disallow tracking
    if any(pysam_model.value(name) != 0 for name in param_tracker_mode):
        raise ValueError(
            "Inspire Practical Pitch,Tilt Consideration Failed: "
            "at least one subarray is using tracking"
        )

    # Disallow vertical fixed-tilt
    if any(pysam_model.value(name) == 90 for name in param_tilt):
        raise ValueError(
            "Inspire Practical Pitch,Tilt Consideration Failed: "
            "at least one subarray is vertical fixed tilt (tilt = 90 deg)"
        )

    # collector width of 2m for the inspire scenarios
    tilt_prac, pitch_prac, gcr_prac = inspire_practical_pitch(
        latitude=meta["latitude"], cw=2
    )

    # Apply practical tilt/GCR (pitch is implied via GCR in SAM)
    for name in param_tilt:
        pysam_model.value(name, tilt_prac)
    for name in param_gcr:
        pysam_model.value(name, gcr_prac)


def _handle_pysam_return(
    pysam_res_dict : dict,
    weather_df: pd.DataFrame,
    tilt: float,
    pitch: float
) -> xr.Dataset:
    """Handle a pysam return object and transform it to an xarray"""

    ground_irradiance = pysam_res_dict["subarray1_ground_rear_spatial"]

    annual_poa = pysam_res_dict["annual_poa_front"]
    annual_energy = pysam_res_dict["annual_energy"]

    subarray1_poa_front = pysam_res_dict["subarray1_poa_front"][:8760]
    subarray1_poa_rear = pysam_res_dict["subarray1_poa_rear"][:8760]
    subarray1_celltemp = pysam_res_dict["subarray1_celltemp"][:8760]
    subarray1_dc_gross = pysam_res_dict["subarray1_dc_gross"][:8760]

    timeseries_index = weather_df.index

    ground_irradiance_values = da.from_array([row[1:] for row in ground_irradiance[1:]])

    single_location_ds = xr.Dataset(
        data_vars={
            # SCALARS
            # for some configs these are caluclated with inspire_practical_pitch
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
            "temp_air": (("time", ), da.array(weather_df["temp_air"].values)),
            "wind_speed": (("time", ), da.array(weather_df["wind_speed"].values)),
            "wind_direction": (("time", ), da.array(
                weather_df["wind_direction"].values
            )),
            "dhi": (("time", ), da.array(weather_df["dhi"].values)),
            "ghi": (("time", ), da.array(weather_df["ghi"].values)),
            "dni": (("time", ), da.array(weather_df["dni"].values)),
            "relative_humidity": (("time", ), da.array(
                weather_df["relative_humidity"].values
            )),
            "albedo": (("time", ), da.array(weather_df["albedo"].values)),

            # SPATIO-TEMPORAL (model outputs)
            "ground_irradiance": (("time", "distance"), ground_irradiance_values),
        },
        coords={
            "time": timeseries_index,
            # distances vary for config and locations (on fixed tilt configs)
            # so we need to use a distance "index" that is not spatially meaningful
            # convient way to match the distances in the template
            "distance": np.arange(10),
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
    "tilt": scalar,
    "pitch": scalar,
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


def inspire_ground_irradiance(weather_df, meta, config_files):
    """
    Get ground irradiance array and annual poa irradiance
    for a given point using pvsamv1

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
    if not isinstance(weather_df, pd.DataFrame) or not isinstance(meta, dict):
        raise ValueError(
            f"""
            weather_df must be pandas DataFrame, meta must be dict.
            weather_df type : {type(weather_df)}
            meta type : {type(meta)}
        """)

    # there is no pitch/gcr output from the model so we might have to do other checks
    # to see that this is being applied correctly verify that our equations are correct.
    # plot the world, view to see if practical applications have been applied.

    # force localize utc from tmy to local time by moving rows
    weather_df = weather.roll_tmy(weather_df, meta)

    tracking_setups = ["01", "02", "03", "04", "05"]
    # fixed tilt setups calculate pitch/gcr as a function of latitude capped at 40 deg
    pratical_considerations_setups = ["06", "07", "08", "09"]
    # vertical tilt (fixed spacing) 10

    print(f"config file string: {config_files['pv']} -- debug")

    cw = 2  # collector width 2 [m]
    pratical_consideration = False
    if any(setup in config_files["pv"] for setup in pratical_considerations_setups):
        print(
            "setup with practical consieration detected, "
            "using pysam inspire_practical_consideration_pitch_tilt=True"
        )
        pratical_consideration = True
        tilt_used, pitch_used, gcr_used = inspire_practical_pitch(
            latitude=meta['latitude'], cw=cw
        )

    # why would this be not in
    # this should check in "10" is in the config files
    elif "10" in config_files["pv"]:
        print("using config 10 with vertical fixed tilt.")
        gcr_used = _load_gcr_from_config(config_files=config_files)
        print(f"gcr used: {gcr_used}")
        pitch_used = cw / gcr_used
        tilt_used = 90.0

    # conf 01- 05 using tracking, default gcr from pysam config
    elif any(setup in config_files["pv"] for setup in tracking_setups):
        print("SAT scenario, using -999.0 as tilt fill value")
        gcr_used = _load_gcr_from_config(config_files=config_files)
        # print(f"gcr used: {gcr_used}")
        pitch_used = cw / gcr_used
        # tracking doesnt have fixed tilt (use placeholder instead)
        tilt_used = -999.0

    else:
        # this is not portable because it is custom for the calculation
        raise ValueError(
            "Valid config not found, "
            "config name must contain setup name from 01-10"
        )

    outputs = pysam(
        weather_df=weather_df,
        meta=meta,
        pv_model="pvsamv1",
        config_files=config_files,
        # tell model to calculate practical tilt, pitch, gcr again inside function
        practical_pitch_tilt_considerations=pratical_consideration,
    )

    ds_result = _handle_pysam_return(
        pysam_res_dict=outputs, weather_df=weather_df, tilt=tilt_used, pitch=pitch_used
    )

    return ds_result


def solar_resource_dict(weather_df, meta):
    """
    Create a solar resource dict mapping from weather and metadata.

    Works on PVGIS and kestrel NSRDB (NOT PSM3 NSRDB from NSRDB api).
    """
    weather_df = utilities.add_time_columns_tmy(weather_df)  # only supports hourly data

    # enforce tmy scheme
    times = pd.date_range(start="2001-01-01", periods=8760, freq="1h")

    # all solar resource dict options
    # lat,lon,tz,elev,year,month,hour,minute,gh,dn,df,poa,tdry,twet,tdew,rhum,pres,snow,alb,aod,wspd,wdir
    sr = {
        "lat": meta["latitude"],
        "lon": meta["longitude"],
        "tz": meta["tz"] if "tz" in meta.keys() else 0,
        "elev": meta["altitude"],
        "year": list(times.year),  # list(weather_df['Year']),
        "month": list(times.month),
        "day": list(times.day),
        "hour": list(times.hour),
        "minute": list(times.minute),
        "gh": list(weather_df["ghi"]),
        "dn": list(weather_df["dni"]),
        "df": list(weather_df["dhi"]),
        "wspd": list(weather_df["wind_speed"]),
        "tdry": list(weather_df["temp_air"]),
        "alb": (
            list(weather_df["albedo"])
            if "albedo" in weather_df.columns.values
            else [0.2] * len(weather_df)
        ),
    }

    # if we have wind direction then add it
    if "wind_direction" in weather_df.columns.values:
        sr["wdir"] = list(weather_df["wind_direction"])

    return sr
