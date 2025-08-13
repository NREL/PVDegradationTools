"""Pysam Integration for pvdeg, supports single site and geospatial calculations.

Produced to support Inspire Agrivoltaics: https://openei.org/wiki/InSPIRE
"""

import dask.array as da
import pandas as pd
import xarray as xr
import numpy as np
import json


from pvdeg import weather, utilities


# TODO: add grid_default, cashloan_default, utilityrate_defaults for expanded pysam
# simulation capabilities
def pysam(
    weather_df: pd.DataFrame,
    meta: dict,
    pv_model: str,
    pv_model_default: str = None,
    config_files: dict[str:str] = None,
    results: list[str] = None,
) -> dict:
    """Run pySam simulation.

    Only works with pysam weather.

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

            options: ``pvwatts8``, ``pysamv1``, etc.

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

    # https://nrel-pysam.readthedocs.io/en/main/modules/Pvwattsv8.html
    # https://nrel-pysam.readthedocs.io/en/main/modules/Pvsamv1.html
    model_map = {
        "pvwatts8": pv8,
        "pysamv1": pv1,
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

        for k, v in pv_inputs.items():
            if k not in ({"number_inputs", "solar_resource_file"} | bad_parameters):
                pysam_model.value(k, v)

    pysam_model.unassign("solar_resource_file")  # unassign file

    # Duplicate Columns in the dataframe seem to cause this issue
    # Error (-4) converting nested tuple 0 into row in matrix.
    pysam_model.SolarResource.solar_resource_data = sr
    pysam_model.execute()
    outputs = pysam_model.Outputs.export()

    if not results:
        return outputs

    pysam_res = {key: outputs[key] for key in results}
    return pysam_res


# class inspirePysamReturn():
#     """simple struct to facilitate handling weirdly shaped pysam simulation
#        return values"""

#     # removes __dict__ atribute and breaks pickle
#     # __slots__ = ("annual_poa", "ground_irradiance", "timeseries_index")

#     def __init__(self, annual_poa, ground_irradiance, timeseries_index, annual_energy,
#                   poa_front, poa_rear, subarray1_poa_front, subarray1_poa_rear):
#         self.annual_energy = annual_energy
#         self.annual_poa = annual_poa
#         self.ground_irradiance = ground_irradiance
#         self.timeseries_index = timeseries_index
#         self.poa_front = poa_front
#         self.poa_rear = poa_rear
#         self.subarray1_poa_front = subarray1_poa_front
#         self.subarray1_poa_rear = subarray1_poa_rear


# def _handle_pysam_return(pysam_res : inspirePysamReturn) -> xr.Dataset:
def _handle_pysam_return(pysam_res_dict: dict, weather_df: pd.DataFrame) -> xr.Dataset:
    """Handle a pysam return object and transform it to an xarray."""
    ground_irradiance = pysam_res_dict["subarray1_ground_rear_spatial"]

    annual_poa = pysam_res_dict["annual_poa_front"]
    annual_energy = pysam_res_dict["annual_energy"]

    poa_front = pysam_res_dict["poa_front"][
        :8760
    ]  # 25 * 8760 entries, all pairs of 8760 entries are identical
    poa_rear = pysam_res_dict["poa_rear"][:8760]  # same for the following
    subarray1_poa_front = pysam_res_dict["subarray1_poa_front"][:8760]
    subarray1_poa_rear = pysam_res_dict["subarray1_poa_rear"][:8760]

    timeseries_index = weather_df.index

    # redo this using numba?
    # distances = ground_irradiance[0][1:]
    ground_irradiance_values = da.from_array([row[1:] for row in ground_irradiance[1:]])

    single_location_ds = xr.Dataset(
        data_vars={
            # scalars
            "annual_poa": annual_poa,
            "annual_energy": annual_energy,
            # simple timeseries
            # which poa do we want to use, can elimiate one pair to save memory
            "poa_front": (("time",), da.array(poa_front)),
            "poa_rear": (("time",), da.array(poa_rear)),
            "subarray1_poa_front": (("time",), da.array(subarray1_poa_front)),
            "subarray1_poa_rear": (("time",), da.array(subarray1_poa_rear)),
            # spatio-temporal
            "ground_irradiance": (("time", "distance"), ground_irradiance_values),
        },
        coords={
            "time": timeseries_index,
            # "distance" : distances,
            # would be convient to define distances after being calculated
            # by pysam but we need to know ahead of time to create the template
            "distance": np.arange(
                10
            ),  # convient way to match the distances in the template
        },
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


def inspire_ground_irradiance(weather_df, meta, config_files):
    """Get ground irradiance array and annual poa irradiance.

    Get ground irradiance array and annual poa irradiance for a given point using
    pvsamv1.

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
        """
        )

    # force localize utc from tmy to local time by moving rows
    weather_df = weather.roll_tmy(weather_df, meta)

    outputs = pysam(
        weather_df=weather_df,
        meta=meta,
        pv_model="pysamv1",
        config_files=config_files,
    )

    ds_result = _handle_pysam_return(pysam_res_dict=outputs, weather_df=weather_df)

    return ds_result


def solar_resource_dict(weather_df, meta):
    """Create a solar resource dict mapping from weather and metadata.

    Works on PVGIS and appears to work on NSRDB (NOT PSM3).
    """
    # weather_df = weather_df.reset_index(drop=True) # Probably dont need to do this
    weather_df = utilities.add_time_columns_tmy(weather_df)  # only supports hourly data

    # enforce tmy scheme
    times = pd.date_range(start="2001-01-01", periods=8760, freq="1h")

    # all options
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


def sample_inspire_result(weather_df, meta):  # throw weather, meta away
    """Return a sample inspire_ground_irradiance xarray.

    Dataset for geospatial
    testing. Weather_df and meta exist to provide a homogenous arugment structure for
    geospatial calculations but are not used.

    Parameters
    ----------
    weather: pd.Dataframe
        weather dataframe, is thrown away
    meta: dict
        metadata dictionary, is thrown away

    Returns
    -------
    inspire_ground_irradiance: xr.Dataset
        returns an xarray dataset of the same shape generated by
        inpspire_ground_irradiance()
    """
    return xr.Dataset(
        data_vars={
            "poa_rear": (("time",), np.zeros((8760,))),
            "ground_irradiance": (("time", "distance"), np.zeros((8760, 10))),
            "annual_energy": ((), 0),
            "annual_poa": ((), 0),
            "subarray1_poa_front": (("time",), np.zeros((8760,))),
            "subarray1_poa_rear": (("time",), np.zeros((8760,))),
            "poa_front": (("time",), np.zeros((8760,))),
        },
        coords={
            "time": pd.date_range(start="2001-01-01 00:30:00", periods=8760, freq="h"),
            "distance": np.arange(10),
        },
    )


def ground_irradiance_monthly(inspire_res_ds: xr.Dataset) -> xr.Dataset:
    """Drop the rows and calculate the monthly average irradiance at each distance.

    Many rows are not populated because the model only calculates ground irradiance
    when certain measurements are met.
    """
    nonzero_mask = (inspire_res_ds["ground_irradiance"] != 0).any(dim="distance")
    filtered_data = inspire_res_ds["ground_irradiance"].where(nonzero_mask, drop=True)

    monthly_avg_ground_irradiance = filtered_data.groupby(
        filtered_data.time.dt.month
    ).mean()
    return monthly_avg_ground_irradiance
