import pvdeg
  
import os
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

geo_weather = xr.load_dataset(os.path.join(pvdeg.TEST_DATA_DIR, "summit-weather.nc"))
meta = pd.read_csv(os.path.join(pvdeg.TEST_DATA_DIR, "summit-meta.csv"), index_col=0)

# fill in dummy wind direction and albedo values
geo_weather = geo_weather.assign(wind_direction=geo_weather["temp_air"] * 0+0)
geo_weather = geo_weather.assign(albedo=geo_weather["temp_air"] * 0 + 0.2) 

# split into multiple tests?
def test_pysam_inspire_practical_tilt():

    mid_lat = meta.iloc[0].to_dict()

    res = pvdeg.pysam.pysam(
        weather_df = geo_weather.isel(gid=0).to_dataframe()[::2],
        meta=mid_lat,
        config_files={'pv':os.path.join(pvdeg.TEST_DATA_DIR, Path("SAM/08/08_pvsamv1.json"))},
        pv_model='pvsamv1',
        inspire_practical_pitch_tilt=True,
    )

    # use latitude tilt under 40 deg N latitude
    assert mid_lat['latitude'] == max(res["subarray1_surf_tilt"])

    high_lat = meta.iloc[0].to_dict()
    high_lat['latitude'] = 45 # cant use latitude tilt above 40 deg N

    res = pvdeg.pysam.pysam(
        weather_df = geo_weather.isel(gid=0).to_dataframe()[::2],
        meta=high_lat,
        config_files={'pv':os.path.join(pvdeg.TEST_DATA_DIR, Path("SAM/08/08_pvsamv1.json"))},
        pv_model='pvsamv1',
        inspire_practical_pitch_tilt=True,
    )

    # latitude point is above 40 deg N, so we floor to 40 deg tilt due to practical racking considerations
    assert 40 == max(res["subarray1_surf_tilt"])

    res = pvdeg.pysam.pysam(
        weather_df = geo_weather.isel(gid=0).to_dataframe()[::2],
        meta=high_lat,
        config_files={'pv':os.path.join(pvdeg.TEST_DATA_DIR, Path("SAM/08/08_pvsamv1.json"))},
        pv_model='pvsamv1',
        inspire_practical_pitch_tilt=False,
    )

    # flag is set to false, ignore practical considerations
    assert 45 == max(res["subarray1_surf_tilt"])

def test_inspire_configs_pitches():

    configs = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    config_paths=[os.path.join(pvdeg.TEST_DATA_DIR, Path(f"SAM/{conf}/{conf}_pvsamv1.json")) for conf in configs]

    tracking = {"01", "02", "03", "04", "05"}
    fixed = {"06", "07", "08", "09"}
    fixed_vertical = {"10"}

    mid_lat = meta.iloc[0].to_dict()

    high_lat = meta.iloc[0].to_dict()
    high_lat['latitude'] = 45

    for conf in config_paths:

        config_files = {"pv": conf}

        res_mid_lat = pvdeg.pysam.inspire_ground_irradiance(
            weather_df=geo_weather.isel(gid=0).to_dataframe()[::2],
            meta=mid_lat,
            config_files=config_files
        )

        res_high_lat = pvdeg.pysam.inspire_ground_irradiance(
            weather_df=geo_weather.isel(gid=0).to_dataframe()[::2],
            meta=high_lat,
            config_files=config_files
        )

        cw = 2 # collector width [m]

        if conf in tracking:
            # tracking, we leave the pitch unchanged from the original sam config
            assert res_mid_lat.pitch.item() == (pvdeg.pysam.load_gcr_from_config(config_files=config_files) / cw) # use original pitch
            assert res_high_lat.pitch.item() == (pvdeg.pysam.load_gcr_from_config(config_files=config_files) / cw) # use original pitch

            # tracking does not have fixed tilt
            assert res_mid_lat.tilt.item() == -999
            assert res_mid_lat.tilt.item() == -999

        elif conf in fixed:
            # 
            tilt, pitch, gcr = pvdeg.pysam.inspire_practical_pitch(latitude=meta["latitude"], cw=cw)

            assert res_mid_lat.pitch.item() == pitch
            assert res_high_lat.pitch.item() == 12 # max of 12 meters

            assert res_mid_lat.tilt == res_mid_lat['latitude']
            assert res_high_lat.tilt == 40 # no latitude tilt above 40

        elif conf in fixed_vertical:
            assert res_mid_lat.pitch.item() == (pvdeg.pysam.load_gcr_from_config(config_files=config_files) / cw) # use original pitch
            assert res_high_lat.pitch.item() == (pvdeg.pysam.load_gcr_from_config(config_files=config_files) / cw) # use original pitch

            assert res_mid_lat.tilt == 90 # fixed vertical tilt
            assert res_high_lat.tilt == 90