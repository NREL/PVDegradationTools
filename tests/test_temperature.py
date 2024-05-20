"""
Using pytest to create unit tests for pvdeg

to run unit tests, run pytest from the command line in the pvdeg directory
to run coverage tests, run py.test --cov-report term-missing --cov=pvdeg
"""

import os
import pandas as pd
import pvdeg
from pvdeg import TEST_DATA_DIR
import json
import pvlib.temperature

with open(os.path.join(TEST_DATA_DIR, "meta.json"), "r") as j:
    META = json.load(j)

# Load weather data
WEATHER = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "weather_day_pytest.csv"), index_col=0, parse_dates=True
)

# Load input dataframes
input_data = pd.read_csv(
    os.path.join(TEST_DATA_DIR, r"input_day_pytest.csv"), index_col=0, parse_dates=True
)
poa = [col for col in input_data.columns if "poa" in col]
poa = input_data[poa]

# load expected results
sapm_modtemp_expected = input_data["module_temp"]
sapm_celltemp_expected = input_data["cell_temp"]

TEMPERATURE_MODELS_RESULTS = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "temperatures.csv"), index_col=0, parse_dates=True
)

def test_module():
    result = pvdeg.temperature.module(
        WEATHER,
        META,
        poa,
        temp_model="sapm",
        conf="open_rack_glass_polymer",
        wind_factor=0,
    )

    pd.testing.assert_series_equal(
        result, sapm_modtemp_expected, check_dtype=False, check_names=False
    )


def test_cell():
    result = pvdeg.temperature.cell(
        WEATHER,
        META,
        poa=poa,
        temp_model="sapm",
        conf="open_rack_glass_polymer",
        wind_factor=0,
    )
    pd.testing.assert_series_equal(
        result, sapm_celltemp_expected, check_dtype=False, check_names=False
    )

# can we check against pvlib outputs
def test_temperature():

    weather_args = {'poa_global':poa['poa_global'], 'temp_air':WEATHER['temp_air'], 'wind_speed':WEATHER['wind_speed'], 'wind_factor':0}

    result_df = pd.DataFrame()

    result_df['sapm_cell'] = pvdeg.temperature.temperature(
        cell_or_mod='cell',
        temp_model='sapm',
        conf='open_rack_glass_polymer',
        weather_df=WEATHER,
        meta=META,
        poa=poa,
        wind_factor=0
    )

    result_df['sapm_module'] = pvdeg.temperature.temperature(
        cell_or_mod='mod',
        temp_model='sapm',
        conf='open_rack_glass_polymer',
        weather_df=WEATHER,
        meta=META,
        poa=poa,
        wind_factor=0
    )

    result_df['pvsyst_cell'] = pvdeg.temperature.temperature(
        cell_or_mod='cell',
        temp_model='pvsyst',
        weather_df=WEATHER,
        meta=META,
        poa=poa,
        conf='freestanding', # different configurations
        wind_factor=0
    )

    result_df['faiman'] = pvdeg.temperature.temperature(
        temp_model='faiman', 
        weather_df=WEATHER,
        meta=META,
        poa=poa,
        wind_factor=0   
    )

    result_df['faiman_rad'] = pvdeg.temperature.temperature(
        temp_model='faiman_rad', 
        weather_df=WEATHER,
        meta=META,
        poa=poa,
        wind_factor=0   
    )

    noct = pvlib.temperature.sapm_cell(poa_global=800, temp_air=20, wind_speed=1, **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer'])
    result_df['ross'] = pvdeg.temperature.temperature(
        temp_model='ross',
        weather_df=WEATHER,
        meta=META,
        poa=poa,
        wind_factor=0,
        model_kwarg={'noct' : noct}
    )

    result_df['noct_sam'] = pvdeg.temperature.temperature(
        temp_model='noct_sam',
        weather_df=WEATHER,
        meta=META,
        poa=poa,
        wind_factor=0,
        model_kwarg={'noct' : noct, 'module_efficiency' : 0.1875} # 18.75% at 300 [W] / (1000 [W/m^2] * 1.6 [m^2])
    )

    result_df['fuentes'] = pvdeg.temperature.temperature(
        temp_model='fuentes',
        weather_df=WEATHER,
        meta=META,
        poa=poa,
        wind_factor=0,
        model_kwarg={'noct_installed' : 45} # 45 [c] for freestanding modules
    )
    
    # pvlib call for the same function
    # res = pvlib.temperature.generic_linear(
    #     poa_global=poa['poa_global'],
    #     temp_air=WEATHER['temp_air'],
    #     wind_speed=WEATHER['wind_speed'],
    #     u_const=glm.__dict__['u_const'],
    #     du_wind=glm.__dict__['du_wind'],
    #     module_efficiency=0.1875,
    #     absorptance=0.85
    # )

    glm = pvlib.temperature.GenericLinearModel(
        module_efficiency=0.1875,
        absorptance=0.85)
    glm.use_faiman(16, 8)

    result_df['generic_linear'] = pvdeg.temperature.temperature(
        temp_model='generic_linear',
        weather_df=WEATHER,
        meta=META,
        poa=poa,
        wind_factor=0,
        model_kwarg={
            'u_const' : glm.__dict__['u_const'],
            'du_wind' : glm.__dict__['du_wind'],
            'module_efficiency' : 0.1875,
            'absorptance' : 0.85,
        }
    )

    pd.testing.assert_frame_equal(result_df, TEMPERATURE_MODELS_RESULTS, check_dtype=False,check_names=False,check_like=False)
