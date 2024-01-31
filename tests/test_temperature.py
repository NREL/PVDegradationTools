import os
import pandas as pd
import pvdeg
from pvdeg import TEST_DATA_DIR
import json

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

# Load expected results
modtemp_expected = input_data["module_temp"]
celltemp_expected = input_data["cell_temp"]


def test_module():
    result = pvdeg.temperature.module(
        WEATHER,
        META,
        poa,
        temp_model="sapm",
        conf="open_rack_glass_polymer",
        wind_speed_factor=1,
    )

    pd.testing.assert_series_equal(
        result, modtemp_expected, check_dtype=False, check_names=False
    )


def test_cell():
    result = pvdeg.temperature.cell(WEATHER, META, poa=poa)
    pd.testing.assert_series_equal(
        result, celltemp_expected, check_dtype=False, check_names=False
    )
