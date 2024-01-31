import os
import json
import pandas as pd
import pvdeg
from pvdeg import TEST_DATA_DIR

WEATHER = pd.read_csv(
    os.path.join(TEST_DATA_DIR, r"weather_day_pytest.csv"),
    index_col=0,
    parse_dates=True,
)

with open(os.path.join(TEST_DATA_DIR, "meta.json")) as file:
    META = json.load(file)

# Load expected results
results_expected = pd.read_csv(
    os.path.join(TEST_DATA_DIR, r"input_day_pytest.csv"), index_col=0, parse_dates=True
)
solpos = [
    "apparent_zenith",
    "zenith",
    "apparent_elevation",
    "elevation",
    "azimuth",
    "equation_of_time",
]
poa = [col for col in results_expected.columns if "poa" in col]
solpos_expected = results_expected[solpos]
poa_expected = results_expected[poa]


def test_solar_position():
    """
    test pvdeg.spectral.solar_position

    Requires:
    ---------
    weather dataframe and meta dictionary
    """
    result = pvdeg.spectral.solar_position(WEATHER, META)
    pd.testing.assert_frame_equal(result, solpos_expected, check_dtype=False)


def test_poa_irradiance():
    """
    test pvdeg.spectral.poa_irradiance

    Requires:
    ---------
    weather dataframe, meta dictionary, and solar_position dataframe
    """
    result = pvdeg.spectral.poa_irradiance(
        WEATHER, META, solpos_expected, tilt=None, azimuth=180, sky_model="isotropic"
    )

    pd.testing.assert_frame_equal(result, poa_expected, check_dtype=False)
