import pvdeg
from pvdeg import TEST_DATA_DIR
import os
import json
import pandas as pd
import hashlib
import json

# Load weather data
WEATHER = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "weather_year_pytest.csv"),
    index_col=0,
    parse_dates=True,
)

with open(os.path.join(TEST_DATA_DIR, "meta.json"), "r") as file:
    META = json.load(file)

def test_pysam_pvwatts8():

    res = pvdeg.pysam.pysam(
        weather_df=WEATHER.iloc[::6], # downselect to hourly data
        meta=META,
        pv_model="pvwatts8",
        pv_model_default="FuelCellCommercial",
    )

    serialized = json.dumps(res, sort_keys=True)
    hashed_res = hashlib.sha256(serialized.encode()).hexdigest()

    assert hashed_res == '36aed0c69d65c8a06807754744093bcdf1a8a9260ab7970880df76b57f4bbad3'

# def test_pysam_pvsamv1():

#     res = pvdeg.pysam.pysam(
#         weather_df=WEATHER.iloc[::6],
#         meta=META,
#         pv_model="pvsamv1",
#         pv_model_default="FlatPlatePVCommercial",
#     )

#     serialized = json.dumps(res, sort_keys=True)
#     hashed_res = hashlib.sha256(serialized.encode()).hexdigest()

#     assert hashed_res == "d0e5368157e38c20a26717f67b820622f7e419e2557a62781ca51dfcf88319bf"

