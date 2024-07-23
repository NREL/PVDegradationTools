import pvdeg
from pvdeg import TEST_DATA_DIR
import pickle
import pandas as pd
import numpy as np
import os

with open(r"C:\Users\tford\Downloads\summit-weather.pkl", 'rb') as f:
    GEO_WEATHER = pickle.load(f)

GEO_META = pd.read_csv(r"C:\Users\tford\Downloads\summit-meta.csv", index_col=0)

# refactor
def test_analysis_standoff():
    res_ds = pvdeg.geospatial.analysis(
        weather_ds=GEO_WEATHER,
        meta_df=GEO_META,
        func=pvdeg.standards.standoff,
    )

    data_var = res_ds["x"]

    # Stack the latitude and longitude coordinates into a single dimension
    # convert to dataframe, this can be done with xr.dataset.to_dataframe as well
    stacked = data_var.stack(z=("latitude", "longitude"))
    latitudes = stacked['latitude'].values
    longitudes = stacked['longitude'].values
    data_values = stacked.values
    combined_array = np.column_stack((latitudes, longitudes, data_values))

    res = pd.DataFrame(combined_array).dropna()
    ans = pd.read_csv(os.path.join(TEST_DATA_DIR, 'summit-standoff-res.csv'), index_col=0)
    res.columns = ans.columns

    pd.testing.assert_frame_equal(res, ans, check_dtype=False, check_names=False)
