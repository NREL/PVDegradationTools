import pandas as pd
import pytest
from pvdeg.weather import map_meta

def test_map_meta_dict():
    meta = {
        "Elevation": 150,
        "Time Zone": "UTC-7",
        "Longitude": -120.5,
        "Latitude": 38.5,
        "SomeKey": "value"
    }
    mapped = map_meta(meta)
    assert "altitude" in mapped and mapped["altitude"] == 150
    assert "tz" in mapped and mapped["tz"] == "UTC-7"
    assert "longitude" in mapped and mapped["longitude"] == -120.5
    assert "latitude" in mapped and mapped["latitude"] == 38.5
    assert "SomeKey" in mapped  # unchanged keys remain

def test_map_meta_dataframe():
    df = pd.DataFrame({
        "Elevation": [100, 200],
        "Time Zone": ["UTC-5", "UTC-6"],
        "Longitude": [-80, -81],
        "Latitude": [35, 36],
        "ExtraCol": [1, 2]
    })
    mapped_df = map_meta(df)
    assert "altitude" in mapped_df.columns
    assert "tz" in mapped_df.columns
    assert "longitude" in mapped_df.columns
    assert "latitude" in mapped_df.columns
    assert "ExtraCol" in mapped_df.columns
    # Original column names should not exist
    assert "Elevation" not in mapped_df.columns
    assert "Time Zone" not in mapped_df.columns

def test_map_meta_invalid_input():
    with pytest.raises(TypeError):
        map_meta(["invalid", "input"])
