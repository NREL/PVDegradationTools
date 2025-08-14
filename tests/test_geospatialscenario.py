import pvdeg
import pandas as pd
import xarray as xr
import numpy as np
import os


def monkeypatch_addLocation(self, *args, **kwargs) -> None:
    """
    Mocker function to be monkey patched at runtime for Scenario.addLocation.

    Avoids psm3 api calls and uses local weather files instead.
    """

    self.gids, self.weather_data, self.meta_data = None, None, None

    GEO_META = pd.read_csv(
        os.path.join(pvdeg.TEST_DATA_DIR, "summit-meta.csv"), index_col=0
    )
    GEO_WEATHER = xr.load_dataset(
        os.path.join(pvdeg.TEST_DATA_DIR, "summit-weather.nc")
    )

    self.weather_data = GEO_WEATHER
    self.meta_data = GEO_META
    self.gids = GEO_WEATHER.gid.values


def test_standoff_autotemplate(monkeypatch, tmp_path):
    monkeypatch.setattr(
        target=pvdeg.GeospatialScenario,
        name="addLocation",
        value=monkeypatch_addLocation,
    )

    # Create a scenario, add locations, run analysis using an autotemplated function
    geo_scenario = pvdeg.GeospatialScenario(
        path=tmp_path,
    )
    geo_scenario.addLocation()

    geo_scenario.addJob(
        func=pvdeg.standards.standoff,
    )

    geo_scenario.run()
    # End scenario run

    data_var = geo_scenario.results["x"]
    # Stack the latitude and longitude coordinates into a single dimension
    # convert to dataframe, this can be done with xr.dataset.to_dataframe
    stacked = data_var.stack(z=("latitude", "longitude"))
    latitudes = stacked["latitude"].values
    longitudes = stacked["longitude"].values
    data_values = stacked.values
    combined_array = np.column_stack((latitudes, longitudes, data_values))

    res = pd.DataFrame(combined_array).dropna()
    ans = pd.read_csv(
        os.path.join(pvdeg.TEST_DATA_DIR, "summit-standoff-res.csv"), index_col=0
    )
    res.columns = ans.columns

    pd.testing.assert_frame_equal(res, ans)


def test_geospatial_data(monkeypatch, tmp_path):
    GEO_META = pd.read_csv(
        os.path.join(pvdeg.TEST_DATA_DIR, "summit-meta.csv"), index_col=0
    )
    GEO_WEATHER = xr.load_dataset(
        os.path.join(pvdeg.TEST_DATA_DIR, "summit-weather.nc")
    )

    monkeypatch.setattr(
        target=pvdeg.GeospatialScenario,
        name="addLocation",
        value=monkeypatch_addLocation,
    )

    geo_scenario = pvdeg.GeospatialScenario(
        path=tmp_path,
    )
    geo_scenario.addLocation()

    scenario_weather, scenario_meta = geo_scenario.geospatial_data

    xr.testing.assert_equal(GEO_WEATHER, scenario_weather)
    pd.testing.assert_frame_equal(GEO_META, scenario_meta)


def test_downselect_elevation_stochastic_no_kdtree(monkeypatch, tmp_path):
    monkeypatch.setattr(
        target=pvdeg.GeospatialScenario,
        name="addLocation",
        value=monkeypatch_addLocation,
    )

    np.random.seed(0)

    geo_scenario = pvdeg.GeospatialScenario(
        path=tmp_path,
    )
    geo_scenario.addLocation()

    geo_scenario.downselect_elevation_stochastic(
        downselect_prop=0.8,
        k_neighbors=3,
        method="mean",
        normalization="linear",
        kdtree=None,  # the scenario object will create its own kdtree
    )

    remaining_gids = np.array([453020, 454916, 455867, 455877, 457776], dtype=int)

    np.testing.assert_array_equal(geo_scenario.gids, remaining_gids)
    assert geo_scenario.kdtree is not None


def test_downselect_elevation_stochastic_kdtree(monkeypatch, tmp_path):
    monkeypatch.setattr(
        target=pvdeg.GeospatialScenario,
        name="addLocation",
        value=monkeypatch_addLocation,
    )

    np.random.seed(0)

    geo_scenario = pvdeg.GeospatialScenario(
        path=tmp_path,
    )
    geo_scenario.addLocation()

    tree = pvdeg.geospatial.meta_KDtree(meta_df=geo_scenario.meta_data, leaf_size=40)

    geo_scenario.downselect_elevation_stochastic(
        downselect_prop=0.8,
        k_neighbors=3,
        method="mean",
        normalization="linear",
        kdtree=tree,  # we create and provide a kdtree
    )

    remaining_gids = np.array([453020, 454916, 455867, 455877, 457776], dtype=int)

    np.testing.assert_array_equal(geo_scenario.gids, remaining_gids)
    assert geo_scenario.kdtree == tree


def test_gid_downsample(monkeypatch, tmp_path):
    monkeypatch.setattr(
        target=pvdeg.GeospatialScenario,
        name="addLocation",
        value=monkeypatch_addLocation,
    )

    geo_scenario = pvdeg.GeospatialScenario(
        path=tmp_path,
    )
    geo_scenario.addLocation()

    original_meta = geo_scenario.meta_data
    remaining_gids = np.array([455867, 455877, 457776, 460613], dtype=int)

    geo_scenario.gid_downsample(1)

    np.testing.assert_array_equal(geo_scenario.gids, remaining_gids)
    pd.testing.assert_frame_equal(
        geo_scenario.meta_data, original_meta.loc[remaining_gids]
    )

    pd.testing.assert_frame_equal(
        geo_scenario.meta_data, original_meta.loc[remaining_gids]
    )


def test_downselect_CONUS(monkeypatch, tmp_path):
    monkeypatch.setattr(
        target=pvdeg.GeospatialScenario,
        name="addLocation",
        value=monkeypatch_addLocation,
    )

    geo_scenario = pvdeg.GeospatialScenario(
        path=tmp_path,
    )
    geo_scenario.addLocation()

    co_df = geo_scenario.meta_data.copy()

    ak_hi_df = pd.DataFrame(
        data=[
            [-99, -99, -1, "+100", "United States", "Alaska", "filler", 2],
            [-99, -99, -1, "+100", "United States", "Hawaii", "filler", 2],
        ],
        columns=[
            "latitude",
            "longitude",
            "altitude",
            "tz",
            "country",
            "state",
            "county",
            "wind_height",
        ],
    )

    # add rows that contain points in alaska and hawaii
    geo_scenario.meta_data = pd.concat([geo_scenario.meta_data, ak_hi_df])

    geo_scenario.downselect_CONUS()

    pd.testing.assert_frame_equal(geo_scenario.meta_data, co_df, check_dtype=False)
    np.testing.assert_array_equal(geo_scenario.gids, co_df.index.values)


def test_coords(monkeypatch, tmp_path):
    monkeypatch.setattr(
        target=pvdeg.GeospatialScenario,
        name="addLocation",
        value=monkeypatch_addLocation,
    )

    geo_scenario = pvdeg.GeospatialScenario(
        path=tmp_path,
    )
    geo_scenario.addLocation()

    # coords is a property so we should test it, not just an attribute
    coords_res = geo_scenario.coords

    coords_correct = np.array(
        [
            [39.89, -106.42],
            [39.89, -106.3],
            [39.69, -106.26],
            [39.81, -106.18],
            [39.81, -106.14],
            [39.41, -106.14],
            [39.45, -106.1],
            [39.41, -106.06],
            [39.65, -105.98],
            [39.53, -105.94],
            [39.57, -105.86],
        ]
    )

    np.testing.assert_array_equal(coords_res, coords_correct)
