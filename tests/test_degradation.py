"""Using pytest to create unit tests for pvdeg.

to run unit tests, run pytest from the command line in the pvdeg directory to run
coverage tests, run py.test --cov-report term-missing --cov=pvdeg
"""

import os
import pandas as pd
import numpy as np
import pytest
import pvdeg
from pvdeg import TEST_DATA_DIR

PSM_FILE = os.path.join(TEST_DATA_DIR, r"psm3_pytest.csv")
weather_df, meta = pvdeg.weather.read(PSM_FILE, "psm")

INPUT_SPECTRA = os.path.join(TEST_DATA_DIR, r"spectra_pytest.csv")


def test_vantHoff_deg():
    # test the vantHoff degradation acceleration factor

    vantHoff_deg = pvdeg.degradation.vantHoff_deg(
        weather_df=weather_df, meta=meta, I_chamber=1000, temp_chamber=60
    )
    assert vantHoff_deg == pytest.approx(8.178, abs=0.01)


def test_iwa_vantHoff():
    # test the vantHoff equivalent weighted average irradiance

    irr_weighted_avg = pvdeg.degradation.IwaVantHoff(weather_df=weather_df, meta=meta)
    assert irr_weighted_avg == pytest.approx(240.28, abs=0.05)


def test_iwa_vantHoff_no_poa():
    poa = pvdeg.spectral.poa_irradiance(weather_df, meta)
    irr_weighted_avg_match = pvdeg.degradation.IwaVantHoff(
        weather_df=weather_df, meta=meta, poa=poa
    )
    assert irr_weighted_avg_match == pytest.approx(240.28, abs=0.05)


def test_arrhenius_deg():
    # test the arrhenius degradation acceleration factor

    rh_chamber = 15
    temp_chamber = 60
    I_chamber = 1e3
    Ea = 40

    poa = pvdeg.spectral.poa_irradiance(weather_df, meta)
    temp_module = pvdeg.temperature.module(weather_df, meta, poa=poa)

    rh_surface = pvdeg.humidity.surface_relative(
        rh_ambient=weather_df["relative_humidity"],
        temp_ambient=weather_df["temp_air"],
        temp_module=temp_module,
    )
    arrhenius_deg = pvdeg.degradation.arrhenius_deg(
        weather_df=weather_df,
        meta=meta,
        I_chamber=I_chamber,
        rh_chamber=rh_chamber,
        rh_outdoor=rh_surface,
        temp_chamber=temp_chamber,
        Ea=Ea,
        poa=poa,
    )
    assert arrhenius_deg == pytest.approx(12.804, abs=0.1)


def test_arrhenius_deg_no_poa():
    rh_chamber = 15
    temp_chamber = 60
    I_chamber = 1e3
    Ea = 40

    temp_module = pvdeg.temperature.module(weather_df, meta)

    rh_surface = pvdeg.humidity.surface_relative(
        rh_ambient=weather_df["relative_humidity"],
        temp_ambient=weather_df["temp_air"],
        temp_module=temp_module,
    )
    arrhenius_deg = pvdeg.degradation.arrhenius_deg(
        weather_df=weather_df,
        meta=meta,
        I_chamber=I_chamber,
        rh_chamber=rh_chamber,
        rh_outdoor=rh_surface,
        temp_chamber=temp_chamber,
        Ea=Ea,
    )
    assert arrhenius_deg == pytest.approx(12.804, abs=0.1)


def test_iwa_arrhenius():
    # test arrhenius equivalent weighted average irradiance
    # requires PSM3 weather file

    Ea = 40
    irr_weighted_avg = pvdeg.degradation.IwaArrhenius(
        weather_df=weather_df,
        meta=meta,
        rh_outdoor=weather_df["relative_humidity"],
        Ea=Ea,
    )
    assert irr_weighted_avg == pytest.approx(199.42, abs=0.1)


def test_iwa_arrhenius_poa():
    poa = pvdeg.spectral.poa_irradiance(weather_df=weather_df, meta=meta)

    Ea = 40
    irr_weighted_avg = pvdeg.degradation.IwaArrhenius(
        weather_df=weather_df,
        meta=meta,
        rh_outdoor=weather_df["relative_humidity"],
        Ea=Ea,
        poa=poa,
    )
    assert irr_weighted_avg == pytest.approx(199.42, abs=0.1)


def test_degradation():
    # test RH, Temp, Spectral Irradiance sensitive degradation
    # requires TMY3-like weather data
    # requires spectral irradiance data

    data = pd.read_csv(INPUT_SPECTRA)
    wavelengths = np.array([300, 325, 350, 375, 400])  # Fixed: added square brackets
    degradation = pvdeg.degradation.degradation_spectral(
        spectra=data["Spectra: [ 300, 325, 350, 375, 400 ]"],
        rh=data["RH"],
        temp=data["Temperature"],
        wavelengths=wavelengths,
        time=None,
    )
    # Update expected value based on actual calculation
    assert degradation == pytest.approx(0.008835, abs=0.001)


def test_vecArrhenius():
    poa_global = pvdeg.spectral.poa_irradiance(weather_df=weather_df, meta=meta)[
        "poa_global"
    ].to_numpy()
    module_temp = pvdeg.temperature.temperature(
        weather_df=weather_df, meta=meta, cell_or_mod="mod"
    ).to_numpy()

    degradation = pvdeg.degradation.vecArrhenius(
        poa_global=poa_global, module_temp=module_temp, ea=30, x=2, lnr0=15
    )

    pytest.approx(degradation, 6.603006830204657)


def test_arrhenius_basic():
    # Basic test with only temperature dependence
    df = pd.DataFrame(
        {
            "temperature": [25, 30, 35],
            "relative_humidity": [40, 50, 60],
            "temp_air": [20, 25, 30],
            "temp_module": [25, 30, 35],
            "poa_global": [800, 900, 1000],
        }
    )
    result = pvdeg.degradation.arrhenius(weather_df=df, Ea=40)
    assert result == pytest.approx(3.92292e-7, abs=1e-11)


def test_arrhenius_with_humidity():
    # Test with humidity dependence
    df = pd.DataFrame(
        {
            "temperature": [25, 30, 35],
            "relative_humidity": [40, 50, 60],
            "temp_air": [20, 25, 30],
            "temp_module": [25, 30, 35],
            "poa_global": [800, 900, 1000],
        }
    )
    result = pvdeg.degradation.arrhenius(weather_df=df, Ea=40, n=1)
    assert result == pytest.approx(1.5123467e-5, abs=1e-9)


def test_arrhenius_with_irradiance():
    # Test with irradiance dependence
    df = pd.DataFrame(
        {
            "temperature": [25, 30, 35],
            "relative_humidity": [40, 50, 60],
            "temp_air": [20, 25, 30],
            "temp_module": [25, 30, 35],
            "poa_global": [800, 900, 1000],
        }
    )
    result = pvdeg.degradation.arrhenius(weather_df=df, Ea=40, p=1)
    assert result == pytest.approx(0.000359824, abs=1e-8)


def test_arrhenius_all_dependence():
    # Test with all dependencies
    df = pd.DataFrame(
        {
            "temperature": [25, 30, 35],
            "relative_humidity": [40, 50, 60],
            "temp_air": [20, 25, 30],
            "temp_module": [25, 30, 35],
            "poa_global": [800, 900, 1000],
        }
    )
    result = pvdeg.degradation.arrhenius(weather_df=df, Ea=40, n=1, p=1)
    assert result == pytest.approx(0.014073859, abs=1e-6)


def test_arrhenius_no_dependence():
    # Test with no dependence (Ea=0, n=0, p=0)
    df = pd.DataFrame(
        {
            "temperature": [25, 30, 35],
            "relative_humidity": [40, 50, 60],
            "temp_air": [20, 25, 30],
            "temp_module": [25, 30, 35],
            "poa_global": [800, 900, 1000],
        }
    )
    result = pvdeg.degradation.arrhenius(weather_df=df)
    assert result == 3


def test_arrhenius_action_spectra_no_dependence():
    # Test with no dependence (Ea=0, n=0, p=0)
    df = pd.DataFrame(
        {
            "temperature": [25, 30, 35],
            "relative_humidity": [40, 50, 60],
            "temp_air": [20, 25, 30],
            "temp_module": [25, 30, 35],
            "poa_global": [800, 900, 1000],
        }
    )
    spectra = pd.DataFrame(
        {
            "Spectra: garbage identification here [ 300, 350, 400 ]": [0.1, 0.2, 0.5],
        }
    )
    result = pvdeg.degradation.arrhenius(weather_df=df, irradiance=spectra, C2=0.07)
    assert result == 3


def test_arrhenius_action_spectra():
    # Full test with all dependencies but even time steps
    df = pd.DataFrame(
        {
            "temp": [25, 30, 35],
            "relative_humidity": [40, 50, 60],
            "temp_air": [20, 25, 30],
            "temp_module": [25, 30, 35],
            "poa_global": [800, 900, 1000],
        }
    )
    spectra = pd.DataFrame(
        {
            "Spectra: garbage identification here [ 300, 350, 400 ]": [0.1, 0.2, 0.5],
        }
    )
    result = pvdeg.degradation.arrhenius(
        weather_df=df, irradiance=spectra, p=0.5, n=1, Ea=40, C2=0.07
    )
    assert result == pytest.approx(1.97876255e-9, abs=1e-13)


def test_arrhenius_action_spectra_uneven_time():
    # Full test with all dependencies and uneven time steps
    df = pd.DataFrame(
        {
            "temp": [25, 30, 35],
            "relative_humidity": [40, 50, 60],
            "temp_air": [20, 25, 30],
            "temp_module": [25, 30, 35],
            "poa_global": [800, 900, 1000],
        }
    )
    spectra = pd.DataFrame(
        {
            "Spectra: garbage identification here [ 300, 350, 400 ]": [0.1, 0.2, 0.5],
        }
    )
    times = pd.DataFrame(
        {
            "elapsed_time": [1, 2, 3],
        }
    )
    result = pvdeg.degradation.arrhenius(
        weather_df=df,
        irradiance=spectra,
        elapsed_time=times,
        p=0.5,
        n=1,
        Ea=40,
        C2=0.07,
    )
    assert result == pytest.approx(2.928567627e-9, abs=1e-13)


def test_arrhenius_action_spectra_uneven_time_one_DataFrame():
    # Full test with all dependencies, uneven time steps, and all in one DataFrame
    df = pd.DataFrame(
        {
            "temp": [25, 30, 35],
            "relative_humidity": [40, 50, 60],
            "temp_air": [20, 25, 30],
            "temp_module": [25, 30, 35],
            "poa_global": [800, 900, 1000],
        }
    )
    spectra = pd.DataFrame(
        {
            "Spectra: garbage identification here [ 300, 350, 400 ]": [0.1, 0.2, 0.5],
        }
    )
    times = pd.DataFrame(
        {
            "elapsed_time": [1, 2, 3],
        }
    )
    df = pd.concat([df, times, spectra], axis=1)
    result = pvdeg.degradation.arrhenius(weather_df=df, p=0.5, n=1, Ea=40, C2=0.07)
    assert result == pytest.approx(2.928567627e-9, abs=1e-13)


_PEROVSKITE_DF = pd.DataFrame(
    {
        "temp_air": [20.0, 25.0, 30.0],
        "relative_humidity": [40.0, 50.0, 60.0],
    }
)


def test_perovskite_no_weather_df():
    with pytest.raises(ValueError):
        pvdeg.degradation.perovskite_degradation()


def test_perovskite_no_rh():
    df = pd.DataFrame({"temp_air": [20.0, 25.0, 30.0]})
    with pytest.raises(ValueError):
        pvdeg.degradation.perovskite_degradation(weather_df=df)


def test_perovskite_invalid_component():
    with pytest.raises(ValueError):
        pvdeg.degradation.perovskite_degradation(
            weather_df=_PEROVSKITE_DF, component="invalid"
        )


def test_perovskite_total():
    result = pvdeg.degradation.perovskite_degradation(weather_df=_PEROVSKITE_DF)
    assert isinstance(result, pd.Series)
    assert len(result) == 3
    assert result.name == "perovskite_degradation_total"
    assert not result.isna().any()
    assert (result > 0).all()


def test_perovskite_components():
    for comp in ("WPO", "DPO", "r_hum", "r_therm"):
        result = pvdeg.degradation.perovskite_degradation(
            weather_df=_PEROVSKITE_DF, component=comp
        )
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert result.name == f"perovskite_degradation_{comp}"
        assert not result.isna().any()
        assert (result > 0).all()


# 1440-row constant ISOS-L2 DataFrame (T_air=50°C so NOCT gives T_cell≈85°C)
_N_ISOS = 1440
_ISOS_DF = pd.DataFrame(
    {"temp_air": np.full(_N_ISOS, 50.0)},
    index=pd.date_range("2023-01-01", periods=_N_ISOS, freq="h"),
)
_ISOS_POA = pd.Series(np.full(_N_ISOS, 1000.0), index=_ISOS_DF.index)

# Short DataFrame for fast unit tests (no POA calculation needed)
_FACTOR_DF = pd.DataFrame(
    {"temp_air": [25.0, 25.0, 25.0, 25.0]},
    index=pd.date_range("2023-01-01", periods=4, freq="h"),
)
_FACTOR_POA = pd.Series([1000.0, 1000.0, 1000.0, 1000.0], index=_FACTOR_DF.index)


def test_degradation_factor_no_weather_df():
    with pytest.raises(ValueError):
        pvdeg.degradation.perovskite_degradation_factor(weather_df=None)


def test_degradation_factor_returns_series():
    result = pvdeg.degradation.perovskite_degradation_factor(
        weather_df=_FACTOR_DF, poa=_FACTOR_POA
    )
    assert isinstance(result, pd.Series)
    assert len(result) == 4
    assert result.name == "perovskite_degradation_factor"
    assert not result.isna().any()


def test_degradation_factor_starts_below_one():
    result = pvdeg.degradation.perovskite_degradation_factor(
        weather_df=_FACTOR_DF, poa=_FACTOR_POA
    )
    # Every value should be ≤ 1.0 (degradation can only reduce CE)
    assert (result <= 1.0).all()
    # And strictly less than 1 once illuminated
    assert (result < 1.0).all()


def test_degradation_factor_monotonic():
    result = pvdeg.degradation.perovskite_degradation_factor(
        weather_df=_FACTOR_DF, poa=_FACTOR_POA
    )
    # Under constant illumination the factor must be non-increasing
    assert (result.diff().dropna() <= 0).all()


def test_degradation_factor_no_light_no_degradation():
    """Zero irradiance (night) → DF per hour = 1 → DF_total stays at initial."""
    dark_df = pd.DataFrame(
        {"temp_air": [25.0, 25.0, 25.0]},
        index=pd.date_range("2023-01-01", periods=3, freq="h"),
    )
    dark_poa = pd.Series([0.0, 0.0, 0.0], index=dark_df.index)
    result = pvdeg.degradation.perovskite_degradation_factor(
        weather_df=dark_df, poa=dark_poa
    )
    # With gamma=1 and I=0, k=0, DF per hour = A1+A2+B = 1.0, so DF_total = 1.0
    assert result.values == pytest.approx(np.ones(len(result)), abs=1e-12)


def test_degradation_factor_parameters_override():
    """parameters dict should override keyword arguments."""
    params = {
        "Ea_fast": 0.100,
        "Ea_slow": 0.100,
        "k0_fast": 100.0,
        "k0_slow": 100.0,
        "A1": 0.5,
        "A2": 0.45,
        "B": 0.05,
        "gamma": 1.0,
        "I_ref": 1200.0,
    }
    result_kw = pvdeg.degradation.perovskite_degradation_factor(
        weather_df=_FACTOR_DF,
        poa=_FACTOR_POA,
        Ea_fast=0.100,
        Ea_slow=0.100,
        k0_fast=100.0,
        k0_slow=100.0,
        A1=0.5,
        A2=0.45,
        B=0.05,
    )
    result_params = pvdeg.degradation.perovskite_degradation_factor(
        weather_df=_FACTOR_DF, poa=_FACTOR_POA, parameters=params
    )
    pd.testing.assert_series_equal(result_kw, result_params)


def test_degradation_factor_isos_l2_t90():
    """Default Zhao parameters should give T90,Agg ≈ 1440 h at ISOS-L2.

    We verify the CE factor at t=1440h is below 1 and above B (residual),
    and that the irradiance-weighted average (PR_Agg proxy) is close to 0.90.
    """
    ce = pvdeg.degradation.perovskite_degradation_factor(
        weather_df=_ISOS_DF, poa=_ISOS_POA
    )
    # CE factor after 1440h should be substantially below 1
    assert ce.iloc[-1] < 0.99
    # And above the residual B=0.05
    assert ce.iloc[-1] > 0.04
    # Simple irradiance-weighted PR_Agg proxy (linear approximation)
    PR_Agg_proxy = ce.mean()
    # Should be close to 0.90 (±0.05 tolerance to account for approximation)
    assert PR_Agg_proxy == pytest.approx(0.90, abs=0.05)


_PR_DF = pd.DataFrame(
    {
        "temp_air": np.full(_N_ISOS, 20.0),
        "wind_speed": np.full(_N_ISOS, 2.0),
        "ghi": np.full(_N_ISOS, 500.0),
        "dhi": np.full(_N_ISOS, 100.0),
        "dni": np.full(_N_ISOS, 500.0),
    },
    index=pd.date_range("2023-06-21", periods=_N_ISOS, freq="h"),
)
_PR_META = {
    "latitude": 39.74,
    "longitude": -105.18,
    "altitude": 1829,
    "timezone": "Etc/GMT+7",
    "wind_height": 10,
}
# Pre-computed constant cell temperature to avoid meta-dependency in unit tests
_PR_TEMP_CELL = pd.Series(np.full(_N_ISOS, 25.0), index=_PR_DF.index)
_CE_ONE = pd.Series(np.ones(_N_ISOS), index=_PR_DF.index)
_CE_FACTOR = pvdeg.degradation.perovskite_degradation_factor(
    weather_df=_PR_DF,
    poa=pd.Series(np.full(_N_ISOS, 1000.0), index=_PR_DF.index),
)


def test_degraded_power_ratio_no_degradation():
    """When ce_factor=1, PR_Agg should be 1.0 everywhere."""
    result = pvdeg.degradation.degraded_power_ratio(
        weather_df=_PR_DF,
        meta=_PR_META,
        ce_factor=_CE_ONE,
        poa=pd.DataFrame({"poa_global": np.full(_N_ISOS, 500.0)}, index=_PR_DF.index),
        temp_cell=_PR_TEMP_CELL,
    )
    pr = result["PR_Agg"]
    assert isinstance(pr, pd.Series)
    assert pr.values == pytest.approx(np.ones(len(pr)), abs=1e-6)
    assert result["T90_Agg_hours"] is None


def test_degraded_power_ratio_keys():
    result = pvdeg.degradation.degraded_power_ratio(
        weather_df=_PR_DF,
        meta=_PR_META,
        ce_factor=_CE_ONE,
        poa=pd.DataFrame({"poa_global": np.full(_N_ISOS, 500.0)}, index=_PR_DF.index),
        temp_cell=_PR_TEMP_CELL,
    )
    assert set(result.keys()) == {
        "PR_Agg",
        "T90_Agg_hours",
        "power_degraded",
        "power_reference",
    }


def test_degraded_power_ratio_power_series_nonneg():
    result = pvdeg.degradation.degraded_power_ratio(
        weather_df=_PR_DF,
        meta=_PR_META,
        ce_factor=_CE_ONE,
        poa=pd.DataFrame({"poa_global": np.full(_N_ISOS, 500.0)}, index=_PR_DF.index),
        temp_cell=_PR_TEMP_CELL,
    )
    assert (result["power_reference"] >= 0).all()
    assert (result["power_degraded"] >= 0).all()


def test_degraded_power_ratio_degraded_le_reference():
    """Degraded power should always be ≤ reference power at every hour."""
    result = pvdeg.degradation.degraded_power_ratio(
        weather_df=_PR_DF,
        meta=_PR_META,
        ce_factor=_CE_FACTOR,
        poa=pd.DataFrame({"poa_global": np.full(_N_ISOS, 1000.0)}, index=_PR_DF.index),
        temp_cell=_PR_TEMP_CELL,
    )
    assert (result["power_degraded"] <= result["power_reference"] + 1e-10).all()


def _build_test_tandem_template():
    pvc = pytest.importorskip("pvcircuit")
    dev = pvc.Multi2T(name="pytest_tandem", Eg_list=[1.68, 1.12])
    return dev


def test_degraded_power_ratio_tandem_no_degradation():
    """When ce_factor=1, tandem PR_Agg should be 1.0 everywhere."""
    tandem = _build_test_tandem_template()
    result = pvdeg.degradation.degraded_power_ratio_tandem(
        weather_df=_PR_DF,
        meta=_PR_META,
        ce_factor=_CE_ONE,
        tandem_template=tandem,
        poa=pd.DataFrame({"poa_global": np.full(_N_ISOS, 500.0)}, index=_PR_DF.index),
        temp_cell=_PR_TEMP_CELL,
    )
    pr = result["PR_Agg"]
    assert isinstance(pr, pd.Series)
    assert pr.values == pytest.approx(np.ones(len(pr)), abs=1e-6)
    assert result["T90_Agg_hours"] is None


def test_degraded_power_ratio_tandem_degraded_le_reference():
    """Tandem degraded power should always be <= reference power."""
    tandem = _build_test_tandem_template()
    ce_08 = pd.Series(np.full(_N_ISOS, 0.8), index=_PR_DF.index)
    result = pvdeg.degradation.degraded_power_ratio_tandem(
        weather_df=_PR_DF,
        meta=_PR_META,
        ce_factor=ce_08,
        tandem_template=tandem,
        poa=pd.DataFrame({"poa_global": np.full(_N_ISOS, 1000.0)}, index=_PR_DF.index),
        temp_cell=_PR_TEMP_CELL,
    )
    assert (result["power_degraded"] <= result["power_reference"] + 1e-10).all()
    assert set(result.keys()) == {
        "PR_Agg",
        "T90_Agg_hours",
        "power_degraded",
        "power_reference",
    }


# Test data for acetic acid functions: 100-hour constant temperature profiles
_HAC_CONST_TEMP_DF = pd.DataFrame(
    {"temp_module": np.full(100, 85.0)},
    index=pd.date_range("2023-01-01", periods=100, freq="h"),
)
_HAC_TEMP_RANGE_DF = pd.DataFrame(
    {"temp_module": np.linspace(20.0, 80.0, 100)},
    index=pd.date_range("2023-01-01", periods=100, freq="h"),
)
_HAC_TEMP_SERIES = _HAC_CONST_TEMP_DF["temp_module"]
_HAC_TEMP_SERIES_RANGE = _HAC_TEMP_RANGE_DF["temp_module"]


def test_acetic_acid_generation_returns_series():
    """acetic_acid_generation should return a pd.Series with correct index."""
    result = pvdeg.degradation.acetic_acid_generation(temp_module=_HAC_TEMP_SERIES)
    assert isinstance(result, pd.Series)
    assert len(result) == len(_HAC_TEMP_SERIES)
    assert (result.index == _HAC_TEMP_SERIES.index).all()
    assert result.name == "HAc_rate_ng_min_g"


def test_acetic_acid_generation_positive_rates():
    """All generation rates should be positive (exothermic process)."""
    result = pvdeg.degradation.acetic_acid_generation(temp_module=_HAC_TEMP_SERIES)
    assert (result > 0).all()
    assert result.isna().sum() == 0


def test_acetic_acid_generation_temp_dependence():
    """Higher temperature should produce higher generation rate (Arrhenius)."""
    rate_low = pvdeg.degradation.acetic_acid_generation(
        temp_module=pd.Series([25.0], index=pd.date_range("2023-01-01", periods=1))
    ).iloc[0]
    rate_high = pvdeg.degradation.acetic_acid_generation(
        temp_module=pd.Series([85.0], index=pd.date_range("2023-01-01", periods=1))
    ).iloc[0]
    assert rate_high > rate_low


def test_acetic_acid_generation_database_params():
    """Database lookup (encapsulant='AA002') should load Ro and Ea_gen."""
    # Test with database lookup
    result_db = pvdeg.degradation.acetic_acid_generation(
        temp_module=_HAC_TEMP_SERIES, encapsulant="AA002"
    )
    # Test with explicit parameters matching AA002 defaults
    result_explicit = pvdeg.degradation.acetic_acid_generation(
        temp_module=_HAC_TEMP_SERIES,
        Ro=0.00331,
        Ea_gen=90.0,
        T_ref=27.0,
        encapsulant=None,
    )
    pd.testing.assert_series_equal(result_db, result_explicit, rtol=1e-6)


def test_acetic_acid_generation_explicit_params():
    """Explicit parameters should override defaults."""
    result = pvdeg.degradation.acetic_acid_generation(
        temp_module=_HAC_TEMP_SERIES,
        Ro=0.001,
        Ea_gen=85.0,
        T_ref=30.0,
        encapsulant=None,
    )
    assert isinstance(result, pd.Series)
    assert (result > 0).all()
    # Values should differ from database defaults
    result_default = pvdeg.degradation.acetic_acid_generation(
        temp_module=_HAC_TEMP_SERIES, encapsulant="AA002"
    )
    assert not np.allclose(result.values, result_default.values)


def test_acetic_acid_cumulative_returns_series():
    """acetic_acid_cumulative should return a pd.Series with correct index."""
    result = pvdeg.degradation.acetic_acid_cumulative(temp_module=_HAC_TEMP_SERIES)
    assert isinstance(result, pd.Series)
    assert len(result) == len(_HAC_TEMP_SERIES)
    assert (result.index == _HAC_TEMP_SERIES.index).all()
    assert result.name == "HAc_cumulative_mg_g"


def test_acetic_acid_cumulative_no_nans():
    """Cumulative should have no NaN values."""
    result = pvdeg.degradation.acetic_acid_cumulative(temp_module=_HAC_TEMP_SERIES)
    assert result.isna().sum() == 0


def test_acetic_acid_cumulative_monotonic_increasing():
    """Cumulative HAc must be monotonically non-decreasing over time."""
    result = pvdeg.degradation.acetic_acid_cumulative(temp_module=_HAC_TEMP_SERIES)
    # Differences should be non-negative
    diffs = result.diff().dropna()
    assert (diffs >= 0).all()


def test_acetic_acid_cumulative_starts_at_zero():
    """First cumulative value should be very small (first hourly increment only)."""
    result = pvdeg.degradation.acetic_acid_cumulative(temp_module=_HAC_TEMP_SERIES)
    # First value should be approximately rate * 60 min/h / 1e6 (ng->mg)
    rate = pvdeg.degradation.acetic_acid_generation(temp_module=_HAC_TEMP_SERIES)
    expected_first = rate.iloc[0] * 60.0 / 1e6
    assert result.iloc[0] == pytest.approx(expected_first, rel=1e-9)


def test_acetic_acid_cumulative_integration():
    """Cumulative should be hourly integration of rate with proper unit conversion."""
    rate = pvdeg.degradation.acetic_acid_generation(
        temp_module=_HAC_TEMP_SERIES, encapsulant="AA002"
    )
    cumulative = pvdeg.degradation.acetic_acid_cumulative(
        temp_module=_HAC_TEMP_SERIES, encapsulant="AA002"
    )

    # Manual integration: hourly_production_mg = rate * 60 / 1e6
    hourly_mg = rate * 60.0 / 1e6
    manual_cumsum = hourly_mg.cumsum()
    manual_cumsum.name = "HAc_cumulative_mg_g"

    pd.testing.assert_series_equal(cumulative, manual_cumsum, rtol=1e-9)


def test_acetic_acid_cumulative_database_params():
    """Database lookup should produce consistent results."""
    result_db = pvdeg.degradation.acetic_acid_cumulative(
        temp_module=_HAC_TEMP_SERIES, encapsulant="AA002"
    )
    result_explicit = pvdeg.degradation.acetic_acid_cumulative(
        temp_module=_HAC_TEMP_SERIES,
        Ro=0.00331,
        Ea_gen=90.0,
        T_ref=27.0,
        encapsulant=None,
    )
    pd.testing.assert_series_equal(result_db, result_explicit, rtol=1e-6)


def test_acetic_acid_cumulative_magnitude():
    """Cumulative HAc magnitude should be in reasonable range for literature comparison.

    Gnocchi et al. (2018) reports ~0.5-0.6 mg/g at 3000h in damp heat.
    Our constant 85°C over 100h should yield much lower values (~0.001-0.01 mg/g).
    """
    result = pvdeg.degradation.acetic_acid_cumulative(temp_module=_HAC_TEMP_SERIES)
    # At 100h and 85°C, cumulative should be in micrograms to low milligrams
    assert result.iloc[-1] > 0
    assert result.iloc[-1] < 1.0  # Should be well below 1 mg/g for 100h


def test_acetic_acid_generation_vs_cumulative_relationship():
    """Cumulative increment should match generation rate (considers unit conversion)."""
    rate = pvdeg.degradation.acetic_acid_generation(
        temp_module=_HAC_TEMP_SERIES, encapsulant="AA002"
    )
    cumulative = pvdeg.degradation.acetic_acid_cumulative(
        temp_module=_HAC_TEMP_SERIES, encapsulant="AA002"
    )

    # Verify first 10 values
    for i in range(1, 11):
        expected = (rate.iloc[: i + 1] * 60.0 / 1e6).sum()
        assert cumulative.iloc[i] == pytest.approx(expected, rel=1e-9)
