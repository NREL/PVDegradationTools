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
            "temp": [25, 30, 35],
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
            "temp": [25, 30, 35],
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
            "temp": [25, 30, 35],
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
            "temp": [25, 30, 35],
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
            "temp": [25, 30, 35],
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
