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


def test_arrhenius_deg():
    # test the arrhenius degradation acceleration factor

    rh_chamber = 15
    temp_chamber = 60
    I_chamber = 1e3
    Ea = 40

    poa = pvdeg.spectral.poa_irradiance(weather_df, meta)
    temp_module = pvdeg.temperature.module(weather_df, meta, poa=poa)

    rh_surface = pvdeg.humidity.surface_outside(
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


def test_degradation():
    # test RH, Temp, Spectral Irradiance sensitive degradation
    # requires TMY3-like weather data
    # requires spectral irradiance data

    data = pd.read_csv(INPUT_SPECTRA)
    wavelengths = np.array(range(280, 420, 20))
    degradation = pvdeg.degradation.degradation(
        spectra=data["Spectra"],
        rh_module=data["RH"],
        temp_module=data["Temperature"],
        wavelengths=wavelengths,
    )
    assert degradation == pytest.approx(4.4969e-38, abs=0.02e-38)
