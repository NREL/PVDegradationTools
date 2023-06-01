import os
import pandas as pd
import numpy as np
import pytest
import pvdeg
from pvdeg import TEST_DATA_DIR

PSM_FILE = os.path.join(TEST_DATA_DIR,r'psm3_pytest.csv')
PSM = pd.read_csv(PSM_FILE, header=2)

INPUTWEATHERSPECTRA = os.path.join('data',r'test_weatherandspectra.csv')

def test_vantHoff_deg():
    # test the vantHoff degradation acceleration factor

    vantHoff_deg = pvdeg.Degradation.vantHoff_deg(I_chamber=1000, poa_global=PSM['poa_global'],
                                                temp_cell=PSM['temp_cell'], temp_chamber=60)
    assert vantHoff_deg == pytest.approx(8.38, abs=.02)

def test_iwa_vantHoff():
    # test the vantHoff equivalent weighted average irradiance

    irr_weighted_avg = pvdeg.Degradation.IwaVantHoff(poa_global=PSM['poa_global'],
                                                    temp_cell=PSM['temp_cell'])
    assert irr_weighted_avg == pytest.approx(232.47, abs=0.5)

def test_arrhenius_deg():
    # test the arrhenius degradation acceleration factor

    rh_chamber = 15
    temp_chamber = 60
    I_chamber = 1e3
    Ea = 40
    rh_surface = pvdeg.StressFactors.rh_surface_outside(rh_ambient=PSM['Relative Humidity'],
                                                        temp_ambient=PSM['Temperature'],
                                                        temp_module=PSM['temp_module'])
    arrhenius_deg = pvdeg.Degradation.arrhenius_deg(I_chamber=I_chamber, rh_chamber=rh_chamber,
                                              rh_outdoor=rh_surface, poa_global=PSM['poa_global'],
                                              temp_chamber=temp_chamber, temp_cell=PSM['temp_cell'],
                                              Ea=Ea)
    assert arrhenius_deg == pytest.approx(12.804, abs=0.1)

def test_iwa_arrhenius():
    # test arrhenius equivalent weighted average irradiance
    # requires PSM3 weather file

    Ea = 40
    irr_weighted_avg = pvdeg.Degradation.IwaArrhenius(poa_global=PSM['poa_global'],
                                                  rh_outdoor=PSM['Relative Humidity'],
                                                  temp_cell=PSM['temp_cell'], Ea=Ea)
    assert irr_weighted_avg == pytest.approx(194.66, abs=0.1)

def test_degradation():
    # test RH, Temp, Spectral Irradiance sensitive degradation
    # requires TMY3-like weather data
    # requires spectral irradiance data

    data=pd.read_csv(INPUTWEATHERSPECTRA)
    wavelengths = np.array(range(280,420,20))
    degradation = pvdeg.Degradation.degradation(spectra=data['Spectra'], rh_module=data['RH'],
                                                temp_module=data['Temperature'],
                                                wavelengths=wavelengths)
    assert degradation == pytest.approx(4.4969e-38, abs=0.02e-38)
