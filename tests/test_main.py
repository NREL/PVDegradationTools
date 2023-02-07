"""
Created on January 2nd 2023

Using pytest to create unit tests for PV Degradation Tools
to run unit tests, run pytest from the command line in the PVDegradationTools directory
to run coverage tests, run py.test --cov-report term-missing --cov=bifacial_radiance
"""

import os
import numpy as np
import pytest
import pandas as pd
import pvlib
import PVDegradationTools as PVD

# try navigating to tests directory so tests run from here.
try:
    os.chdir('tests')
except:
    pass


TESTDIR = os.path.dirname(__file__)  # this folder

INPUTWEATHERSPECTRA = r'test_weatherandspectra.csv'

WEATHERFILE = r'722740TYA.CSV'

PSM3FILE = r'psm3_pytest_2.csv'

PSM = pd.read_csv(PSM3FILE, header=2)

# --------------------------------------------------------------------------------------------------
# -- StressFactors

# def test_water_vapor_pressure():
#     # test water vapor pressure
#     # *DEPRECATED*

#     wvp = PVD.StressFactors.water_vapor_pressure(PSM['Dew Point'])
#     assert wvp.__len__() == PSM.__len__()
#     avg_wvp = wvp.mean()
#     assert avg_wvp == pytest.approx(0.54218, abs=0.0001)

def test_k():
    # test calculation for constant k

    wvp = PVD.StressFactors.psat(PSM['Dew Point'])
    avg_wvp = wvp.mean()
    k = PVD.StressFactors.k(avg_wvp=avg_wvp)
    assert k == pytest.approx(.00096, abs=.000005)

def test_edge_seal_width():
    # test for edge_seal_width

    water_vapor_pressure = PVD.StressFactors.psat(PSM['Dew Point'])
    avg_wvp = water_vapor_pressure.mean()
    k = PVD.StressFactors.k(avg_wvp=avg_wvp)
    edge_seal_width = PVD.StressFactors.edge_seal_width(k=k)
    assert edge_seal_width == pytest.approx(0.449, abs=0.002)

def test_psat():
    # test saturation point

    psat = PVD.StressFactors.psat(PSM['Dew Point'])
    assert psat.__len__() == PSM.__len__()
    avg_psat = psat.mean()
    assert avg_psat == pytest.approx(0.54218, abs=0.0001)

def test_rh_surface_outside():
    # test calculation for the RH just outside a module surface
    # requires PSM3 weather file

    rh_surface = PVD.StressFactors.rh_surface_outside(rh_ambient=PSM['Relative Humidity'],
                                                        temp_ambient=PSM['Temperature'],
                                                        temp_module=PSM['temp_module'])
    assert rh_surface.__len__() == PSM.__len__()
    assert rh_surface[17] == pytest.approx(81.99, abs=0.1)

def test_rh_front_encap():
    # test calculation for RH of module fronside encapsulant
    # requires PSM3 weather file

    rh_front_encap = PVD.StressFactors.rh_front_encap(rh_ambient=PSM['Relative Humidity'],
                                                    temp_ambient=PSM['Temperature'],
                                                    temp_module=PSM['temp_module'])
    assert rh_front_encap.__len__() == PSM.__len__()
    assert rh_front_encap.iloc[17] == pytest.approx(50.289, abs=.001)

def test_rh_back_encap():
    # test calculation for RH of module backside encapsulant
    # requires PSM3 weather file

    rh_back_encap = PVD.StressFactors.rh_back_encap(rh_ambient=PSM['Relative Humidity'],
                                                    temp_ambient=PSM['Temperature'],
                                                    temp_module=PSM['temp_module'])
    assert rh_back_encap.__len__() == PSM.__len__()
    assert rh_back_encap[17] == pytest.approx(80.4576, abs=0.001)

def test_rh_backsheet_from_encap():
    # test the calculation for backsheet relative humidity
    # requires PSM3 weather file

    rh_back_encap = PVD.StressFactors.rh_back_encap(rh_ambient=PSM['Relative Humidity'],
                                                    temp_ambient=PSM['Temperature'],
                                                    temp_module=PSM['temp_module'])
    rh_surface = PVD.StressFactors.rh_surface_outside(rh_ambient=PSM['Relative Humidity'],
                                                        temp_ambient=PSM['Temperature'],
                                                        temp_module=PSM['temp_module'])
    rh_backsheet = PVD.StressFactors.rh_backsheet_from_encap(rh_back_encap=rh_back_encap,
                                                            rh_surface_outside=rh_surface)
    assert rh_backsheet.__len__() == PSM.__len__()
    assert rh_backsheet[17] == pytest.approx(81.2238, abs=0.001)

def test_rh_backsheet():
    # test the calculation for backsheet relative humidity directly from weather variables
    # requires PSM3 weather file

    rh_backsheet = PVD.StressFactors.rh_backsheet(rh_ambient=PSM['Relative Humidity'],
                                                    temp_ambient=PSM['Temperature'],
                                                    temp_module=PSM['temp_module'])
    assert rh_backsheet.__len__() == PSM.__len__()
    assert rh_backsheet[17] == pytest.approx(81.2238, abs=0.001)

def test_rh_module():
    # test the rh_module function
    # requires PSM3 weather file

    rh_module = PVD.StressFactors.rh_module(rh_ambient=PSM['Relative Humidity'],
                                            temp_ambient=PSM['Temperature'],
                                            temp_module=PSM['temp_module'])
    assert rh_module.__len__() == PSM.__len__()
    assert rh_module['surface_outside'][17] == pytest.approx(81.99, abs=0.1)
    assert rh_module['front_encap'][17] == pytest.approx(50.2891, abs=0.001)
    assert rh_module['back_encap'][17] == pytest.approx(80.4576, abs=0.001)
    assert rh_module['backsheet'][17] == pytest.approx(81.2238, abs=0.001)

# --------------------------------------------------------------------------------------------------
# -- Degradation

def test_vantHoff_deg():
    # test the vantHoff degradation acceleration factor

    vantHoff_deg = PVD.Degradation.vantHoff_deg(I_chamber=1000, poa_global=PSM['poa_global'],
                                                temp_cell=PSM['temp_cell'], temp_chamber=60)
    print(vantHoff_deg)
    assert vantHoff_deg == pytest.approx(8.38, abs=.02)

def test_iwa_vantHoff():
    # test the vantHoff equivalent weighted average irradiance

    irr_weighted_avg = PVD.Degradation.IwaVantHoff(poa_global=PSM['poa_global'],
                                                    temp_cell=PSM['temp_cell'])
    assert irr_weighted_avg == pytest.approx(232.47, abs=0.5)

def test_arrhenius_deg():
    # test the arrhenius degradation acceleration factor

    rh_chamber = 15
    temp_chamber = 60
    I_chamber = 1e3
    Ea = 40
    rh_surface = PVD.StressFactors.rh_surface_outside(rh_ambient=PSM['Relative Humidity'],
                                                        temp_ambient=PSM['Temperature'],
                                                        temp_module=PSM['temp_module'])
    arrhenius_deg = PVD.Degradation.arrhenius_deg(I_chamber=I_chamber, rh_chamber=rh_chamber,
                                              rh_outdoor=rh_surface, poa_global=PSM['poa_global'],
                                              temp_chamber=temp_chamber, temp_cell=PSM['temp_cell'],
                                              Ea=Ea)
    assert arrhenius_deg == pytest.approx(12.804, abs=0.1)

def test_iwa_arrhenius():
    # test arrhenius equivalent weighted average irradiance
    # requires PSM3 weather file

    Ea = 40
    irr_weighted_avg = PVD.Degradation.IwaArrhenius(poa_global=PSM['poa_global'],
                                                  rh_outdoor=PSM['Relative Humidity'],
                                                  temp_cell=PSM['temp_cell'], Ea=Ea)
    assert irr_weighted_avg == pytest.approx(194.66, abs=0.1)

def test_degradation():
    # test RH, Temp, Spectral Irradiance sensitive degradation
    # requires TMY3-like weather data
    # requires spectral irradiance data

    data=pd.read_csv(INPUTWEATHERSPECTRA)
    wavelengths = np.array(range(280,420,20))
    degradation = PVD.Degradation.degradation(spectra=data['Spectra'], rh_module=data['RH'],
                                                temp_module=data['Temperature'],
                                                wavelengths=wavelengths)
    assert degradation == pytest.approx(4.4969e-38, abs=0.02e-38)

def test_solder_fatigue():
    # test solder fatique with default parameters
    # requires PSM3 weather file

    damage = PVD.Degradation.solder_fatigue(time_range=PSM['time_range'],
                                            temp_cell=PSM['temp_cell'])
    assert damage == pytest.approx(14.25, abs=0.1)

# --------------------------------------------------------------------------------------------------
# -- Standards

def test_ideal_installation_distance():
    # test ideal installation calculation
    # requires TMY3-like weather file

    df_tmy, metadata = pvlib.iotools.read_tmy3(filename=WEATHERFILE,
                                               coerce_year=2021, recolumn=True)
    df_tmy['air_temperature'] = df_tmy['DryBulb']
    df_tmy['wind_speed'] = df_tmy['Wspd']
    df_tmy['dni']=df_tmy['DNI']
    df_tmy['ghi']=df_tmy['GHI']
    df_tmy['dhi']=df_tmy['DHI']
    x = PVD.Standards.ideal_installation_distance(df_tmy, metadata)
    assert x == pytest.approx(5.11657, abs=0.0001)

def test_calculate_T98Temperature():
    # test T98 of temperature calculation
    # requires TMY3-like weather file

    df_tmy, metadata = pvlib.iotools.read_tmy3(filename=WEATHERFILE,
                                               coerce_year=2021, recolumn=True)
    df_tmy['air_temperature'] = df_tmy['DryBulb']
    df_tmy['wind_speed'] = df_tmy['Wspd']
    df_tmy['dni']=df_tmy['DNI']
    df_tmy['ghi']=df_tmy['GHI']
    df_tmy['dhi']=df_tmy['DHI']
    T98 = PVD.Standards.calculate_T98Temperature(df_tmy, metadata)
    assert T98 == pytest.approx(59.32, abs=0.01)