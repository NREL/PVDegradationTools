# -*- coding: utf-8 -*-
"""
Created on March 13 2020

Using pytest to create unit tests for PV ICE
to run unit tests, run pytest from the command line in the bifacial_radiance directory
to run coverage tests, run py.test --cov-report term-missing --cov=bifacial_radiance

cd C:\Users\sayala\Documents\GitHub\\PVDegTool\PVDegradationTools\tests

"""

import PVDegradationTools as PVD
import numpy as np
import pytest
import os
import pandas as pd
import pvlib

# try navigating to tests directory so tests run from here.
try:
    os.chdir('tests')
except:
    pass


TESTDIR = os.path.dirname(__file__)  # this folder

INPUTWEATHERSPECTRA = 'test_weatherandspectra.csv'

WEATHERFILE = '722740TYA.CSV'

PSM3FILE = 'psm3_pytest.csv'

PSM, = pvlib.iotools.read_psm3(PSM3FILE)


def test_water_vapor_pressure():
    wvp = PVD.EnergyCalcs.water_vapor_pressure(PSM['Dew Point'])
    assert wvp.__len__() == PSM.__len__()
    avg_wvp = wvp.mean()
    assert wvp == pytest.approx(0.542, abs=0.001)

def test_k():
    wvp = PVD.EnergyCalcs.water_vapor_pressure(PSM['Dew Point'])
    avg_wvp = wvp.mean()
    k = PVD.EnergyCalcs.k(avg_wvp=avg_wvp)
    assert k == pytest.approx(.00096, abs=.000005)

def test_edge_seal_width():
    water_vapor_pressure = PVD.EnergyCalcs.water_vapor_pressure(PSM['Dew Point'])
    avg_wvp = water_vapor_pressure.mean()
    k = PVD.EnergyCalcs.k(avg_wvp=avg_wvp)
    edge_seal_width = PVD.EnergyCalcs.edge_seal_width(k=k)
    assert edge_seal_width == pytest.approx(0.449, abs=0.002)

def test_vantHoff_deg():
    vantHoff_deg = PVD.EnergyCalcs.vantHoff_deg(I_chamber=1e3, poa_global=PSM['poa_global'],
                                                temp_cell=PSM['temp_cell'], temp_chamber=60)
    assert vantHoff_deg == pytest.approx(11, abs=.05)

def test_iwa_vantHoff():
    irr_weighted_avg = PVD.EnergyCalcs.IwaVantHoff(poa_global=PSM['poa_global'],
                                                    temp_cell=PSM['temp_cell'])
    assert irr_weighted_avg == pytest.approx(226.7, abs=0.5)

def test_arrhenius_deg():
    pass

def test_iwa_arrhenius():
    pass


def test_degradation():
    data=pd.read_csv(INPUTWEATHERSPECTRA)
    wavelengths = np.array(range(280,420,20))
    degradation = PVD.Degradation.degradation(data, wavelengths)
    assert (degradation == 3.252597282885626e-39)
    
def test_ideal_installation_distance():
    df_tmy, metadata = pvlib.iotools.read_tmy3(filename=WEATHERFILE, 
                                               coerce_year=2021, recolumn=True)
    df_tmy['air_temperature'] = df_tmy['DryBulb']
    df_tmy['wind_speed'] = df_tmy['Wspd']
    df_tmy['dni']=df_tmy['DNI']
    df_tmy['ghi']=df_tmy['GHI']
    df_tmy['dhi']=df_tmy['DHI']
    x = PVD.Standards.ideal_installation_distance(df_tmy, 
                                                                 metadata)
    assert (x == 5.116572312951921)
    