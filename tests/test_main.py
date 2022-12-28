# -*- coding: utf-8 -*-
"""
Created on March 13 2020

Using pytest to create unit tests for PV ICE
to run unit tests, run pytest from the command line in the bifacial_radiance directory
to run coverage tests, run py.test --cov-report term-missing --cov=bifacial_radiance

cd C:\Users\sayala\Documents\GitHub\\PVDegTool\PVDegradationTools\tests

"""

import PVDegradationTools
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

def test_degradation():
    data=pd.read_csv(INPUTWEATHERSPECTRA)
    wavelengths = np.array(range(280,420,20))
    degradation = PVDegradationTools.Degradation.Degradation(data, wavelengths)
    assert (degradation == 3.252597282885626e-39)
    
def test_ideal_installation_distance():
    df_tmy, metadata = pvlib.iotools.read_tmy3(filename=WEATHERFILE, 
                                               coerce_year=2021, recolumn=True)
    df_tmy['air_temperature'] = df_tmy['DryBulb']
    df_tmy['wind_speed'] = df_tmy['Wspd']
    df_tmy['dni']=df_tmy['DNI']
    df_tmy['ghi']=df_tmy['GHI']
    df_tmy['dhi']=df_tmy['DHI']
    x = PVDegradationTools.Standards.ideal_installation_distance(df_tmy, 
                                                                 metadata)
    assert (x == 5.116572312951921)
    