# -*- coding: utf-8 -*-
"""
Created on March 13 2020

Using pytest to create unit tests for PV ICE
to run unit tests, run pytest from the command line in the bifacial_radiance directory
to run coverage tests, run py.test --cov-report term-missing --cov=bifacial_radiance

cd C:\Users\sayala\Documents\GitHub\PVDegradationTools\tests

"""

import PVDegradationTools
import numpy as np
import pytest
import os
import pandas as pd


# try navigating to tests directory so tests run from here.
try:
    os.chdir('tests')
except:
    pass


TESTDIR = os.path.dirname(__file__)  # this folder

INPUTWEATHERSPECTRA = 'test_weatherandspectra.csv'

def test_degradation():
    data=pd.read_csv(INPUTWEATHERSPECTRA)
    wavelengths = np.array(range(280,420,20))
    degradation = PVDegradationTools.Degradation.Degradation(data, wavelengths)
    assert (degradation == 3.252597282885626e-39)
    