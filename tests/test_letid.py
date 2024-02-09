import pytest
import os
import pandas as pd
import numpy as np

# do we need the following
from scipy.constants import convert_temperature, elementary_charge, Boltzmann 
from scipy.integrate import simpson
import datetime
import pvlib
# do we need the above

from pvdeg import collection, TEST_DIR, DATA_DIR

def test_tau_now():
    pass

def test_k_ij():
    pass

def test_carrier_factor():
    pass

def test_carrier_factor_wafer():
    pass

def test_calc_dn():
    pass

def test_convert_i_to_v():
    pass

def test_j0_gray():
    pass

def test_calc_voc_from_tau():
    pass

def test_calc_device_params():
    pass

def test_calc_energy_loss():
    pass

def test_calc_regeneration_time():
    pass

def test_calc_pmp_loss_from_tau_loss():
    pass

def test_calc_ndd():
    pass

def test_ff_green():
    pass

def test_calc_injection_outdoors():
    pass

def test_calc_letid_outdoors():
    pass

def test_calc_letid_lab():
    pass