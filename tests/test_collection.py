import pytest
import os
import pandas as pd
import numpy as np
from PVDegradationTools import collection

TESTDIR = os.path.dirname(__file__)  # this folder
GENERATIONFILE = os.path.join(TESTDIR, 'data', r'PVL_GenProfile.xlsx')
generation_df = pd.read_excel(GENERATIONFILE, header=0, engine="openpyxl")
generation = generation_df['Generation (cm-3s-1)']
depth = generation_df['Depth (um)']


def test_collection_probability():
    s = 1000
    l = 100*1e-4
    d = 27

    thickness = 180*1e-4
    x = thickness

    cp = collection.collection_probability(x, thickness, s, l, d)

    assert cp == pytest.approx(0.238255, abs=.000005)


def test_calculate_jsc_from_tau_cp():
    tau = 100
    wafer_thickness = 180
    d_base = 27
    s_rear = 1000

    jsc = collection.calculate_jsc_from_tau_cp(tau, wafer_thickness, d_base, s_rear, generation,
                                               depth)

    assert jsc == pytest.approx(36.886672761936545, abs=.000005)


def test_calculate_jsc_from_tau_iqe():
    tau = 100
    wafer_thickness = 180
    d_base = 27
    s_rear = 1000

    # dummy data
    spectrum = np.array([0, 1, 0])  # photon flux [cm^–2s^–1nm^-1]
    absorption = np.array([1e6, 1e4, 1e2])  # absorption coefficient
    wavelengths = np.array([400, 600, 1000])  # photon wavelength [nm]

    jsc = collection.calculate_jsc_from_tau_iqe(tau, wafer_thickness, d_base, s_rear, spectrum,
                                                absorption, wavelengths)

    assert jsc == pytest.approx(38.5981695901184, abs=.000005)


def test_generation_current():
    jgen = collection.generation_current(generation, depth)
    assert jgen == pytest.approx(42.36324575251117, abs=.000005)
