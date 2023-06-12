import os
import pytest
from pvdeg import design, weather, humidity, TEST_DATA_DIR

PSM_FILE = os.path.join(TEST_DATA_DIR,'psm3_pytest.csv')
PSM, META = weather.read(PSM_FILE, 'psm')

def test_k():
    # test calculation for constant k

    psat, avg_psat = humidity.psat(PSM['Dew Point'])
    k = design.k(avg_psat=avg_psat)
    assert k == pytest.approx(.00096, abs=.000005)

def test_edge_seal_width():
    # test for edge_seal_width

    psat, avg_psat = humidity.psat(PSM['Dew Point'])
    k =design.k(avg_psat=avg_psat)
    edge_seal_width = design.edge_seal_width(k=k)
    assert edge_seal_width == pytest.approx(0.449, abs=0.002)
