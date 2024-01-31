import os
import pytest
from pvdeg import design, weather, humidity, TEST_DATA_DIR

PSM_FILE = os.path.join(TEST_DATA_DIR, "psm3_pytest.csv")
PSM, META = weather.read(PSM_FILE, "psm")


def test_edge_seal_ingress_rate():
    # test calculation for constant k

    # "Dew Point" fallback handles key-name bug in pvlib < v0.10.3.
    psat, avg_psat = humidity.psat(PSM.get("temp_dew", PSM.get("Dew Point")))
    k = design.edge_seal_ingress_rate(avg_psat=avg_psat)
    assert k == pytest.approx(0.00096, abs=0.000005)


def test_edge_seal_width():
    # test for edge_seal_width

    edge_seal_width = design.edge_seal_width(weather_df=PSM, meta=META)
    edge_seal_from_dewpt = design.edge_seal_width(
        weather_df=PSM, meta=META, from_dew_point=True
    )
    assert edge_seal_width == pytest.approx(0.7171, abs=0.0005)
    assert edge_seal_from_dewpt == pytest.approx(0.4499, abs=0.0005)
