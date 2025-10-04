"""Using pytest to create unit tests for pvdeg.

to run unit tests, run pytest from the command line in the pvdeg directory to run
coverage tests, run py.test --cov-report term-missing --cov=pvdeg
"""

import os
import pytest
from pvdeg import design, weather, humidity, TEST_DATA_DIR

PSM_FILE = os.path.join(TEST_DATA_DIR, "psm3_pytest.csv")
PSM, META = weather.read(PSM_FILE, "psm")


def test_edge_seal_ingress_rate():
    # test calculation for constant k

    water_saturation_pressure, avg_water_saturation_pressure = (
        humidity.water_saturation_pressure(PSM.get("dew_point"))
    )
    k = design.edge_seal_ingress_rate(avg_water_saturation_pressure)
    assert k == pytest.approx(0.00096, abs=0.000005)


def test_edge_seal_width():
    # test for edge_seal_width

    edge_seal_width = design.edge_seal_width(weather_df=PSM, meta=META)
    edge_seal_from_dewpt = design.edge_seal_width(
        weather_df=PSM, meta=META, from_dew_point=True
    )
    assert edge_seal_width == pytest.approx(0.7171, abs=0.0005)
    assert edge_seal_from_dewpt == pytest.approx(0.4499, abs=0.0005)
