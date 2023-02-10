import PVDegradationTools as PVD
import pytest

nsrdb_fp = '/datasets/NSRDB/current/nsrdb_tmy-2021.h5'
gid = 479494 #NREL location

def test_calc_standoff():
    res = PVD.standards.calc_standoff(
        nsrdb_fp,
        gid,
        tilt=None,
        azimuth=180,
        sky_model='isotropic',
        temp_model='sapm',
        module_type='glass_polymer',
        level=0,
        x_0=6.1)

    assert res == pytest.approx({'x': 2.459131550393533, 
                                 'T98_0': 79.39508117890611, 
                                 'T98_inf': 51.07779401289873})


if __name__ == "__main__":
    test_calc_standoff()