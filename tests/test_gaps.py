import PVDegradationTools as PVD
import gaps
import numpy as np

nsrdb_file = '/datasets/NSRDB/current/nsrdb_tmy-2021.h5'

gid = 479494
gids = np.array([481541, 482563, 483583, 483584, 483585, 484606, 484607, 484608,
                 485630, 485631, 485632, 485633, 486654, 486655, 486657, 487679,
                 487682, 488705, 489730, 489731, 490757, 491780, 491781, 492807,
                 492808, 493834, 493835])

T = PVD.standards.module_temperature(nsrdb_file, gid, 
                    temp_model='sapm', conf='open_rack_glass_polymer', 
                    tilt=None, azimuth=180, sky_model='isotropic')
print(T)

x = PVD.standards.test_pipeline(nsrdb_file, gid)
print(x)