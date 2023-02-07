import PVDegradationTools as PVD
import numpy as np
import pytest
import os
import pandas as pd

nsrdb_file = '/datasets/NSRDB/current/nsrdb_tmy-2021.h5'

gid = 479494
gids = np.array([481541, 482563, 483583, 483584, 483585, 484606, 484607, 484608,
                 485630, 485631, 485632, 485633, 486654, 486655, 486657, 487679,
                 487682, 488705, 489730, 489731, 490757, 491780, 491781, 492807,
                 492808, 493834, 493835])

def test_pipeline_single():
    x = PVD.standards.test_pipeline(nsrdb_file, gid)
    assert x == pytest.approx(2.2574347934306602)

def test_pipeline_multiple():
    x = []
    for gid in gids:
        x.append(PVD.standards.test_pipeline(nsrdb_file, gid))
    df = pd.DataFrame(x, index = gids, columns=['x'])
    df.index.name = 'gid'
    
    data = pd.read_csv(os.path.join(PVD.TEST_DATA_DIR, 'gid_x.csv'), index_col='gid')
    pd.testing.assert_frame_equal(df, data)

if __name__ == "__main__":
    test_pipeline_single()
    test_pipeline_multiple()