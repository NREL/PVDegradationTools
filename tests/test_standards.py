import PVDegradationTools as PVD
import pandas as pd

if __name__ == "__main__":

    nsrdb_file = '/datasets/NSRDB/current/nsrdb_tmy-2021.h5'
    gid = 467358

    # T0 = pd.DataFrame([45,46,47])
    # T1 = pd.DataFrame([55,56,57])

    # x = PVD.standards.ideal_installation_distance(T1, T0)

    x = PVD.standards.test_pipeline(nsrdb_file, gid)

    print(x)