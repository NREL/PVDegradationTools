import PVDegradationTools as PVD
import pandas as pd

if __name__ == "__main__":

    T0 = pd.DataFrame([45,46,47])
    T1 = pd.DataFrame([55,56,57])

    x = PVD.standards.ideal_installation_distance(T1, T0)

    print(x)