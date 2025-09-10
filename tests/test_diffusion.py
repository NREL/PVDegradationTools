import os
import pandas as pd
import pvdeg
from pvdeg import TEST_DATA_DIR
import json
from copy import copy


WEATHER = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "weather_day_pytest.csv"), index_col=0, parse_dates=True
)
with open(os.path.join(TEST_DATA_DIR, "meta.json"), "r") as file:
    META = json.load(file)

RESULT_1D = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "1d-oxygen-profile.csv"), index_col=0, dtype="float64"
)


def test_diffusion_1d():
    temperature = pvdeg.temperature.temperature(
        weather_df=WEATHER,
        meta=META,
        cell_or_mod="module",
        temp_model="sapm",
        conf="open_rack_glass_polymer",
    )

    temperature = pd.DataFrame(temperature, columns=["module_temperature"])
    temperature["time"] = list(range(len(temperature)))

    pressure = 0.2109 * (1 - 0.0065 * META["altitude"] / 288.15) ** 5.25588

    oxygen_profile = pvdeg.diffusion.esdiffusion(
        temperature=temperature,
        edge_seal="OX005",
        encapsulant="OX003",
        edge_seal_width=1.5,
        encapsulant_width=10,
        seal_nodes=20,
        encapsulant_nodes=50,
        press=pressure,
        repeat=2,
    )

    # CSV has an extra row because it was saved weird
    col_list = copy(RESULT_1D.columns).values
    col_list[21] = "1.5"
    RESULT_1D.columns = col_list.astype(float)

    pd.testing.assert_frame_equal(
        oxygen_profile,
        RESULT_1D,
        check_dtype=False,
        check_column_type=False,
        atol=1e-3,
        rtol=1e-3,
    )

    # pd.testing.assert_frame_equal(
    #    oxygen_profile,
    #    RESULT_1D,
    #    check_dtype=False,
    #    check_column_type=False,
    # )
