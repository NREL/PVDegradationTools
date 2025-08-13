# TODO:
# correlation list is empty AND correlation list is populated with r = 0's

import os
import json
import pandas as pd
import pvdeg
from pvdeg import TEST_DATA_DIR

WEATHER = pd.read_csv(
    os.path.join(TEST_DATA_DIR, r"weather_day_pytest.csv"),
    index_col=0,
    parse_dates=True,
)

with open(os.path.join(TEST_DATA_DIR, "meta.json"), "r") as file:
    META = json.load(file)


CORRELATED_SAMPLES_1 = pd.read_csv(
    os.path.join(TEST_DATA_DIR, r"correlated_samples_arrhenius.csv"),
)

CORRELATED_SAMPLES_2 = pd.read_csv(
    os.path.join(TEST_DATA_DIR, r"noCorrEmpty_samples_arrhenius.csv")
)

CORRELATED_SAMPLES_3 = pd.read_csv(
    os.path.join(TEST_DATA_DIR, r"noCorrR0_arrhenius.csv")
)

ARRHENIUS_RESULT = pd.read_csv(
    os.path.join(TEST_DATA_DIR, r"monte_carlo_arrhenius.csv"),
)


def test_generateCorrelatedSamples():
    """Test pvdeg.montecarlo.generateCorrelatedSamples.

    Requires:
    ---------
    list of correlations, stats dictionary
    (mean and standard deviation for each variable), number of iterations, seed,
    DataFrame to check against
    """
    # standard case
    result_1 = pvdeg.montecarlo.generateCorrelatedSamples(
        corr=[
            pvdeg.montecarlo.Corr("Ea", "X", 0.0269),
            pvdeg.montecarlo.Corr("Ea", "LnR0", -0.9995),
            pvdeg.montecarlo.Corr("X", "LnR0", -0.0400),
        ],
        stats={
            "Ea": {"mean": 62.08, "stdev": 7.3858},
            "LnR0": {"mean": 13.7223084, "stdev": 2.47334772},
            "X": {"mean": 0.0341, "stdev": 0.0992757},
        },
        n=50,
        seed=1,
    )

    # EMPTY CORRELATION LIST
    result_2 = pvdeg.montecarlo.generateCorrelatedSamples(
        corr=[],
        stats={
            "Ea": {"mean": 62.08, "stdev": 7.3858},
            "LnR0": {"mean": 13.7223084, "stdev": 2.47334772},
            "X": {"mean": 0.0341, "stdev": 0.0992757},
        },
        n=50,
        seed=1,
    )

    # populated correlation list, ALL R = 0
    result_3 = pvdeg.montecarlo.generateCorrelatedSamples(
        corr=[
            pvdeg.montecarlo.Corr("Ea", "X", 0),
            pvdeg.montecarlo.Corr("Ea", "LnR0", 0),
            pvdeg.montecarlo.Corr("X", "LnR0", 0),
        ],
        stats={
            "Ea": {"mean": 62.08, "stdev": 7.3858},
            "LnR0": {"mean": 13.7223084, "stdev": 2.47334772},
            "X": {"mean": 0.0341, "stdev": 0.0992757},
        },
        n=50,
        seed=1,
    )

    pd.testing.assert_frame_equal(result_1, CORRELATED_SAMPLES_1)
    pd.testing.assert_frame_equal(result_2, CORRELATED_SAMPLES_2)
    pd.testing.assert_frame_equal(result_3, CORRELATED_SAMPLES_3)


def test_simulate():
    """Test pvdeg.montecarlo.simulate.

    Requires:
    ---------
    target function, correlated samples dataframe, weather dataframe, meta dictionary
    """

    poa_irradiance = pvdeg.spectral.poa_irradiance(WEATHER, META)
    temp_mod = pvdeg.temperature.module(
        weather_df=WEATHER,
        meta=META,
        poa=poa_irradiance,
        conf="open_rack_glass_polymer",
    )

    poa_global = poa_irradiance["poa_global"].to_numpy()
    cell_temperature = temp_mod.to_numpy()

    function_kwargs = {"poa_global": poa_global, "module_temp": cell_temperature}

    new_results = pvdeg.montecarlo.simulate(
        func=pvdeg.degradation.vecArrhenius,
        correlated_samples=CORRELATED_SAMPLES_1,
        **function_kwargs,
    )

    new_results_df = pd.DataFrame(new_results)

    new_results_df.columns = ARRHENIUS_RESULT.columns

    pd.testing.assert_frame_equal(new_results_df, ARRHENIUS_RESULT)
