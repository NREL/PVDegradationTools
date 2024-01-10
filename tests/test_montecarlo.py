# TODO:
# add data to compare against

import pytest
import os 
import json 
import pandas as pd
import pvdeg
from pvdeg import TEST_DATA_DIR

WEATHER = pd.read_csv(
    os.path.join(TEST_DATA_DIR, r"weather_day_pytest.csv"),
    index_col=0,
    parse_dates=True
)

with open(os.path.join(TEST_DATA_DIR, "meta.json"), "r") as file:
    META = json.load(file)

def test_generateCorrelatedSamples():
    """
    test pvdeg.montecarlo.generateCorrelatedSamples    

    Requires:
    ---------
    list of correlations, stats dictionary (mean and standard deviation for each variable), number of iterations, seed
    """
    result = pvdeg.montecarlo.generateCorrelatedSamples(
        corr=[pvdeg.montecarlo.Corr('Ea', 'X', 0.0269), pvdeg.montecarlo.Corr('Ea', 'LnR0', -0.9995), pvdeg.montecarlo.Corr('X', 'LnR0', -0.0400)],
        stats={'Ea' : {'mean' : 62.08, 'stdev' : 7.3858 }, 'LnR0' : {'mean' : 13.7223084 , 'stdev' : 2.47334772}, 'X' : {'mean' : 0.0341 , 'stdev' : 0.0992757}},
        n = 20000,
        seed = 1
    )

    # want to generate the result and store it somewhere then compare
    pd.testing.assert_frame_equal(result, )


# how can i test this with two different target functions
# I assume I should not make a second test_simulate function
# can they both be in there
def test_simulate():
    """
    test pvdeg.montecarlo.simulate

    Requires:
    ---------
    target function, correlated samples dataframe, weather dataframe, meta dictionary
    """

    sol_pos = pvdeg.spectral.solar_position(WEATHER, META)
    poa_irradiance = pvdeg.spectral.poa_irradiance(WEATHER, META)
    temp_mod = pvdeg.temperature.module(weather_df=WEATHER, meta=META, poa=poa_irradiance, conf='open_rack_glass_polymer')
    poa_global = poa_irradiance['poa_global'].to_numpy()
    cell_temperature = temp_mod.to_numpy()

    function_kwargs = {'poa_global': poa_global, 'module_temp': cell_temperature}

    results = pvdeg.montecarlo.simulate(
        func=pvdeg.degradation.vecArrhenius,
        correlated_samples=mc_inputs, # WILL READ INPUTS FROM FILE 
        **function_kwargs
    )
    
    # want to generate the result and store it somewhere then compare
    pd.testing.assert_frame_equal(results, )