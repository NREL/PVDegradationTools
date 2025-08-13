import pytest
import os
import json
import numpy as np
import pandas as pd
import sympy as sp  # not a dependency, may cause issues
import pvdeg
from pvdeg import TEST_DATA_DIR

WEATHER = pd.read_csv(
    os.path.join(TEST_DATA_DIR, r"weather_day_pytest.csv"),
    index_col=0,
    parse_dates=True,
)

with open(os.path.join(TEST_DATA_DIR, "meta.json"), "r") as file:
    META = json.load(file)

# D = k_d  * E * Ileak
# degradation rate, D
# degradation constant, k_d

# electric field, E = Vbias / d
# Vbias, potential diference between cells and frame
# d, encapsulant thickness

# leakage current, Ileak = Vbias / Rencap
# Vbias, potential diference between cells and frame
# Rencap, resistance of encapsulant

k_d, Vbias, Rencap, d = sp.symbols("k_d Vbias Rencap d")
pid = k_d * (Vbias / d) * (Vbias / Rencap)

pid_kwarg = {
    "Vbias": 1000,
    "Rencap": 1e9,
    "d": 0.0005,
    "k_d": 1e-9,
}


def test_symbolic_floats():
    res = pvdeg.symbolic.calc_kwarg_floats(expr=pid, kwarg=pid_kwarg)

    assert res == pytest.approx(2e-9)


def test_symbolic_df():
    pid_df = pd.DataFrame([pid_kwarg] * 5)

    res_series = pvdeg.symbolic.calc_df_symbolic(expr=pid, df=pid_df)

    pid_values = pd.Series([2e-9] * 5)

    pd.testing.assert_series_equal(res_series, pid_values, check_dtype=False)


def test_symbolic_timeseries():
    lnR_0, I, X, Ea, k, T = sp.symbols("lnR_0 I X Ea k T")
    ln_R_D_expr = lnR_0 * I**X * sp.exp((-Ea) / (k * T))

    module_temps = pvdeg.temperature.module(
        weather_df=WEATHER, meta=META, conf="open_rack_glass_glass"
    )
    poa_irradiance = pvdeg.spectral.poa_irradiance(weather_df=WEATHER, meta=META)

    module_temps_k = module_temps + 273.15  # convert C -> K
    poa_global = poa_irradiance[
        "poa_global"
    ]  # take only the global irradiance series from the total irradiance dataframe
    poa_global_kw = poa_global / 1000  # [W/m^2] -> [kW/m^2]

    values_kwarg = {
        "Ea": 62.08,  # activation energy, [kJ/mol]
        "k": 8.31446e-3,  # boltzmans constant, [kJ/(mol * K)]
        "T": module_temps_k,  # module temperature, [K]
        "I": poa_global_kw,  # module plane of array irradiance, [W/m2]
        "X": 0.0341,  # irradiance relation, [unitless]
        "lnR_0": 13.72,  # prefactor degradation [ln(%/h)]
    }

    res = pvdeg.symbolic.calc_kwarg_timeseries(
        expr=ln_R_D_expr, kwarg=values_kwarg
    ).sum()

    assert res == pytest.approx(6.5617e-09)


def test_calc_df_symbolic_bad():
    expr = sp.symbols("not_in_columns")
    df = pd.DataFrame([[1, 2, 3, 5]], columns=["a", "b", "c", "d"])

    with pytest.raises(ValueError):
        pvdeg.symbolic.calc_df_symbolic(expr=expr, df=df)


def test_calc_kwarg_timeseries_bad_type():
    # try passing an invalid argument type
    with pytest.raises(ValueError, match="only simple numerics or timeseries allowed"):
        pvdeg.symbolic.calc_kwarg_timeseries(expr=None, kwarg={"bad": pd.DataFrame()})


def test_calc_kwarg_timeseries_bad_mismatch_lengths():
    # arrays of different lengths
    with pytest.raises(
        NotImplementedError,
        match="arrays/series are different lengths",
    ):
        pvdeg.symbolic.calc_kwarg_timeseries(
            expr=None,
            kwarg={
                "len1": np.zeros((5,)),
                "len2": np.zeros(
                    10,
                ),
            },
        )
