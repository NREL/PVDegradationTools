import pvdeg
from pvdeg import TEST_DATA_DIR
import pandas as pd
import numpy as np
import pytest
import os

CHAMBER_CONDITIONS = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "chamber_conditions_result.csv"), index_col=0
)
CHAMBER_CONDITIONS.index = CHAMBER_CONDITIONS.index.astype("timedelta64[s]")

SAMPLE_CONDITIONS = pd.read_csv(
    os.path.join(TEST_DATA_DIR, "sample_conditions_results.csv"), index_col=0
)
SAMPLE_CONDITIONS.index = SAMPLE_CONDITIONS.index.astype("timedelta64[s]")

test_chamber = pvdeg.chamber.Chamber(
    os.path.join(TEST_DATA_DIR, "chamber-setpoints.csv"),
    setpoint_names=["temperature", "relative_humidity"],
    skiprows=[1],
)
test_chamber.setpoints = test_chamber.setpoints.iloc[:100]  # we only care about the first 100 setpoints


def test_chamber_conditions():
    test_chamber.setpoints = test_chamber.setpoints  # .iloc[:100]
    chamber_result = test_chamber.chamber_conditions(tau_c=10, air_temp_0=25)

    pd.testing.assert_frame_equal(chamber_result, CHAMBER_CONDITIONS, check_dtype=False)


def test_chamber_sample_conditions():
    test_chamber.setpoints = test_chamber.setpoints  # .iloc[:100]
    test_chamber.setBacksheet(id="ST504", thickness=0.5)  # PET
    test_chamber.setEncapsulant(id="EVA", thickness=0.1)  # EVA
    sample_result = test_chamber.sample_conditions(
        tau_s=15, sample_temp_0=25, n_steps=20
    )

    pd.testing.assert_frame_equal(sample_result, SAMPLE_CONDITIONS)


def test_chamber_plotsetpoints():
    test_chamber.setpoints = test_chamber.setpoints  # .iloc[:100]
    try:
        test_chamber.plot_setpoints()
    except Exception as e:
        pytest.fail(f"plot_setpoints() raised an exception: {e}")


def test_chamber_plottemperatures():
    test_chamber.setpoints = test_chamber.setpoints
    try:
        test_chamber.plot_temperatures()
    except Exception as e:
        pytest.fail(f"plot_setpoints() raised an exception: {e}")


def test_chamber_calc_temperatures_no_irradiance():
    test_chamber.calc_temperatures(air_temp_0=25, sample_temp_0=25, tau_c=10, tau_s=15)

    pd.testing.assert_series_equal(
        test_chamber.air_temperature,
        CHAMBER_CONDITIONS["Air Temperature"],
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        test_chamber.sample_temperature,
        SAMPLE_CONDITIONS["Sample Temperature"],
        check_dtype=False,
    )


def test_setpoint_series_bad():
    bad_df = pd.DataFrame(np.nan, index=pd.RangeIndex(5), columns=["Temperature"])

    with pytest.raises(ValueError) as excinfo:
        pvdeg.chamber.setpoint_series(df=bad_df, setpoint_name="Temperature")

    assert (
        str(excinfo.value)
        == "column: Temperature contains NaN values. Remove from setpoint list or remove NaN's in input."
    )


def test_num_steps():
    assert 3 == pvdeg.chamber.num_steps(9, 3)
    assert 4 == pvdeg.chamber.num_steps(7, 2)


def test_linear_ramp_time():
    assert 2 == pvdeg.chamber.linear_ramp_time(y_0=5, y_f=10, rate=2.5)
