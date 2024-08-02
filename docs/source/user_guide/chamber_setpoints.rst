.. _chamber-setpoints:

Chamber Stress Testing
======================

Accelerated Stress Test Chambers allow us to preform accelerated PV testing on samples. 

The class provided by `pvdeg.Chamber` allows us to simulate test chamber conditions and sample conditions during a test.

Initialize a chamber object as shown below. We need to define the behavior of the chamber over time, this is done with a CSV file containing setpoints.

.. code:: python

    test_chamber = pvdeg.Chamber(
        fp = "path/to/your/setpoints.csv",
        setpoint_names = ["temperature", "relative_humidity", "irradiance_340"]
    )

Chamber Test Tutorial
=====================
There is a comprehensive jupyter notebook tutorial for Chamber testing at ``PVDegradationTools/tutorials_and_tools/tutorials_and_tools/"Chamber Stressor.ipynb"``. This will also be added to the pvdeg jupyterbook soon.

Setpoints CSV Schema
====================

All time units for the setpoints csv must be in MINUTES.

Required Columns
-----------------
- `step_length`
- `step_divisions`
- `..._ramp`


`step_length` gives the duration of a setpoint in minutes. `step_divisions` gives us the number of points to create within the setpoint timeseries for a given setpoint (think of this as a way to control temporal resolution).

.. list-table:: Time Columns
   :widths: 25 25
   :header-rows: 1

   * - step_length
     - step_divisions
   * - 15 
     - 5
   * - 60 
     - 10
   * - 5
     - 1

Lets Review what each of these values mean.  

- Row 1: 15 minutes long, with 5 divisions, we will have 3 timeseries values at this setpoint in the chamber/sample calulcations.  
- Row 2: 60 mintues (1 hr), 10 divisions, 6 timeseries values in calculations.  
- Row 3: 5 minutes, 1 division, 1 timeseries value in calculations.  

In total the test will last 80 minutes with 16 values in the timeseries.

Adding Setpoints
----------------
For each setpoint column given in the CSV, we must include a ramp rate (this is how fast we transition into the setpoint from the old setpoint in [units/minute]). replace the `...` with your setpoint name.
given a setpoint column name of `temperature` we must include a `tempeature_ramp` column. Populate with `0` values for instant transitions.

To add a `temperature` setpoint to the CSV shown above, we will need to add the corresponding ramp, this will be called `temperature_ramp`. We want instant changes so we will set the ramp values to 0.

.. list-table:: Adding Setpoints
   :widths: 25 25 25 25
   :header-rows: 1

   * - step_length
     - step_divisions
     - temperature
     - temperature_ramp
   * - 15 
     - 5
     - 25
     - 0
   * - 60 
     - 10
     - 85
     - 0
   * - 5
     - 1
     - 25
     - 0

Reviewing again:  

- Row 1: 15 minutes at a termperature of 25 [C], instant change of setpoint up to 25, this is not actually the chamber temperature, just the temperature that the chamber heating element is trying to raise the chamber air temperature to.
- Row 2: 60 minutes at a 85 [C]
- Row 3: 5 minute at 25 [C]


Adding More Setpoints
---------------------

In the following code block with the constructor, we tell the method to create a `Chamber` with the setpoints ["temperature", "relative_humidity", "irradiance_340"]. This means we need to have all of these columns and corresponding ramp rates in the CSV. This is shown below. This CSV will work with the following code block (same as above)

.. code:: python

    test_chamber = pvdeg.Chamber(
        fp = "path/to/your/setpoints.csv",
        setpoint_names = ["temperature", "relative_humidity", "irradiance_340"]
    )
 

.. list-table:: Adding Additional Setpoints
   :widths: 10 10 10 10 10 10 10 10
   :header-rows: 1

   * - step_length
     - step_divisions
     - temperature
     - temperature_ramp
     - relative_humidity
     - relative_humidity_ramp
     - irradiance_340
     - irradiance_340_ramp
   * - 15 
     - 5
     - 25
     - 0
     - 40
     - 0
     - 0
     - 0
   * - 60 
     - 10
     - 85
     - 0
     - 85
     - 0
     - 0
     - 0
   * - 5
     - 1
     - 25
     - 0
     - 50
     - 0
     - 0
     - 0

Adding units
============

Pandas cannot handle units traditonally but we can include them in our file. However, this changes how we read the csv. 

.. list-table:: CSV with units
   :widths: 10 10 10 10 10 10 10 10
   :header-rows: 1

   * - step_length
     - step_divisions
     - temperature
     - temperature_ramp
     - relative_humidity
     - relative_humidity_ramp
     - irradiance_340
     - irradiance_340_ramp
   * - [minutes]
     - unitless
     - [C]
     - [C/minute]
     - [%]
     - [%/minute]
     - [w/m^2/nm @ 340 nm]
     - [(w/m^2/nm @ 340 nm)/minute]
   * - 15 
     - 5
     - 25
     - 0
     - 40
     - 0
     - 0
     - 0
   * - 60 
     - 10
     - 85
     - 0
     - 85
     - 0
     - 0
     - 0
   * - 5
     - 1
     - 25
     - 0
     - 50
     - 0
     - 0
     - 0

To ignore the row with units, we can use the `skipprows` argument provided by `pandas.read_csv`_ as documented in the `pandas documentation <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html>`_. The code block below reads the CSV properly, skipping the row with units. *Note: we have to pass a list or we will skip the first row*.

.. code:: python

    chamber = pvdeg.Chamber(
        fp=r"./chamber-setpoints.csv",
        setpoint_names=["temperature", "relative_humidity", "irradiance_340"],
        skiprows=[1] # skip the row at index 1
    )
