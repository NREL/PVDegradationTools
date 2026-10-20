# Welcome to the PVDEG Tutorials

Here is a collection of Jupyter Notebooks that provide tutorials for using PVDeg.
These are ready to read, install and use, or run in your browser through Google Colab.
Clicking on the rocket-icon on the top allows you to launch the notebooks on
[Google Colaboratory](https://colab.research.google.com/) for interactive mode.
As per the instructions in each notebook, you should uncomment the first line
that reads `pip install ...` to install the environment if you follow this mode.

You can also clone the repository and run them locally, following the
instructions on the [PVDeg Github page](https://github.com/NatLabRockies/PVDegradationTools)

## Tutorial Categories

**[01 - Basics](01_basics/01_basics_humidity_design.ipynb)** - Introduction to PVDeg fundamentals including humidity design, degradation overview, spectral analysis, and weather database access

**[02 - Degradation Mechanisms](02_degradation/01_bo_lid_accelerated_test.ipynb)** - Detailed modeling of specific degradation mechanisms: B-O LID, LETID (accelerated tests, outdoor environments, scenarios), and Van't Hoff models

**[03 - Monte Carlo Simulations](03_monte_carlo/01_arrhenius.ipynb)** - Uncertainty quantification through Monte Carlo methods for Arrhenius degradation and standoff calculations

**[04 - Scenarios](04_scenario/01_scenario_temperature.ipynb)** - Scenario-based workflows allowing for easy analysis of different PV module materials or deployment locations.

**[05 - Geospatial Analysis](05_geospatial/01_geospatial_templates.ipynb)** - Large-scale spatial analysis, geospatial templates, and world map visualizations for regional degradation assessments (HPC)

**[06 - Advanced Topics](06_advanced/01_custom_functions_nopython.ipynb)** - Custom functions with Numba, PySAM integration, and distributed computing with NSRDB/PVGIS APIs

**[10 - Workshop Demos](10_workshop_demos/01_astm_live_demo.ipynb)** - Interactive demonstrations from ASTM and DuraMAT workshops

## Tools

**[Edge Seal Oxygen Ingress Calculator](tools/Tools%20-%20Edge%20Seal%20Oxygen%20Ingress.ipynb)** - Calculation of oxygen ingress profile through an edge seal and into the encapsulant

**[Degradation and Acceleration Factors](tools/Tools%20-%20Degradation.ipynb)** - Estimation of degradation and calculation of acceleration factors using the degradation database

**[Module Standoff for IEC TS 63126](tools/Tools%20-%20Module%20Standoff%20for%20IEC%20TS%2063126.ipynb)** - Calculation of module standoff distance according to IEC TS 63126
