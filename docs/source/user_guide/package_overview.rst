.. _package_overview:

Package Overview
================

pvdeg provides a python library of common degradation modes for fielded
photovoltaics and accelerated testing.
It currently offers functions to calculate test-chamber irradiance settings,
the humidity of PV materials, the spectral degradation in backsheets, and more.
Functionality has been simplified so you can use .psm3 weather files retrieved
from NREL's `National Solar Radiation Database (NSRDB) <https://nsrdb.nrel.gov/>`_.

In some cases, such as calculating the relative backsheet spectral degradation,
you will need spectraly resolved irradiance. This can be field data or data
produced via simulation (for example: results from `bifacial_radiance
<https://github.com/NREL/bifacial_radiance>`_)

**Package Functions:**

1. Stress Factors

  * Water Vapor Pressure, empirical model
  * Water Ingress Rate through PV module
  * Edge Seal Width for 25 year lifetime

2. Degradation

  * Van 't Hoff degradation acceleration factor
  * Van 't Hoff environment characterization
  * Arrhenius degradation acceleration factor
  * Arrhenius environment characterization
  * Spectral Degradation
  * LETID outdoor, lab
  * Solder Fatigue

3. Standards

  * Ideal Installation Distance
  * Effective Operating Temperatures

4. Monte Carlo

  * Uncorrelated Monte Carlo simulation
  * Multi-variate Correlated Monte Carlo simulation

5. Geospatial

  * Geospatial analysis framework
  * High Performance Computing Tools

6. Weather Utilities

  * Parallelized downloads of NSRDB, and PVGIS

For an in depth look at each class and function, please refer to API