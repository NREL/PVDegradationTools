.. pvdeg documentation master file, created by
   sphinx-quickstart on Thu Jan 18 15:25:51 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. .. image:: ../../tutorials_and_tools/pvdeg_logo.png
..    :width: 500

.. image:: ./_static/logo-vectors/PVdeg-Logo-Horiz-Color.svg


Welcome to pvdeg!
==============================================================

This PVDegradation Tools Model is a first-of-its-kind detailed spatio-temporal
modeling assessment tool that empowers users to calculate various PV degradation
modes, for different PV technologies and materials in its database.

NREL is developing the pvdeg model to help PV Researchers, Module manufacturers,
and other PV stakeholders assess different materials and technologies for
different degradation modes in all locations throughout the world. Available as
open source, the PVDegradation model currently supports PV energy calculations
for water vapor pressure, ingress rate, edge seal width, Van Hofft Irradiance
Degradation, Weighted Average Irradiance, Arrehnius Acceleration Factor,
relative humidity in the outside, front and back encapsulant and backsheet,
spectral degradation, and solder fatigue. More functions for standards and
other degradation profiles are in the works.

The source code for pvdeg is hosted on `github <https://github.com/NREL/pvdeg>`_. Please see the :ref:`installation` page for installation help.

See :ref:`tutorials` to learn how to use and experiment with various functionalities


.. image::  ./_static/PVDeg-Flow.svg
    :alt: PVDeg-Flow diagram.


How the Model Works
===================

Coupled with pvlib for module performan and weather/irradiance calculations,
PVDegradation Tool estimates degradations, and accelerated factors on
user-defined parameters. The `Data Library` is under development as part of the
PVDegradationTool project, compiling literature parameters and functions.

The PVDegradationTool simulatineously reads tens of terabytes of time-series
solar data from state-of-art resource data set National Solar Radiation Database
(NSRDB), publicly available on the cloud, enabling the execution of pvdeg
beyond the confines of NREL's high-performance computing capabilities.

Citing PVDegradation Tools
==========================

If you use this calculator in a published work, please cite:

.. code-block::

   Holsapple, Derek, Ayala Pelaez, Silvana, Kempe, Michael. "PV Degradation Tools", NREL Github 2020, Software Record SWR-20-71.


Please also cite the DOI corresponding to the specific version that you used.
DOIs are listed at Zenodo.org. `linked here <https://zenodo.org/records/13760911>`_

.. code-block::

   Martin Springer, Tobin Ford, Matt, MDKempe, Silvana Ovaitt, AidanWesley, joe karas, Mark Campanelli, Derek M Holsapple, and Kevin Anderson. “NREL/PVDegradationTools: 0.4.2.” Zenodo, 2024. doi:10.5281/zenodo.13760911.


.. toctree::
   :hidden:
   :titlesonly:

   user_guide/index
   tutorials/index
   api
   whatsnew/index

..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
