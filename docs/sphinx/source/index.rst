

Welcome to pvdeg!
==============================================================

This PVDegradation Tools Model is a first-of-its-kind detailed spatio-temporal modeling assessment tool that empowers users to calcualte various PV degradation modes, with various PV technology and materials databases.

NREL developed the pvdeg model to help PV Researchers, Module manufacturesrs, and ohter PV stakeholders assess different materials and technologies for different degradation modes in all locations through the world. Available as open source since January 2023, the PVDegradation model currently supports PV energy calculations for water vapor pressure, ingres rate, edge seal widht, Van Hofft Irradiance Degradation, Weighted Average Irradiance, Arrehnius Acceleration Factor, relative humidity in the outside, front and back encapsulant and backsheet,spectral degradation, and solder fatigue. More functions for standards and other degradation profiles are in the works.

The source code for pvdeg is hosted on `github
<https://github.com/NREL/pvdeg>`_.

Please see the :ref:`installation` page for installation help.

How the Model Works
===================
Coupled with pvlib for module performan and weather/irradiance calculations, the PVDegradation Tool estiamtes degradations, and accelerated factors on user-defined parameters including [working on long list here]

The PVDegradationTool simulatineously reads tens of terabytes of time-series solar data from state-of-art resource data set National Solar Radiation Database (NSRDB), publicly avialable no the cloud, enabling the execution of pvdeg beyond the confines of NREL's high-performance computing capabilities.


Citing PVDegradation Tools
==========================

If you use this calculator in a published work, please cite:

        

Please also cite the DOI corresponding to the specific version that you used. DOIs will be listed at Zenodo.org


Contents
========

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   package_overview
   whatsnew
   installation
   api
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
