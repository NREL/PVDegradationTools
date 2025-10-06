<img src="https://raw.githubusercontent.com/NREL/PVDegradationTools/refs/heads/main/docs/source/_static/logo-vectors/PVdeg-Logo-Horiz-Color.svg" width="600">


<table>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/NREL/PVDegradationTools/blob/master/LICENSE.md">
    <img src="https://img.shields.io/pypi/l/pvlib.svg" alt="license" />
    </a>
</td>
</tr>
<tr>
  <td>Publications</td>
  <td>
     <a href="https://doi.org/10.5281/zenodo.8088578"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8088578.svg" alt="DOI"></a>
  </td>
</tr>
<tr>
  <td>Documentation</td>
  <td>
	<a href='https://PVDegradationTools.readthedocs.io'>
	    <img src='https://readthedocs.org/projects/pvdegradationtools/badge/?version=stable' alt='Documentation Status' />
	</a>
  </td>
</tr>
<tr>
  <td>Build status</td>
  <td>
   <a href="https://github.com/NREL/PVDegradationTools/actions/workflows/pytest.yml?query=branch%3Amain">
      <img src="https://github.com/NREL/PVDegradationTools/actions/workflows/pytest.yml/badge.svg?branch=main" alt="GitHub Actions Testing Status" />
   </a>
   <a href="https://codecov.io/gh/NREL/PVDegradationTools" >
   <img src="https://codecov.io/gh/NREL/PVDegradationTools/graph/badge.svg?token=4I24S8BTG7"/>
   </a>
  </td>
</tr>
</table>



# PV Degradation Tools (pvdeg)

This repository contains functions for calculating degradation of photovoltaic modules. For example, functions to calculate front and rear relative Humidity, as well as Acceleration Factors. A degradation calculation function is also being developed, considering humidity and spectral irradiances models.


Tutorials
=========

### Jupyter Book

For in depth Tutorials you can run online, see our [jupyter-book](https://nrel.github.io/PVDegradationTools/intro.html) [![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://nrel.github.io/PVDegradationTools/intro.html)

Clicking on the rocket-icon on the top allows you to launch the journals on [Google Colaboratory](https://colab.research.google.com/) for interactive mode.
Just uncomment the first line `pip install ...`  to install the environment on each journal if you follow this mode.

### Binder

To run these tutorials in Binder, you can click here:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NREL/PVDegradationTools/main)
It takes a minute to load the environment.

### Locally

You can also run the tutorial locally in a virtual environment, i.e., `venv` or
[miniconda](https://docs.conda.io/en/latest/miniconda.html).

1. Create and activate a new environment, e.g., on Mac/Linux terminal with `venv`:
   ```
   python -m venv pvdeg
   . pvdeg/bin/activate
   ```
   or with `conda`:
   ```
   conda create -n pvdeg
   conda activate pvdeg
   ```

1. Install `pvdeg` into the new environment with `pip`:
   ```
   python -m pip install pvdeg
   ```

1. Start a Jupyter session:

   ```
   jupyter notebook
   ```

1. Use the file explorer in Jupyter lab to browse to `tutorials`
   and start the first Tutorial.


Documentation
=============

Documentation is available in [ReadTheDocs](https://PVDegradationTools.readthedocs.io) where you can find more details on the API functions.


Installation
============

Relative Humidity and Acceleration Factors for Solar Modules releases may be installed using the ``pip`` and ``conda`` tools. Compatible with Python 3.5 and above.

Install with:

    pip install pvdeg

For developer installation, clone the repository, navigate to the folder location and install as:

    pip install -e .[all]

Running jupyter notebooks using anaconda prompt
===============================================

Note that in order to run notebooks cleanly and validate outputs, use the following
commands to run either one notebook:

    jupyter nbconvert --to notebook --execute --inplace "tutorials_and_tools/
    tutorials_and_tools/Monte Carlo - Arrhenius.ipynb"

or all notebooks inside the tutorials and tools folder:

    jupyter nbconvert --to notebook --execute --inplace "tutorials_and_tools/
    tutorials_and_tools/*.ipynb"

This avoids formatting issues that may arise depending on your own local environment
or IDE.


License
=======

[BSD 3-clause](https://github.com/NREL/PVDegradationTools/blob/main/LICENSE.md)


Contributing
=======

We welcome contributiosn to this software, but please read the copyright license agreement (cla-1.0.md), with instructions on signing it in sign-CLA.md. For questions, email us.


Getting support
===============

If you suspect that you may have discovered a bug or if you'd like to
change something about pvdeg, then please make an issue on our
[GitHub issues page](hhttps://github.com/NREL/PVDegradationTools/issues).


Citing
======

If you use this functions in a published work, please cite:

	Holsapple, Derek, Ayala Pelaez, Silvana, Kempe, Michael. "PV Degradation Tools", NREL Github 2020, Software Record SWR-20-71.

And/or the specific release from Zenodo:

	Martin Springer, Tobin Ford, Rajiv Daxini, Matthew Brown, Silvana Ovaitt, Joseph Karas, Mark Campanelli, Derek M Holsapple, Kevin Anderson, Michael Kempe. (2025). NREL/PVDegradationTools: 0.6.1 (0.6.1). Zenodo. https://doi.org/10.5281/zenodo.17265988
