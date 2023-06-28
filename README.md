<img src="pvdeg_tutorials/PVD_logo.png" width="100">

<table>
<tr>
  <td>Version</td>
  <td>
  <a href="https://zenodo.org/badge/latestdoi/248347431"> FORTHCOMING </a>
</td>
</tr>

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
	[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8088403.svg)](https://doi.org/10.5281/zenodo.8088403)
  </td>
</tr>
<tr>
  <td>Documentation</td>
  <td>
	<a href='https://pvdegradationtools.readthedocs.io/en/latest/?badge=latest'>
	    <img src='https://readthedocs.org/projects/pvdegradationtools/badge/?version=latest' alt='Documentation Status' />
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

You can also run the tutorial locally with
[miniconda](https://docs.conda.io/en/latest/miniconda.html) by following thes
steps:

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).

1. Clone the repository:

   ```
   git clone https://github.com/NREL/PVDegradationTools.git
   ```

1. Create the environment and install the requirements. The repository includes
   a `requirements.txt` file that contains a list the packages needed to run
   this tutorial. To install them using conda run:

   ```
   conda create -n pvdeg jupyter -c pvlib --file requirements.txt
   conda activate pvdeg
   ```

   or you can install it with `pip install pvdeg` as explained in the installation instructions into the environment.

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

For developer installation, download the repository, navigate to the folder location and install as:

    pip install -e .


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

	Ovaitt, Silvana, Brown, Matt, Springer, Martin, Karas, Joe, Holsapple, Derek, Kempe, Michael. (2023). NREL/PVDegradationTools: v0.1.0 official release (0.1.0). Zenodo. https://doi.org/10.5281/zenodo.8088403
