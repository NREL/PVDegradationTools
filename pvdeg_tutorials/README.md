<img src="PVD_logo.png" width="100">

# TUTORIALS for PV Degradation Tools (pvdeg)

### Jupyter Book

For in depth Tutorials you can run online, see our [jupyter-book](https://nrel.github.io/PVDegradationTools/intro.html)
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

We also have documentation in [ReadTheDocs](https://PVDegradationTools.readthedocs.io) where you can find more details on the API functions.
