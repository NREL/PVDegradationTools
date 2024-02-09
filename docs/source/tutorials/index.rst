.. _tutorials:

==========
Tutorials
==========

Gallery Coming Soon  
In the mean time check the jupyter-book for interactive trainings  
*Nbgallery element*  

Jupyter Book
------------

For in depth Tutorials you can run online, see our `jupyter-book
<https://nrel.github.io/PVDegradationTools/intro.html>`_
Clicking on the rocket-icon on the top allows you to launch the journals on `Google Colaboratory
<https://colab.research.google.com/>`_ 
for interactive mode.
Just uncomment the first line `pip install ...`  to install the environment on each journal if you follow this mode.

Binder
------

To run these tutorials in Binder, you can click here:

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/NREL/PVDegradationTools/main
    :alt: Binder

It takes a minute to load the environment.

Locally
-------

You can also run the tutorial locally in a virtual environment, i.e., `venv` or
`miniconda
<https://docs.conda.io/en/latest/miniconda.html>`_.

1. Create and activate a new environment, e.g., on Mac/Linux terminal with `venv`:
   ``python -m venv pvdeg . pvdeg/bin/activate``
   or with `conda`:
   ``conda create -n pvdeg conda activate pvdeg``

1. Install `pvdeg` into the new environment with `pip`:
   ``python -m pip install pvdeg``

1. Start a Jupyter session:
   ``jupyter notebook``

1. Use the file explorer in Jupyter lab to browse to `tutorials`
   and start the first Tutorial.

