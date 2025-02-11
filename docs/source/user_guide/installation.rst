.. _installation:

Installation
============

pvdeg releases may be installed using the pip and conda tools.

Base Install:
---------------------
To get PVDeg the simplest way use:

.. code::

        pip install pvdeg

Optional Install:
----------------
PVDeg provides optional installs for testing, and documenation. We can specify these using `[]` syntax.


.. list-table:: Extra Installs in PVDeg
   :widths: 30 70
   :header-rows: 1

   * - **Extra Install Name**
     - **Explanation**
   * - ``docs``
     - Documentation dependencies
   * - ``test``
     - Testing dependencies
   * - ``books``
     - Jupyter utilities
   * - ``sam``
     - NREL System Advisor Model dependencies
   * - ``all``
     - all of the above names combined

To download packages required for testing, run:

.. code::

        pip install pvdeg[test]

To download all optional packages, run:

.. code::

        pip install pvdeg[all]


Developer Installation
----------------------


For developer installation, use the following steps.


1. clone the reposityory using 

.. code::
        
        git clone https://github.com/NREL/PVDegradationTools.git

2. Navigate the the root of the repository such that you are in ``path/to/PVDegradationTools``, then run the following command.

.. code::

        pip install -e .

Compatible with Python 3.9 and above.
