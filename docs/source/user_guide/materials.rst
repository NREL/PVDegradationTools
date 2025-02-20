.. materials::

Materials Storage and Access
============================

PVDeg contains a library of material parameters suitable for estimating the durability of materials and components.


These material parameters and other relevant information sit in a directiory at ``PVDegradationTools/pvdeg/data``. 

This location can be quickly accessed through a special variable as shown below.

.. code-block:: Python

    import pvdeg

    file_path = os.path.join(pvdeg.DATA_DIR, <file-to-access.ext>)

.. code-block:: Python

    from pvdeg import DATA_DIR

    file_path = os.path.join(DATA_DIR, <file-to-access.ext>)

File Organization
------------------------------------
There are many files in this directory. We will generally be interested in one of the following files.


- `AApermeation.json <AApermeation_>`_ (acetic acid permeation parameters)
- `H2Opermeation.json <H2Opermeation_>`_ (water permeation parameters)
- `O2permeation.json <O2permeation_>`_ (oxygen permeation parameters)
- kinetic_parameters.json (letid/bolid parameters)
- DegradationDatabase.json (degredation models)

Material Parameters
------------------------------------
Each of the material permeation parameters files above is a json indexed by arbitrary names.
These are not a mapping of material names or aliases and are not consistent across the three files below.

- `AApermeation.json <AApermeation_>`_ (acetic acid permeation parameters)
- `H2Opermeation.json <H2Opermeation_>`_ (water permeation parameters)
- `O2permeation.json <O2permeation_>`_ (oxygen permeation parameters)

Accessing Material Parameters
-----------------------------

PVDeg provides convenience methods/functions to access material parameters. ``pvdeg.utilities.read_material`` is the simplest way to access material parameters. We will also show a sample use.

.. autofunction:: pvdeg.utilities.read_material

.. code-block:: Python

    material_dict = pvdeg.utilities.read_material(
        pvdeg_file = "AApermeation",
        key = "AA001",
    )

.. code-block:: Python

    material_dict = pvdeg.utilities.read_material(
        pvdeg_file = "H2Opermeation",
        key = "W003",
    )


The result of both of these functions will be a dictionary that looks like the following. The keys may vary depending on the structure of the json but this is the general idea.

.. code-block:: Python

    {
		"name": string,
		"alias": string,
		"contributor": string,
		"source": string,
		"Fickian": bool,
		"Ead": numeric,
		"Do": numeric,
		"Eas": numeric,
		"So": numeric,
		"Eap": numeric,
		"Po": numeric 
    }

There are also convenience functions to view and search jsons in jupyter notebooks called ``pvdeg.utilities.display_json`` and ``pvdeg.utilities.search_json``.


.. _AApermeation:

AApermeation
~~~~~~~~~~~~
.. literalinclude:: ../../../pvdeg/data/AApermeation.json
    :language: json

.. _H2Opermeation:

H2Opermeation
~~~~~~~~~~~~
.. literalinclude:: ../../../pvdeg/data/H2Opermeation.json
    :language: json

.. _O2permeation:

O2permeation
~~~~~~~~~~~~
.. literalinclude:: ../../../pvdeg/data/O2permeation.json
    :language: json