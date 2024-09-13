.. _geospatial-templates:

Geospatial Templates
===================
Using 3 dimensional labeled arrays (`Xarray`) we are able to run calculations using meteorological data across many points at once. This process has been parallelized using `dask` and `xarray`. Both of these packages can be run locally or on cloud HPC environments. 

This presents a new issue, our models produce outputs in many different shapes and sizes. We can have single numerical results, multiple numeric results or a timeseries of numeric results at each location. To parallelize this process, we cannot wait until runtime to know what shape to store the outputs in. This is where the need for `templates` arises.

Previously, ``pvdeg.geospatial`` provided minimal templates and forced users to create their own for each function they wanted to use in a geospatial calculation.

Auto-templating: allows users to skip creating templates for most functions within pvdeg by using ``pvdeg.geospatial.autotemplate`` to generate templates on the spot, instead of figuring out the output shape. For any given function within the source code decorated with `geospatial_result_type`, we can use `pvdeg.geospatial.autotemplate`


Example
--------

Here we are providing a function to autotemplate along with an ``Xarray.Dataset`` of weather data. Combining these two will give us enough information to produce an output template.

Autotemplate approach to creating a template

.. code-block:: Python

    edge_seal_template = pvdeg.geospatial.auto_template(
        func=pvdeg.design.edge_seal_width,
        ds_gids=geo_weather
    )

Manual Approach to Creating the Sample Template

.. code-block:: Python

    shapes = {
        "width" : ("gid",) # one return value at each datapoint, only dependent on datapoint, not time
    }

    template = pvdeg.geospatial.output_template(
        ds_gids=geo_weather, # xarray dataset 
        shapes=shapes, # output shapes
    )

