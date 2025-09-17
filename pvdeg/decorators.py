"""Utility Decorators for PVDeg.

Private API, should only be used in PVDeg implemenation files.
"""

import functools
import inspect
import warnings


def geospatial_quick_shape(numeric_or_timeseries: str, shape_names: list[str]) -> None:
    """Add an attribute to the functions that can be run with geospatial analysis.

    Strict typing is not enough for this purpose so we can view this attribute at
    runtime to create a template for the function.

    For single numeric results, includes tabular numeric data
    >>> value = 'numeric'

    Example if a function returns a dataframe with 1 row of numerics (not timeseries)
    `pvdeg.standards.standoff` does this.

    For timeseries results
    >>> value = 'timeseries'

    Example, `pvdeg.temperature.temperature`

    For both numeric and timeseries results, we care about the output names of the
    funtion. When a function returns a dataframe, the names will simply be the dataframe
    column names.
    >>> return df # function returns dataframe
    >>> df.columns = ["rh", "dry_bulb", "irradiance"] # dataframe column names
    >>> func.shape_names = ["rh", "dry_bulb", "irradiance"] # function attribute names

    When a function returns a numeric, or tuple of numerics, the names will correspond
    to the meanings of each unpacked variable.
    >>> return (T98, x_eff) # function return tuple of numerics
    >>> func.shape_names = ["T98", "x_eff"] # function attribute names

    * Note: we cannot autotemplate functions with ambiguous return types that depend on
    runtime input,
    the function will need to strictly return a timeseries or numeric.

    * Note: this is accessed through the ``decorators.geospatial_quick_shape`` namespace

    Parameters
    ----------
    numeric_or_timeseries: bool
        indicate whether the function returns a single numeric/tuple of numerics
        or a timeseries/tuple of timeseries. False when numeric, True when timeseries

    shape_names: list[str]
        list of return value names. These will become the xarray datavariable names in
        the output.

    Modifies
    --------
    func.numeric_or_timeseries
        sets to numeric_or_timeseries argument
    func.shape_names
        sets to shape_names argument
    """

    def decorator(func):
        setattr(func, "numeric_or_timeseries", numeric_or_timeseries)
        setattr(func, "shape_names", shape_names)
        return func

    return decorator


# Taken from: https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically  # noqa
# A future Python version (after 3.13) will include the warnings.deprecated decorator
def deprecated(reason):
    """Warn user of deprecated functions.

    Decorator function to mark functions as deprecated. Returns a warning
    when the deprecated function is used.
    """
    string_types = (type(b""), type(""))

    if isinstance(reason, string_types):
        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):
            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter("always", DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                warnings.simplefilter("default", DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):
        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))
