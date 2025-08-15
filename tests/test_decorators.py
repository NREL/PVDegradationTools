import warnings
from pvdeg import decorators

# Test geospatial_quick_shape decorator


def test_geospatial_quick_shape_numeric():
    @decorators.geospatial_quick_shape("numeric", ["T98", "x_eff"])
    def dummy_func():
        return (1, 2)

    assert hasattr(dummy_func, "numeric_or_timeseries")
    assert hasattr(dummy_func, "shape_names")
    assert dummy_func.numeric_or_timeseries == "numeric"
    assert dummy_func.shape_names == ["T98", "x_eff"]


def test_geospatial_quick_shape_timeseries():
    @decorators.geospatial_quick_shape("timeseries", ["rh", "dry_bulb", "irradiance"])
    def dummy_func():
        return [1, 2, 3]

    assert dummy_func.numeric_or_timeseries == "timeseries"
    assert dummy_func.shape_names == ["rh", "dry_bulb", "irradiance"]


# Test deprecated decorator with reason


def test_deprecated_with_reason():
    @decorators.deprecated("use another function")
    def old_func():
        return 42

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = old_func()
        assert result == 42
        assert any(issubclass(warn.category, DeprecationWarning) for warn in w)
        assert any("use another function" in str(warn.message) for warn in w)


# Test deprecated decorator without reason


def test_deprecated_without_reason():
    @decorators.deprecated
    def old_func():
        return 99

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = old_func()
        assert result == 99
        assert any(issubclass(warn.category, DeprecationWarning) for warn in w)
        assert any("deprecated function" in str(warn.message) for warn in w)


# Test deprecated decorator on class


def test_deprecated_class():
    @decorators.deprecated("class is deprecated")
    class OldClass:
        def __init__(self):
            self.value = 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj = OldClass()
        assert obj.value == 1
        assert any(issubclass(warn.category, DeprecationWarning) for warn in w)
        assert any("class is deprecated" in str(warn.message) for warn in w)
