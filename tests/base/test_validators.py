import numpy as np
import pytest

from SimPEG.utils import (
    validate_string,
    validate_integer,
    validate_float,
    validate_list_of_types,
    validate_location_property,
    validate_ndarray_with_shape,
    validate_type,
    validate_callable,
    validate_direction,
)


def test_string_validation():
    # validate property is a string
    assert validate_string("string_prop", "Hello") == "Hello"

    # in list case sensitive
    assert (
        validate_string("string_prop", "Hello", ["Hello", "World"], case_sensitive=True)
        == "Hello"
    )

    # in list case insensistive (with expected return case in list)
    assert (
        validate_string(
            "string_prop", "Hello", ["hello", "world"], case_sensitive=False
        )
        == "hello"
    )

    # aliased case names
    assert (
        validate_string(
            "string_prop",
            "Hi",
            [
                ("hello", "hi", "howdy"),
            ],
        )
        == "hello"
    )

    # not a string:
    with pytest.raises(TypeError):
        validate_string("string_prop", 4.0)

    # not in the list case sensitive
    with pytest.raises(ValueError):
        validate_string("string_prop", "Hello", ["hello", "world"], case_sensitive=True)

    # not in the list case insensitive
    with pytest.raises(ValueError):
        validate_string("string_prop", "Hello", ["howdy", "earth"])

    # not in the list or aliases case insensitive
    with pytest.raises(ValueError):
        validate_string("string_prop", "Hello", [("howdy", "hi"), "world"])


def test_integer_validation():

    # valid integer
    assert validate_integer("int_prop", -4) == -4
    # float to integer
    assert validate_integer("int_prop", -4.0) == -4
    # float to integer (with discarded decimal)
    assert validate_integer("int_prop", -5.23) == -5
    # string to integer
    assert validate_integer("int_prop", "-4") == -4

    # valid minimum and maximum
    assert validate_integer("int_prop", -4, -10, 6) == -4

    # invalid string to convert
    with pytest.raises(TypeError):
        validate_integer("int_prop", "Pi")

    # outside range
    with pytest.raises(ValueError):
        assert validate_integer("int_prop", -21, -10, 6)


def test_float_validation():

    # These should pass
    assert validate_float("FloatProperty", -4.0) == -4.0  # float
    assert validate_float("FloatProperty", -4) == -4.0  # int converted to float
    assert validate_float("FloatProperty", "-4.0") == -4.0  # str converted to float
    assert validate_float("FloatProperty", -4, -4, 6) == -4.0  # with min and max
    assert validate_float("FloatProperty", -4, -5, 6, inclusive_min=False) == -4.0
    assert validate_float("FloatProperty", 6, -4, 6) == 6  # with min and max
    assert validate_float("FloatProperty", 6, -4, 7, inclusive_max=False) == 6

    # Invalid string to float
    with pytest.raises(TypeError):
        validate_float("FloatProperty", "Hello")
    # can't convert complex to float
    with pytest.raises(TypeError):
        validate_float("FloatProperty", -4 + 6j)
    # outside inclusive range
    with pytest.raises(ValueError):
        validate_float("FloatProperty", -4, 0, 100)
    # at exclusive min
    with pytest.raises(ValueError):
        validate_float("FloatProperty", -4, -4, 100, inclusive_min=False)
    # outside inclusive range
    with pytest.raises(ValueError):
        validate_float("FloatProperty", 100, 0, 100, inclusive_max=False)


def test_list_validation():

    # Empty list should work
    assert validate_list_of_types("ListProperty", [], object) == []  # empty list

    # any object should work
    assert validate_list_of_types("ListProperty", ["Hello", 6, 45.0], object) == [
        "Hello",
        6,
        45.0,
    ]

    # multiple accepted types
    assert validate_list_of_types(
        "ListProperty", [6, 45.0, 6 + 2j], (int, float, complex)
    ) == [6, 45.0, 6 + 2j]

    # convert single object to length 1 list
    assert validate_list_of_types("ListProperty", -4.0, float) == [-4.0]

    # should allow multiple of the same object by default
    assert validate_list_of_types("ListProperty", ["Hello", "Hello", "Hello"], str) == [
        "Hello",
        "Hello",
        "Hello",
    ]

    # item is incorrect type
    with pytest.raises(TypeError):
        validate_list_of_types("ListProperty", 4, float)
    # some items are incorrect type
    with pytest.raises(TypeError):
        validate_list_of_types("ListProperty", [4, 4.0], float)

    # some items are repeated but should be unique
    with pytest.raises(ValueError):
        validate_list_of_types(
            "ListProperty", ["Hello", "Hello", "Hello"], str, ensure_unique=True
        )


def test_location_validation():

    # simple valid location
    first_test = validate_location_property("LocationProperty", np.r_[1, 2, 3])
    # check return type is a float numpy array
    assert np.issubdtype(first_test.dtype, float)
    np.testing.assert_equal(first_test, np.r_[1.0, 2.0, 3.0])

    # column array to 1D array
    assert validate_location_property("LocationProperty", np.c_[1, 2, 3]).shape == (3,)
    # will squeeze out all length one axes
    assert validate_location_property(
        "LocationProperty", np.array([[[1], [2], [3]]])
    ).shape == (3,)

    # will convert array_like
    assert isinstance(
        validate_location_property("LocationProperty", [1, 2, 3]), np.ndarray
    )

    # Has a specific dimension
    np.testing.assert_equal(
        validate_location_property("LocationProperty", np.r_[1, 2, 3], 3),
        np.r_[1.0, 2.0, 3.0],
    )

    # incorrect dimension
    with pytest.raises(ValueError):
        validate_location_property(
            "LocationProperty",
            np.r_[1, 2, 3],
            2,
        )
    # too many axes not length 1
    with pytest.raises(ValueError):
        validate_location_property(
            "LocationProperty",
            np.random.rand(3, 1, 2),
        )

    # incorrect type of input
    with pytest.raises(TypeError):
        validate_location_property(
            "LocationProperty",
            ["a", "b", "c"],
        )


def test_ndarray_validation():

    # should convert anything to an ndarray with None dtype
    # and no shape specified
    assert isinstance(
        validate_ndarray_with_shape("array_prop", "sdsa", dtype=None), np.ndarray
    )

    # should convert to a specified type
    out = validate_ndarray_with_shape("array_prop", ["3", "4", "5"], dtype=float)
    assert np.issubdtype(out.dtype, float)
    np.testing.assert_equal(out, np.array([3.0, 4.0, 5.0]))

    # Valid any shaped arrays
    assert validate_ndarray_with_shape(
        "NDarrayProperty", np.random.rand(3, 3, 3), ("*", "*", "*"), float
    ).shape == (3, 3, 3)

    # Valid any shaped arrays with a restriction on an axis
    assert validate_ndarray_with_shape(
        "NDarrayProperty", np.random.rand(3, 2, 3), ("*", 2, "*"), float
    ).shape == (3, 2, 3)

    # valid array with specific shape.
    assert validate_ndarray_with_shape(
        "NDarrayProperty", np.random.rand(3, 2, 1), (3, 2, 1), float
    ).shape == (3, 2, 1)

    # valid array with that could be multiple shapes
    assert validate_ndarray_with_shape(
        "NDarrayProperty", np.random.rand(3, 2), [(4, 2), (3, 2)], float
    ).shape == (3, 2)
    assert validate_ndarray_with_shape(
        "NDarrayProperty", np.random.rand(4, 2), [(4, 2), (3, 2)], float
    ).shape == (4, 2)

    # should do np.atleast_1d, _2d, _3d.
    assert validate_ndarray_with_shape("NDarrayProperty", 1.0, (1,)).shape == (1,)
    assert validate_ndarray_with_shape(
        "NDarrayProperty", np.random.rand(3), (1, 3)
    ).shape == (1, 3)
    assert validate_ndarray_with_shape(
        "NDarrayProperty", np.random.rand(3), (1, 3, 1)
    ).shape == (1, 3, 1)
    assert validate_ndarray_with_shape(
        "NDarrayProperty", np.random.rand(4, 3), (4, 3, 1)
    ).shape == (4, 3, 1)

    # improper type
    with pytest.raises(TypeError):
        validate_ndarray_with_shape("NDarrayProperty", ["a", "b"], ("*",))
    # a shape is more than 3D
    with pytest.raises(NotImplementedError):
        validate_ndarray_with_shape(
            "NDarrayProperty", np.random.rand(3, 3, 3, 3), ("*", "*", "*", "*")
        )

    # incorrect exact shape
    with pytest.raises(ValueError):
        validate_ndarray_with_shape("NDarrayProperty", np.random.rand(4, 3), (4, 2))

    # incorrect loose shape
    with pytest.raises(ValueError):
        validate_ndarray_with_shape("NDarrayProperty", np.random.rand(4, 3), ("*", 2))

    # input has more dimensions than expected
    # (will not discard length 1 dimensions)
    with pytest.raises(ValueError):
        validate_ndarray_with_shape(
            "NDarrayProperty", np.random.rand(4, 3, 1), ("*", 3)
        )

    # Invalid shape from multiple shapes
    with pytest.raises(ValueError):
        validate_ndarray_with_shape(
            "NDarrayProperty",
            np.random.rand(
                4,
                3,
            ),
            [(3, 2), (4, 2)],
        )


def test_type_validation():

    # should try to cast to type
    assert type(validate_type("type_prop", 4.0, int)) == int

    # isinstance without casting should pass through
    assert type(validate_type("type_prop", 4.0, object, cast=False)) == float

    # should return object without casting if isinstance
    assert type(validate_type("type_prop", True, int, cast=False)) == bool

    # strict type checking
    assert type(validate_type("type_prop", 4.0, float, strict=True)) == float

    # should error if unable to cast to type
    with pytest.raises(TypeError):
        validate_type("type_prop", "four", float)

    # should error if strict and not an exact same class
    with pytest.raises(TypeError):
        validate_type("type_prop", True, int, cast=False, strict=True)

    # should error if not strict and not subclass
    with pytest.raises(TypeError):
        validate_type("type_prop", True, float, cast=False)


def test_callable_validation():
    def func(x):
        return x * x

    # a simple callable function
    assert validate_callable("callable_prop", func) is func

    # other things are callable too, like class constructors
    assert validate_callable("callable_prop", np.ndarray) is np.ndarray

    # some things are not callable
    with pytest.raises(TypeError):
        validate_callable("callable_prop", [1, 2, 3])
    with pytest.raises(TypeError):
        validate_callable("callable_prop", "what?")
    with pytest.raises(TypeError):
        validate_callable("callable_prop", ...)


def test_direction_validation():
    # should convert "x", "y", and "z" to specific arrays
    np.testing.assert_equal(validate_direction("orient", "x"), np.r_[1.0, 0.0, 0.0])
    np.testing.assert_equal(validate_direction("orient", "y"), np.r_[0.0, 1.0, 0.0])
    np.testing.assert_equal(validate_direction("orient", "z"), np.r_[0.0, 0.0, 1.0])

    # should pass an normalized array through
    np.testing.assert_equal(
        validate_direction("orient", np.r_[0.0, 0.0, 1.0]), np.r_[0.0, 0.0, 1.0]
    )

    # should normalize and return arbitrary array
    dir = [1, 2, 3]
    # should pass an normalized array through
    np.testing.assert_equal(
        validate_direction("orient", dir), dir / np.linalg.norm(dir)
    )

    # dimensions should be respected for defaults
    np.testing.assert_equal(validate_direction("orient", "x", dim=1), np.r_[1.0])
    np.testing.assert_equal(validate_direction("orient", "x", dim=2), np.r_[1.0, 0.0])
    np.testing.assert_equal(validate_direction("orient", "y", dim=2), np.r_[0.0, 1.0])
    dir = [
        2.0,
    ]
    np.testing.assert_equal(
        validate_direction("orient", dir, dim=1), dir / np.linalg.norm(dir)
    )

    dir = [2.0, 1.0]
    np.testing.assert_equal(
        validate_direction("orient", dir, dim=2), dir / np.linalg.norm(dir)
    )

    # should error on incorrect dimension and passed array
    with pytest.raises(ValueError):
        validate_direction("orient", [1, 2, 3], dim=2)
    # should error on incorrect shape
    with pytest.raises(ValueError):
        validate_direction("orient", [[1, 2, 3]], dim=3)
    # should error on incorrect string
    with pytest.raises(ValueError):
        validate_direction("orient", "xy")
    # should error on incorrect data_type
    with pytest.raises(TypeError):
        validate_direction("orient", ["x", "y", "z"])
    # should error on incorrect string for dimension
    with pytest.raises(ValueError):
        validate_direction("orient", "z", dim=2)
