"""
Test the UniformBackgroundField class
"""

import pytest
import numpy as np
from simpeg.potential_fields.magnetics import UniformBackgroundField, SourceField, Point


def test_invalid_parameters_argument():
    """
    Test if error is raised after passing 'parameters' as argument
    """
    parameters = (1, 35, 60)
    msg = r"UniformBackgroundField.__init__\(\) got an unexpected keyword argument 'parameters'"
    with pytest.raises(TypeError, match=msg):
        UniformBackgroundField(parameters=parameters)


def test_deprecated_source_field():
    """
    Test if instantiating a magnetics.source.SourceField object raises an error
    """
    msg = "SourceField has been removed, please use UniformBackgroundField."
    with pytest.raises(NotImplementedError, match=msg):
        SourceField()


@pytest.mark.parametrize("receiver_as_list", (True, False))
def test_invalid_receiver_type(receiver_as_list):
    """
    Test if error is raised after passing invalid type of receivers
    """
    receiver_invalid = np.array([[1.0, 1.0, 1.0]])
    if receiver_as_list:
        receiver_valid = Point(locations=np.array([[0.0, 0.0, 0.0]]), components="tmi")
        receiver_list = [receiver_valid, receiver_invalid]
    else:
        receiver_list = receiver_invalid
    msg = f"'receiver_list' must be a list of {Point}"
    with pytest.raises(TypeError, match=msg):
        UniformBackgroundField(
            receiver_list=receiver_list,
            amplitude=55_000,
            inclination=45,
            declination=30,
        )


@pytest.mark.parametrize(
    "receiver_list",
    (None, [Point(locations=np.array([[0.0, 0.0, 0.0]]), components="tmi")]),
    ids=("None", "Point"),
)
def test_value_b0(receiver_list):
    """
    Test UniformBackgroundField.b0 value
    """
    amplitude = 55_000
    inclination = 45
    declination = 10
    expected_b0 = (6753.3292182935065, 38300.03321760104, -38890.87296526011)
    uniform_background_field = UniformBackgroundField(
        receiver_list=receiver_list,
        amplitude=amplitude,
        inclination=inclination,
        declination=declination,
    )
    np.testing.assert_allclose(uniform_background_field.b0, expected_b0)
