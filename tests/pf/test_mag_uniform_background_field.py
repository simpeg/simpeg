"""
Test the UniformBackgroundField class
"""

import pytest
from simpeg.potential_fields.magnetics import UniformBackgroundField, SourceField


def test_invalid_parameters_argument():
    """
    Test if error is raised after passing 'parameters' as argument
    """
    parameters = (1, 35, 60)
    msg = (
        "'parameters' property has been removed."
        "Please pass the amplitude, inclination and declination"
        " through their own arguments."
    )
    with pytest.raises(TypeError, match=msg):
        UniformBackgroundField(parameters=parameters)


def test_deprecated_source_field():
    """
    Test if instantiating a magnetics.source.SourceField object raises an error
    """
    msg = "SourceField has been removed, please use UniformBackgroundField."
    with pytest.raises(NotImplementedError, match=msg):
        SourceField()
