"""
Test the UniformBackgroundField class
"""
import pytest
from SimPEG.potential_fields.magnetics import UniformBackgroundField


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
