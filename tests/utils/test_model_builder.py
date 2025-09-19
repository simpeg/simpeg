"""
Test functions in model_builder.
"""

import pytest
from simpeg.utils.model_builder import create_random_model


class TestRemovalSeedProperty:
    """
    Test removed seed property.
    """

    @pytest.fixture
    def shape(self):
        return (5, 5)

    def test_error_argument(self, shape):
        """
        Test if error is raised after passing ``seed`` as argument.
        """
        msg = "Invalid arguments 'seed'"
        seed = 42135
        with pytest.raises(TypeError, match=msg):
            create_random_model(shape, seed=seed)

    def test_error_invalid_kwarg(self, shape):
        """
        Test error after passing invalid kwargs to the function.
        """
        kwargs = {"foo": 1, "bar": 2}
        msg = "Invalid arguments 'foo', 'bar'."
        with pytest.raises(TypeError, match=msg):
            create_random_model(shape, **kwargs)
