"""
Test functions in model_builder.
"""

import pytest
import numpy as np
from simpeg.utils.model_builder import create_random_model


class TestDeprecateSeedProperty:
    """
    Test deprecation of seed property.
    """

    def get_message_duplicated_error(self, old_name, new_name, version="v0.24.0"):
        msg = (
            f"Cannot pass both '{new_name}' and '{old_name}'."
            f"'{old_name}' has been deprecated and will be removed in "
            f" SimPEG {version}, please use '{new_name}' instead."
        )
        return msg

    def get_message_deprecated_warning(self, old_name, new_name, version="v0.24.0"):
        msg = (
            f"'{old_name}' has been deprecated and will be removed in "
            f" SimPEG {version}, please use '{new_name}' instead."
        )
        return msg

    @pytest.fixture
    def shape(self):
        return (5, 5)

    def test_warning_argument(self, shape):
        """
        Test if warning is raised after passing ``seed`` as argument.
        """
        msg = self.get_message_deprecated_warning("seed", "random_seed")
        seed = 42135
        with pytest.warns(FutureWarning, match=msg):
            result = create_random_model(shape, seed=seed)
        np.testing.assert_allclose(result, create_random_model(shape, random_seed=seed))

    def test_error_duplicated_argument(self, shape):
        """
        Test error after passing ``seed`` and ``random_seed`` as arguments.
        """
        msg = self.get_message_duplicated_error("seed", "random_seed")
        with pytest.raises(TypeError, match=msg):
            create_random_model(shape, seed=42, random_seed=42)

    def test_error_invalid_kwarg(self, shape):
        """
        Test error after passing invalid kwargs to the function.
        """
        kwargs = {"foo": 1, "bar": 2}
        msg = "Invalid arguments 'foo', 'bar'."
        with pytest.raises(TypeError, match=msg):
            with pytest.warns(FutureWarning):
                create_random_model(shape, seed=10, **kwargs)
