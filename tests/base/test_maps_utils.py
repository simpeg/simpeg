"""
Test private map utility functions.
"""

import pytest
import numpy as np

from simpeg.maps._utils import columns_dot_vector


class TestColumnsDotVector:
    """Test the ``columns_dot_vector`` function."""

    n_cols = 5
    n_rows = 3

    @pytest.fixture
    def vector(self):
        """Sample vector for following tests."""
        rng = np.random.default_rng(seed=4141)
        return rng.uniform(low=-1.0, high=1.0, size=self.n_cols)

    @pytest.fixture(params=["all-1d", "some-2d"])
    def columns(self, request):
        """Sample columns for following tests."""
        rng = np.random.default_rng(seed=42)
        match request.param:
            case "all-1d":
                columns = [
                    rng.uniform(low=-1.0, high=1.0, size=self.n_rows)
                    for _ in range(self.n_cols)
                ]
            case "some-2d":
                columns = [
                    rng.uniform(low=-1.0, high=1.0, size=self.n_rows),
                    rng.uniform(low=-1.0, high=1.0, size=(self.n_rows, 3)),
                    rng.uniform(low=-1.0, high=1.0, size=self.n_rows),
                ]
            case _:
                raise ValueError()
        return columns

    def test_error_empty_columns(self, vector):
        with pytest.raises(ValueError, match="Invalid empty 'columns' argument"):
            columns_dot_vector([], vector)

    def test_error_invalid_dimension(self, vector):
        columns = [
            np.array([1.0, 2.0, 3.0]),
            np.array([[[1.0, 2.0, 4.0]]]),
        ]
        with pytest.raises(ValueError, match="dimensions"):
            columns_dot_vector(columns, vector)

    def test_error_invalid_number_of_columns(self, vector):
        # Create a list with only 2 columns
        columns = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        ]
        with pytest.raises(ValueError, match="not equal to the size of the vector"):
            columns_dot_vector(columns, vector)

    @pytest.mark.parametrize("columns_type", ["list", "generator"])
    def test_dot_product(self, columns, vector, columns_type):
        """
        Compare the dot product against the expected result using the full matrix.
        """
        true_matrix = np.c_[*columns]
        if columns_type == "generator":
            columns = (c for c in columns)
        np.testing.assert_allclose(
            columns_dot_vector(columns, vector), true_matrix @ vector
        )
