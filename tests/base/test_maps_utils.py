"""
Test utility functions for maps.
"""

import pytest
import numpy as np

from simpeg.maps._utils import ColumnStackedMatrix


class TestColumnStackedMatrix:
    """Test the ``ColumnStackedMatrix`` class."""

    def test_error_empty_columns(self):
        """Test error when passing empty columns."""
        with pytest.raises(ValueError, match="Invalid empty columns"):
            ColumnStackedMatrix(columns=[])

    def test_error_not_1D_arrays(self):
        """Test error when passing not 1D arrays as columns."""
        columns = [
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0, 2.0, 3.0]]),
        ]
        with pytest.raises(ValueError, match="dimensions"):
            ColumnStackedMatrix(columns=columns)

    def test_error_mismatching_sizes(self):
        """Test error when passing column arrays with different sizes."""
        columns = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
        ]
        with pytest.raises(
            ValueError, match="Invalid column arrays with different sizes"
        ):
            ColumnStackedMatrix(columns=columns)

    def test_shape(self):
        """Test the shape attribute."""
        columns = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
        ]
        matrix = ColumnStackedMatrix(columns)
        assert matrix.shape == (3, 2)

    @pytest.fixture
    def shape(self):
        """Sample shape for following tests."""
        return (5, 8)

    @pytest.fixture
    def columns(self, shape):
        """Sample columns for following tests."""
        rng = np.random.default_rng(seed=41412)
        n_rows, n_cols = shape
        columns = [rng.uniform(low=-1.0, high=1.0, size=n_rows) for _ in range(n_cols)]
        return columns

    def test_matvec(self, shape, columns):
        """Test dot product with vector."""
        matrix = ColumnStackedMatrix(columns)
        true_matrix = np.c_[*columns]

        rng = np.random.default_rng(seed=42)
        vector = rng.uniform(low=-1.0, high=1.0, size=shape[1])
        np.testing.assert_allclose(matrix @ vector, true_matrix @ vector)

    def test_rmatvec(self, shape, columns):
        """Test dot product of transpose with vector."""
        matrix = ColumnStackedMatrix(columns)
        true_matrix = np.c_[*columns]

        rng = np.random.default_rng(seed=42)
        vector = rng.uniform(low=-1.0, high=1.0, size=shape[0])
        np.testing.assert_allclose(matrix.T @ vector, true_matrix.T @ vector)
