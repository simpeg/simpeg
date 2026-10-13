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
        """Test error when passing not 1D or 2D arrays as columns."""
        columns = [
            np.array([1.0, 2.0, 3.0]),
            np.array([[[1.0, 2.0, 3.0]]]),
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
            ValueError, match="Invalid column arrays with incompatible sizes"
        ):
            ColumnStackedMatrix(columns=columns)
        columns = [
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0], [2.0], [3.0], [4.0]]),
        ]

    def test_shape(self):
        """Test the shape attribute."""
        columns = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
        ]
        matrix = ColumnStackedMatrix(columns)
        assert matrix.shape == (3, 2)
        columns = [
            np.array([1.0, 2.0, 3.0]),
            np.array(
                [
                    [10.0, 20.0],
                    [30.0, 40.0],
                    [50.0, 60.0],
                ]
            ),
            np.array([1.0, 2.0, 3.0]),
        ]
        matrix = ColumnStackedMatrix(columns)
        assert matrix.shape == (3, 4)

    @pytest.fixture(params=["full-1d", "with-2d"])
    def columns(self, request):
        """Sample columns for following tests."""
        rng = np.random.default_rng(seed=41412)
        n_rows = 5
        match request.param:
            case "full-1d":
                n_cols = 8
                columns = [
                    rng.uniform(low=-1.0, high=1.0, size=n_rows) for _ in range(n_cols)
                ]
            case "with-2d":
                columns = [
                    rng.uniform(low=-1.0, high=1.0, size=n_rows),
                    rng.uniform(low=-1.0, high=1.0, size=(n_rows, 3)),
                    rng.uniform(low=-1.0, high=1.0, size=n_rows),
                ]
            case _:  # pragma: nocover
                raise ValueError()
        return columns

    def test_matvec(self, columns):
        """Test dot product with vector."""
        matrix = ColumnStackedMatrix(columns)
        true_matrix = np.c_[*columns]

        rng = np.random.default_rng(seed=42)
        vector = rng.uniform(low=-1.0, high=1.0, size=matrix.shape[1])
        np.testing.assert_allclose(matrix @ vector, true_matrix @ vector)

    def test_rmatvec(self, columns):
        """Test dot product of transpose with vector."""
        matrix = ColumnStackedMatrix(columns)
        true_matrix = np.c_[*columns]

        rng = np.random.default_rng(seed=42)
        vector = rng.uniform(low=-1.0, high=1.0, size=matrix.shape[0])
        np.testing.assert_allclose(matrix.T @ vector, true_matrix.T @ vector)
