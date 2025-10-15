"""
Test functions in ``static_utils``.
"""

import pytest
import numpy as np

import discretize
from simpeg.electromagnetics.static.utils.static_utils import drapeTopotoLoc


class TestDeprecatedIndActive:
    """Test deprecated ``ind_active`` argument ``drapeTopotoLoc``."""

    OLD_NAME = "ind_active"
    NEW_NAME = "active_cells"

    @pytest.fixture
    def mesh(self):
        """Sample mesh."""
        return discretize.TensorMesh([10, 10, 10], "CCN")

    @pytest.fixture
    def points(self):
        """Sample points."""
        return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    @pytest.fixture
    def active_cells(self, mesh):
        """Sample active cells for the mesh."""
        active_cells = np.ones(mesh.n_cells, dtype=bool)
        active_cells[0] = False
        return active_cells

    def test_error_argument(self, mesh, points, active_cells):
        """
        Test if error is raised after passing ``ind_active`` as argument.
        """
        msg = "Unsupported keyword argument ind_active"
        with pytest.raises(TypeError, match=msg):
            drapeTopotoLoc(mesh, points, ind_active=active_cells)
