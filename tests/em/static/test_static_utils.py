"""
Test functions in ``static_utils``.
"""

import pytest
import numpy as np

import discretize
from simpeg.electromagnetics.static.utils.static_utils import (
    drapeTopotoLoc,
    gettopoCC,
    closestPointsGrid,
)


class TestDeprecatedFunctions:
    """Test deprecated functions."""

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

    def get_message_deprecated_warning(self, old_name, new_name, version="0.26.0"):
        msg = (
            f"{old_name} has been deprecated, please use {new_name}."
            f" It will be removed in version {version} of SimPEG."
        )
        return msg

    def test_drape_topo_warning(self, mesh, points, active_cells):
        """
        Test deprecation warning for `drapeTopotoLoc`.
        """
        old_name = "drapeTopotoLoc"
        new_name = "shift_to_discrete_topography"
        msg = self.get_message_deprecated_warning(old_name, new_name)
        with pytest.warns(FutureWarning, match=msg):
            drapeTopotoLoc(mesh, points, active_cells=active_cells)

    def test_topo_cells_warning(self, mesh, points, active_cells):
        """
        Test deprecation warning for `drapeTopotoLoc`.
        """
        old_name = "gettopoCC"
        new_name = "get_discrete_topography"
        msg = self.get_message_deprecated_warning(old_name, new_name)
        with pytest.warns(FutureWarning, match=msg):
            gettopoCC(mesh, active_cells)

    def test_closest_points(self, mesh, points):
        """
        Test deprecation warning for `closestPointsGrid`.
        """
        msg = "closestPointsGrid is now deprecated. It will be removed in SimPEG version 0.26.0."
        with pytest.warns(FutureWarning, match=msg):
            closestPointsGrid(mesh.cell_centers, points)
