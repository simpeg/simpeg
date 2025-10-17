"""
Test functions in ``static_utils``.
"""

import re
import pytest
import numpy as np

import discretize
from simpeg.electromagnetics.static.utils.static_utils import (
    drapeTopotoLoc,
    gettopoCC,
    closestPointsGrid,
)


@pytest.fixture
def mesh():
    """Sample mesh."""
    return discretize.TensorMesh([10, 10, 10], "CCN")


@pytest.fixture
def points():
    """Sample points."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def active_cells(mesh):
    """Sample active cells for the mesh."""
    active_cells = np.ones(mesh.n_cells, dtype=bool)
    active_cells[0] = False
    return active_cells


class TestDeprecatedIndActive:
    """Test deprecated ``ind_active`` argument ``drapeTopotoLoc``."""

    OLD_NAME = "ind_active"
    NEW_NAME = "active_cells"

    @pytest.mark.filterwarnings("ignore:The `drapeTopotoLoc` function is deprecated")
    def test_error_argument(self, mesh, points, active_cells):
        """
        Test if error is raised after passing ``ind_active`` as argument.
        """
        msg = "Unsupported keyword argument ind_active"
        with pytest.raises(TypeError, match=msg):
            drapeTopotoLoc(mesh, points, ind_active=active_cells)


class TestDeprecatedFunctions:
    """Test deprecated functions."""

    def test_drape_topo_warning(self, mesh, points, active_cells):
        """
        Test deprecation warning for `drapeTopotoLoc`.
        """
        msg = re.escape(
            "The `drapeTopotoLoc` function is deprecated, "
            "and will be removed in SimPEG v0.27.0. "
            "This functionality has been replaced by the "
            "'shift_to_discrete_topography' function, which can be imported from"
            "simpeg.utils"
        )
        with pytest.warns(FutureWarning, match=msg):
            drapeTopotoLoc(mesh, points, active_cells=active_cells)

    def test_topo_cells_warning(self, mesh, active_cells):
        """
        Test deprecation warning for `drapeTopotoLoc`.
        """
        msg = re.escape(
            "The `gettopoCC` function is deprecated, "
            "and will be removed in SimPEG v0.27.0. "
            "This functionality has been replaced by the "
            "'get_discrete_topography' function, which can be imported from"
            "simpeg.utils"
        )
        with pytest.warns(FutureWarning, match=msg):
            gettopoCC(mesh, active_cells)

    def test_closest_points(self, mesh, points):
        """
        Test deprecation warning for `closestPointsGrid`.
        """
        msg = re.escape("The `closestPointsGrid` function is now deprecated.")
        with pytest.warns(FutureWarning, match=msg):
            closestPointsGrid(mesh.cell_centers, points)
