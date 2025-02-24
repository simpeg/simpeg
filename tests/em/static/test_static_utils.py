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

    def test_warning_argument(self, mesh, points, active_cells):
        """
        Test if warning is raised after passing ``ind_active`` as argument.
        """
        msg = self.get_message_deprecated_warning(self.OLD_NAME, self.NEW_NAME)
        with pytest.warns(FutureWarning, match=msg):
            drapeTopotoLoc(mesh, points, ind_active=active_cells)

    def test_error_duplicated_argument(self, mesh, points, active_cells):
        """
        Test error after passing ``ind_active`` and ``active_cells`` as arguments.
        """
        msg = self.get_message_duplicated_error(self.OLD_NAME, self.NEW_NAME)
        with pytest.raises(TypeError, match=msg):
            drapeTopotoLoc(
                mesh, points, active_cells=active_cells, ind_active=active_cells
            )
