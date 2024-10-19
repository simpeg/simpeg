"""
Test ``spectral_ip_mappings`` function.
"""

import pytest
import numpy as np

import discretize
from simpeg.electromagnetics.static.spectral_induced_polarization import (
    spectral_ip_mappings,
)


class TestDeprecatedIndActive:
    """Test deprecated ``indActive`` argument ``spectral_ip_mappings``."""

    OLD_NAME = "indActive"
    NEW_NAME = "active_cells"

    @pytest.fixture
    def mesh(self):
        """Sample mesh."""
        return discretize.TensorMesh([10, 10, 10], "CCN")

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

    def test_warning_argument(self, mesh, active_cells):
        """
        Test if warning is raised after passing ``indActive`` as argument.
        """
        msg = self.get_message_deprecated_warning(self.OLD_NAME, self.NEW_NAME)
        with pytest.warns(FutureWarning, match=msg):
            spectral_ip_mappings(mesh, indActive=active_cells)

    def test_error_duplicated_argument(self, mesh, active_cells):
        """
        Test error after passing ``indActive`` and ``active_cells`` as arguments.
        """
        msg = self.get_message_duplicated_error(self.OLD_NAME, self.NEW_NAME)
        with pytest.raises(TypeError, match=msg):
            spectral_ip_mappings(
                mesh, active_cells=active_cells, indActive=active_cells
            )
