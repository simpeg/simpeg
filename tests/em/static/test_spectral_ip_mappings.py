"""
Test ``spectral_ip_mappings`` function.
"""

import pytest
import numpy as np

import discretize
from simpeg.electromagnetics.static.spectral_induced_polarization import (
    spectral_ip_mappings,
)


class TestRemovedIndActive:
    """Test removed ``indActive`` argument ``spectral_ip_mappings``."""

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

    def test_error_argument(self, mesh, active_cells):
        """
        Test if error is raised after passing ``indActive`` as argument.
        """
        msg = "unexpected keyword argument 'indActive'"
        with pytest.raises(TypeError, match=msg):
            spectral_ip_mappings(mesh, indActive=active_cells)
