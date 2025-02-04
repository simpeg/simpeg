"""
Test mapping functions in regularizations.
"""

import numpy as np
import pytest
from discretize import TensorMesh
from scipy.sparse import diags

from simpeg.maps import IdentityMap, LinearMap, LogMap
from simpeg.regularization import Smallness


@pytest.fixture
def tensor_mesh():
    hx = [(2.0, 10)]
    h = [hx, hx, hx]
    return TensorMesh(h)


@pytest.fixture
def active_cells(tensor_mesh):
    return np.ones(tensor_mesh.n_cells, dtype=bool)


@pytest.fixture
def model(tensor_mesh):
    n = tensor_mesh.n_cells
    return np.linspace(1.0, 51.0, n)


@pytest.fixture
def reference_model(tensor_mesh):
    return np.full(tensor_mesh.n_cells, fill_value=2.0)


class TestMappingInSmallness:
    """
    Test mapping in Smallness regularization.
    """

    def test_default_mapping(self, tensor_mesh, active_cells, model, reference_model):
        """
        Test regularization using the default (identity) mapping.
        """
        reg = Smallness(
            mesh=tensor_mesh, active_cells=active_cells, reference_model=reference_model
        )
        assert isinstance(reg.mapping, IdentityMap)
        volume_weights = tensor_mesh.cell_volumes
        expected = np.sum(volume_weights * (model - reference_model) ** 2)
        np.testing.assert_allclose(reg(model), expected)
        expected_gradient = 2 * volume_weights * (model - reference_model)
        np.testing.assert_allclose(reg.deriv(model), expected_gradient)

    def test_linear_mapping(self, tensor_mesh, active_cells, model, reference_model):
        """
        Test regularization using a linear mapping.
        """
        n_active_cells = active_cells.sum()
        a = np.full(n_active_cells, 3.5)
        a_matrix = diags(a)
        linear_mapping = LinearMap(a_matrix, b=None)
        reg = Smallness(
            mesh=tensor_mesh,
            active_cells=active_cells,
            mapping=linear_mapping,
            reference_model=reference_model,
        )
        assert reg.mapping is linear_mapping
        volume_weights = tensor_mesh.cell_volumes
        expected = np.sum(volume_weights * (a * model - a * reference_model) ** 2)
        np.testing.assert_allclose(reg(model), expected)
        expected_gradient = 2 * a * volume_weights * (a * model - a * reference_model)
        np.testing.assert_allclose(reg.deriv(model), expected_gradient)

    def test_nonlinear_mapping(self, tensor_mesh, active_cells, model, reference_model):
        """
        Test regularization using a non-linear mapping.
        """
        log_mapping = LogMap()
        reg = Smallness(
            mesh=tensor_mesh,
            active_cells=active_cells,
            mapping=log_mapping,
            reference_model=reference_model,
        )
        assert reg.mapping is log_mapping
        volume_weights = tensor_mesh.cell_volumes
        expected = np.sum(
            volume_weights * (np.log(model) - np.log(reference_model)) ** 2
        )
        np.testing.assert_allclose(reg(model), expected)
        expected_gradient = (
            2 * (1 / model) * volume_weights * (np.log(model) - np.log(reference_model))
        )
        np.testing.assert_allclose(reg.deriv(model), expected_gradient)
