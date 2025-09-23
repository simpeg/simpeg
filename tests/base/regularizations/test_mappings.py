"""
Test mapping functions in regularizations.
"""

import numpy as np
import pytest
from discretize import TensorMesh
from scipy.sparse import diags

from simpeg.maps import IdentityMap, LinearMap, LogMap
from simpeg.regularization import Smallness, SmoothnessFirstOrder


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
    return np.random.default_rng(seed=5959).uniform(size=tensor_mesh.n_cells)


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


@pytest.mark.parametrize(
    "use_reference_model", [True, False], ids=["m_ref", "no m_ref"]
)
class TestMappingInSmoothnessFirstOrder:
    """
    Test mapping in SmoothnessFirstOrder regularization.
    """

    def get_regularization(
        self,
        mesh,
        active_cells,
        reference_model,
        use_reference_model,
        orientation,
        **kwargs,
    ):
        """Return instance of SmoothnessFirstOrder"""
        if use_reference_model:
            reg = SmoothnessFirstOrder(
                mesh=mesh,
                active_cells=active_cells,
                orientation=orientation,
                reference_model=reference_model,
                reference_model_in_smooth=True,
                **kwargs,
            )
        else:
            reg = SmoothnessFirstOrder(
                mesh=mesh,
                active_cells=active_cells,
                orientation=orientation,
                **kwargs,
            )
        return reg

    @pytest.mark.parametrize("orientation", ["x", "y", "z"])
    def test_default_mapping(
        self,
        tensor_mesh,
        active_cells,
        model,
        reference_model,
        use_reference_model,
        orientation,
    ):
        """
        Test regularization using the default (identity) mapping.
        """
        reg = self.get_regularization(
            mesh=tensor_mesh,
            active_cells=active_cells,
            orientation=orientation,
            reference_model=reference_model,
            use_reference_model=use_reference_model,
        )

        # Test call
        gradients = getattr(reg.regularization_mesh, f"cell_gradient_{orientation}")
        model_diff = model - reference_model if use_reference_model else model
        r = reg.W @ gradients @ model_diff
        expected = r.T @ r
        np.testing.assert_allclose(reg(model), expected)

        # Test deriv
        expected_gradient = 2 * gradients.T @ reg.W.T @ reg.W @ gradients @ model_diff
        np.testing.assert_allclose(reg.deriv(model), expected_gradient, atol=1e-10)

    @pytest.mark.parametrize("orientation", ["x", "y", "z"])
    def test_linear_mapping(
        self,
        tensor_mesh,
        active_cells,
        model,
        reference_model,
        use_reference_model,
        orientation,
    ):
        """
        Test regularization using a linear mapping.
        """
        n_active_cells = active_cells.sum()
        a = np.full(n_active_cells, 3.5)
        a_matrix = diags(a)
        linear_mapping = LinearMap(a_matrix, b=None)
        reg = self.get_regularization(
            mesh=tensor_mesh,
            active_cells=active_cells,
            orientation=orientation,
            reference_model=reference_model,
            use_reference_model=use_reference_model,
            mapping=linear_mapping,
        )
        assert reg.mapping is linear_mapping

        # Test call
        gradients = getattr(reg.regularization_mesh, f"cell_gradient_{orientation}")
        model_diff = (
            a * model - a * reference_model if use_reference_model else a * model
        )
        r = reg.W @ gradients @ model_diff
        expected = r.T @ r
        np.testing.assert_allclose(reg(model), expected)

        # Test deriv
        f_m_deriv = gradients @ a_matrix
        expected_gradient = 2 * f_m_deriv.T @ reg.W.T @ reg.W @ gradients @ model_diff
        np.testing.assert_allclose(reg.deriv(model), expected_gradient, atol=1e-10)

    @pytest.mark.parametrize("orientation", ["x", "y", "z"])
    def test_nonlinear_mapping(
        self,
        tensor_mesh,
        active_cells,
        model,
        reference_model,
        use_reference_model,
        orientation,
    ):
        """
        Test regularization using a non-linear mapping.
        """
        log_mapping = LogMap()
        reg = self.get_regularization(
            mesh=tensor_mesh,
            active_cells=active_cells,
            orientation=orientation,
            reference_model=reference_model,
            use_reference_model=use_reference_model,
            mapping=log_mapping,
        )
        assert reg.mapping is log_mapping

        # Test call
        gradients = getattr(reg.regularization_mesh, f"cell_gradient_{orientation}")
        model_diff = (
            np.log(model) - np.log(reference_model)
            if use_reference_model
            else np.log(model)
        )
        r = reg.W @ gradients @ model_diff
        expected = r.T @ r
        np.testing.assert_allclose(reg(model), expected)

        # Test deriv
        f_m_deriv = gradients @ diags(1 / model)
        expected_gradient = 2 * f_m_deriv.T @ reg.W.T @ reg.W @ gradients @ model_diff
        np.testing.assert_allclose(reg.deriv(model), expected_gradient, atol=1e-10)
