"""
Test functions in model_builder.
"""

import re
import pytest
import numpy as np
import discretize
from simpeg.utils.model_builder import create_random_model, get_indices_block


class TestRemovalSeedProperty:
    """
    Test removed seed property.
    """

    @pytest.fixture
    def shape(self):
        return (5, 5)

    def test_error_argument(self, shape):
        """
        Test if error is raised after passing ``seed`` as argument.
        """
        msg = "Invalid arguments 'seed'"
        seed = 42135
        with pytest.raises(TypeError, match=msg):
            create_random_model(shape, seed=seed)

    def test_error_invalid_kwarg(self, shape):
        """
        Test error after passing invalid kwargs to the function.
        """
        kwargs = {"foo": 1, "bar": 2}
        msg = "Invalid arguments 'foo', 'bar'."
        with pytest.raises(TypeError, match=msg):
            create_random_model(shape, **kwargs)


class TestGetIndicesBlock:
    """
    Test the ``get_indices_block`` function.
    """

    block_cells_per_dim = 2

    def get_mesh_and_block_corners(self, ndims):

        p0_template = [-4, 2, -10]
        if ndims == 1:
            origin = "C"
            p0 = np.array(p0_template[:1])
        elif ndims == 2:
            origin = "CC"
            p0 = np.array(p0_template[:2])
        else:
            origin = "CCN"
            p0 = np.array(p0_template)

        cell_size = 2.0
        hx = [(cell_size, 10)]
        h = [hx for _ in range(ndims)]
        mesh = discretize.TensorMesh(h, origin=origin)

        p1 = np.array([c + self.block_cells_per_dim * cell_size for c in p0])
        return (mesh, p0, p1)

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_get_indices_block(self, ndim):
        """Test the funciton returns the right indices."""
        mesh, p0, p1 = self.get_mesh_and_block_corners(ndim)
        indices = get_indices_block(p0, p1, mesh.cell_centers)
        assert len(indices) == self.block_cells_per_dim**ndim

    def test_invalid_p0_p1(self):
        # Dummy mesh (not really used)
        hx = [(2.0, 10)]
        mesh = discretize.TensorMesh([hx])

        # Define block corners with different amount of elements
        p0 = np.array([1.0, 2.0, 3.0, 4.0])
        p1 = np.array([10.0, 11.0, 12.0])

        msg = re.escape("Dimension mismatch between `p0` and `p1`.")
        with pytest.raises(ValueError, match=msg):
            get_indices_block(p0, p1, mesh.cell_centers)

    def test_invalid_mesh_dimensions(self):
        # Define a 2d mesh
        hx = [(2.0, 10)]
        mesh = discretize.TensorMesh([hx, hx])

        # Define block corners in 3d
        p0 = np.array([1.0, 2.0, 3.0])
        p1 = np.array([2.0, 3.0, 4.0])

        msg = re.escape(
            "Dimension mismatch between `cell_centers` and dimensions of "
            "block corners."
        )
        with pytest.raises(ValueError, match=msg):
            get_indices_block(p0, p1, mesh.cell_centers)
