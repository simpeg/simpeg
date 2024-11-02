import pytest
import numpy as np

from discretize import TensorMesh

from simpeg import (
    utils,
)


class DepthWeightingTest:
    def test_depth_weighting_3D(self):
        # Mesh
        dh = 5.0
        hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
        hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
        hz = [(dh, 15)]
        mesh = TensorMesh([hx, hy, hz], "CCN")

        rng = np.random.default_rng(seed=42)
        actv = rng.integers(low=0, high=2, size=mesh.n_cells, dtype=bool)

        # Depth weighting
        r_loc = 0.1
        wz = utils.depth_weighting(
            mesh, r_loc, active_cells=actv, exponent=5, threshold=0
        )

        # Define reference locs at random locations
        reference_locs = rng.uniform(
            low=mesh.nodes.min(axis=0), high=mesh.nodes.max(axis=0), size=(1000, 3)
        )
        reference_locs[:, -1] = r_loc  # set them all at the same elevation

        wz2 = utils.depth_weighting(
            mesh, reference_locs, active_cells=actv, exponent=5, threshold=0
        )
        np.testing.assert_allclose(wz, wz2)

        # testing default params
        all_active = np.ones(mesh.n_cells, dtype=bool)
        wz = utils.depth_weighting(
            mesh, r_loc, active_cells=all_active, exponent=2, threshold=0.5 * dh
        )
        wz2 = utils.depth_weighting(mesh, r_loc)

        np.testing.assert_allclose(wz, wz2)

        with self.assertRaises(ValueError):
            wz2 = utils.depth_weighting(mesh, rng.random(size=(10, 3, 3)))

    def test_depth_weighting_2D(self):
        # Mesh
        dh = 5.0
        hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
        hz = [(dh, 15)]
        mesh = TensorMesh([hx, hz], "CN")

        rng = np.random.default_rng(seed=42)
        actv = rng.integers(low=0, high=2, size=mesh.n_cells, dtype=bool)

        r_loc = 0.1
        # Depth weighting
        wz = utils.depth_weighting(
            mesh, r_loc, active_cells=actv, exponent=5, threshold=0
        )

        # Define reference locs at random locations
        reference_locs = rng.uniform(
            low=mesh.nodes.min(axis=0), high=mesh.nodes.max(axis=0), size=(1000, 2)
        )
        reference_locs[:, -1] = r_loc  # set them all at the same elevation

        wz2 = utils.depth_weighting(
            mesh, reference_locs, active_cells=actv, exponent=5, threshold=0
        )
        np.testing.assert_allclose(wz, wz2)


@pytest.fixture
def mesh():
    """Sample mesh."""
    dh = 5.0
    hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
    hz = [(dh, 15)]
    mesh = TensorMesh([hx, hz], "CN")
    return mesh


def test_removed_indactive(mesh):
    """
    Test if error is raised after passing removed indActive argument
    """
    active_cells = np.ones(mesh.nC, dtype=bool)
    msg = "'indActive' argument has been removed. " "Please use 'active_cells' instead."
    with pytest.raises(TypeError, match=msg):
        utils.depth_weighting(mesh, 0, indActive=active_cells)
