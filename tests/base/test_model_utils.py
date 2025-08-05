import numpy as np
import pytest
from discretize import TensorMesh

from simpeg import utils


class TestDepthWeighting:
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

        with pytest.raises(ValueError):
            utils.depth_weighting(mesh, rng.random(size=(10, 3, 3)))

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


class TestDistancehWeighting:
    def test_distance_weighting_3D(self):
        # Mesh
        dh = 5.0
        hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
        hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
        hz = [(dh, 15)]
        mesh = TensorMesh([hx, hy, hz], "CCN")

        rng = np.random.default_rng(seed=42)
        actv = rng.integers(low=0, high=2, size=mesh.n_cells, dtype=bool)

        # Define reference locs at random locations
        reference_locs = rng.uniform(
            low=mesh.nodes.min(axis=0), high=mesh.nodes.max(axis=0), size=(1000, 3)
        )

        # distance weighting
        with pytest.warns():
            wz_scipy = utils.distance_weighting(
                mesh, reference_locs, active_cells=actv, exponent=3, engine="scipy"
            )
        wz_numba = utils.distance_weighting(
            mesh, reference_locs, active_cells=actv, exponent=3, engine="numba"
        )
        np.testing.assert_allclose(wz_scipy, wz_numba)

        with pytest.raises(ValueError):
            utils.distance_weighting(
                mesh, reference_locs, active_cells=actv, exponent=3, engine="test"
            )

    def test_distance_weighting_2D(self):
        # Mesh
        dh = 5.0
        hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
        hz = [(dh, 15)]
        mesh = TensorMesh([hx, hz], "CN")

        rng = np.random.default_rng(seed=42)
        actv = rng.integers(low=0, high=2, size=mesh.n_cells, dtype=bool)

        # Define reference locs at random locations
        reference_locs = rng.uniform(
            low=mesh.nodes.min(axis=0), high=mesh.nodes.max(axis=0), size=(1000, 2)
        )

        # distance weighting
        with pytest.warns():
            wz_scipy = utils.distance_weighting(
                mesh, reference_locs, active_cells=actv, exponent=3, engine="scipy"
            )
        wz_numba = utils.distance_weighting(
            mesh, reference_locs, active_cells=actv, exponent=3, engine="numba"
        )
        np.testing.assert_allclose(wz_scipy, wz_numba)

    def test_distance_weighting_1D(self):
        # Mesh
        dh = 5.0
        hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
        mesh = TensorMesh([hx], "C")

        rng = np.random.default_rng(seed=42)
        actv = rng.integers(low=0, high=2, size=mesh.n_cells, dtype=bool)

        # Define reference locs at random locations
        reference_locs = rng.uniform(
            low=mesh.nodes.min(axis=0), high=mesh.nodes.max(axis=0), size=(1000, 1)
        )

        # distance weighting
        with pytest.warns():
            wz_scipy = utils.distance_weighting(
                mesh, reference_locs, active_cells=actv, exponent=3, engine="scipy"
            )
        wz_numba = utils.distance_weighting(
            mesh, reference_locs, active_cells=actv, exponent=3, engine="numba"
        )
        np.testing.assert_allclose(wz_scipy, wz_numba)

    @pytest.mark.parametrize("ndim", (2, 3))
    def test_invalid_reference_locs(self, ndim):
        """
        Test if errors are raised when invalid reference_locs are passed.
        """
        hx = [5.0, 10]
        h = [hx] * ndim
        origin = "CCN" if ndim == 3 else "CC"
        reference_locs = [1.0, 2.0] if ndim == 3 else [1.0]
        mesh = TensorMesh(h, origin)
        with pytest.raises(ValueError):
            utils.distance_weighting(mesh, reference_locs)

    def test_numba_and_cdist_opts_error(self):
        """Test error when passing numba and cdist_opts."""
        hx = [5.0, 10]
        mesh = TensorMesh([hx, hx, hx])
        msg = "The `cdist_opts` is valid only when engine is 'scipy'."
        with pytest.raises(TypeError, match=msg):
            utils.distance_weighting(mesh, [1.0, 2.0, 3.0], cdist_opts={"foo": "bar"})


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
