import unittest

import numpy as np

from discretize import TensorMesh

from SimPEG import (
    utils,
)


class DepthWeightingTest(unittest.TestCase):
    def test_depth_weighting_3D(self):
        # Mesh
        dh = 5.0
        hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
        hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
        hz = [(dh, 15)]
        mesh = TensorMesh([hx, hy, hz], "CCN")

        actv = np.random.randint(0, 2, mesh.n_cells) == 1

        r_loc = 0.1
        # Depth weighting
        wz = utils.depth_weighting(mesh, r_loc, indActive=actv, exponent=5, threshold=0)

        reference_locs = (
            np.random.rand(1000, 3) * (mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0))
            + mesh.origin
        )
        reference_locs[:, -1] = r_loc

        wz2 = utils.depth_weighting(
            mesh, reference_locs, indActive=actv, exponent=5, threshold=0
        )
        np.testing.assert_allclose(wz, wz2)

        # testing default params
        all_active = np.ones(mesh.n_cells, dtype=bool)
        wz = utils.depth_weighting(
            mesh, r_loc, indActive=all_active, exponent=2, threshold=0.5 * dh
        )
        wz2 = utils.depth_weighting(mesh, r_loc)

        np.testing.assert_allclose(wz, wz2)

        with self.assertRaises(ValueError):
            wz2 = utils.depth_weighting(mesh, np.random.rand(10, 3, 3))

    def test_depth_weighting_2D(self):
        # Mesh
        dh = 5.0
        hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
        hz = [(dh, 15)]
        mesh = TensorMesh([hx, hz], "CN")

        actv = np.random.randint(0, 2, mesh.n_cells) == 1

        r_loc = 0.1
        # Depth weighting
        wz = utils.depth_weighting(mesh, r_loc, indActive=actv, exponent=5, threshold=0)

        reference_locs = (
            np.random.rand(1000, 2) * (mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0))
            + mesh.origin
        )
        reference_locs[:, -1] = r_loc

        wz2 = utils.depth_weighting(
            mesh, reference_locs, indActive=actv, exponent=5, threshold=0
        )
        np.testing.assert_allclose(wz, wz2)


if __name__ == "__main__":
    unittest.main()
