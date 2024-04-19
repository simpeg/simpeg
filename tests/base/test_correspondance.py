import unittest

import numpy as np

from discretize import TensorMesh
from simpeg import (
    maps,
    regularization,
)

np.random.seed(10)


class LinearCorrespondenceTest(unittest.TestCase):
    def setUp(self):
        dh = 1.0
        nx = 12
        ny = 12

        hx = [(dh, nx)]
        hy = [(dh, ny)]
        mesh = TensorMesh([hx, hy], "CN")

        # reg
        actv = np.ones(len(mesh), dtype=bool)

        # maps
        wires = maps.Wires(("m1", mesh.nC), ("m2", mesh.nC))

        corr = regularization.LinearCorrespondence(
            mesh,
            wire_map=wires,
            active_cells=actv,
        )

        self.mesh = mesh
        self.corr = corr

    def test_order_full_hessian(self):
        """

        Test deriv and deriv2 matrix of linear correspondance with approx_hessian=True

        """
        corr = self.corr
        self.assertTrue(corr._test_deriv())
        self.assertTrue(corr._test_deriv2(expectedOrder=2))

    def test_deriv2_no_arg(self):
        m = np.random.randn(2 * len(self.mesh))

        corr = self.corr

        v = np.random.rand(len(m))

        W = corr.deriv2(m)
        Wv = corr.deriv2(m, v)
        np.testing.assert_allclose(Wv, W @ v)
