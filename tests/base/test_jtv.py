import unittest
import pytest

import numpy as np

from discretize import TensorMesh, TreeMesh
from SimPEG import (
    maps,
    regularization,
)

np.random.seed(10)


class JTVTensor2D(unittest.TestCase):
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
        wires = maps.Wires(("m1", mesh.nC), ("m2", mesh.nC), ("m3", mesh.nC))

        jtv = regularization.JointTotalVariation(
            mesh,
            wire_map=wires,
            active_cells=actv,
        )

        self.mesh = mesh
        self.jtv = jtv
        self.x0 = np.random.rand(len(mesh) * len(wires.maps))

    def test_order_full_hessian(self):
        """

        Test deriv and deriv2 matrix of cross-gradient with approx_hessian=True

        """
        jtv = self.jtv
        self.assertTrue(jtv._test_deriv(self.x0))
        self.assertTrue(jtv._test_deriv2(self.x0, expectedOrder=2))

    def test_deriv2_no_arg(self):
        m = self.x0

        jtv = self.jtv

        v = np.random.rand(len(m))

        W = jtv.deriv2(m)
        Wv = jtv.deriv2(m, v)
        np.testing.assert_allclose(Wv, W @ v)


class JTVTensor3D(unittest.TestCase):
    def setUp(self):
        dh = 1.0
        nx = 12
        ny = 12
        nz = 12

        hx = [(dh, nx)]
        hy = [(dh, ny)]
        hz = [(dh, nz)]
        mesh = TensorMesh([hx, hy, hz], "CNN")

        # reg
        actv = np.ones(len(mesh), dtype=bool)

        # maps
        wires = maps.Wires(("m1", mesh.nC), ("m2", mesh.nC))

        jtv = regularization.JointTotalVariation(
            mesh,
            wire_map=wires,
            active_cells=actv,
        )

        self.mesh = mesh
        self.jtv = jtv
        self.x0 = np.random.rand(len(mesh) * 2)

    def test_order_full_hessian(self):
        """

        Test deriv and deriv2 matrix of cross-gradient with approx_hessian=True

        """
        jtv = self.jtv
        self.assertTrue(jtv._test_deriv(self.x0))
        self.assertTrue(jtv._test_deriv2(self.x0, expectedOrder=2))

    def test_deriv2_no_arg(self):
        m = self.x0

        jtv = self.jtv

        v = np.random.rand(len(m))

        W = jtv.deriv2(m)
        Wv = jtv.deriv2(m, v)
        np.testing.assert_allclose(Wv, W @ v)


class JTVTree2D(unittest.TestCase):
    def setUp(self):
        dh = 1.0
        nx = 16
        ny = 16

        hx = [(dh, nx)]
        hy = [(dh, ny)]
        mesh = TreeMesh([hx, hy], "CN")
        mesh.insert_cells([5, 5], [4])

        # reg
        actv = np.ones(len(mesh), dtype=bool)

        # maps
        wires = maps.Wires(("m1", mesh.nC), ("m2", mesh.nC))

        jtv = regularization.JointTotalVariation(
            mesh, wire_map=wires, active_cells=actv
        )

        self.mesh = mesh
        self.jtv = jtv
        self.x0 = np.random.rand(len(mesh) * 2)

    def test_order_full_hessian(self):
        """

        Test deriv and deriv2 matrix of cross-gradient with approx_hessian=True

        """
        jtv = self.jtv
        self.assertTrue(jtv._test_deriv(self.x0))
        self.assertTrue(jtv._test_deriv2(self.x0, expectedOrder=2))

    def test_deriv2_no_arg(self):
        m = self.x0

        jtv = self.jtv

        v = np.random.rand(len(m))

        W = jtv.deriv2(m)
        Wv = jtv.deriv2(m, v)
        np.testing.assert_allclose(Wv, W @ v)


class JTVTree3D(unittest.TestCase):
    def setUp(self):
        dh = 1.0
        nx = 16
        ny = 16
        nz = 16

        hx = [(dh, nx)]
        hy = [(dh, ny)]
        hz = [(dh, nz)]
        mesh = TreeMesh([hx, hy, hz], "CNN")
        mesh.insert_cells([5, 5, 5], [4])

        # reg
        actv = np.ones(len(mesh), dtype=bool)

        # maps
        wires = maps.Wires(("m1", mesh.nC), ("m2", mesh.nC))

        jtv = regularization.JointTotalVariation(
            mesh, wire_map=wires, active_cells=actv
        )

        self.mesh = mesh
        self.jtv = jtv
        self.x0 = np.random.rand(len(mesh) * 2)

    def test_order_full_hessian(self):
        """

        Test deriv and deriv2 matrix of cross-gradient with approx_hessian=True

        """
        jtv = self.jtv
        self.assertTrue(jtv._test_deriv(self.x0))
        self.assertTrue(jtv._test_deriv2(self.x0, expectedOrder=2))

    def test_deriv2_no_arg(self):
        m = self.x0

        jtv = self.jtv

        v = np.random.rand(len(m))

        W = jtv.deriv2(m)
        Wv = jtv.deriv2(m, v)
        np.testing.assert_allclose(Wv, W @ v)


def test_bad_wires():
    dh = 1.0
    nx = 12
    ny = 12

    hx = [(dh, nx)]
    hy = [(dh, ny)]
    mesh = TensorMesh([hx, hy], "CN")

    # reg
    actv = np.ones(len(mesh), dtype=bool)

    # maps
    wires = maps.Wires(("m1", mesh.nC), ("m2", mesh.nC - 2), ("m3", mesh.nC - 3))

    with pytest.raises(ValueError):
        regularization.JointTotalVariation(
            mesh,
            wire_map=wires,
            active_cells=actv,
        )


if __name__ == "__main__":
    unittest.main()
