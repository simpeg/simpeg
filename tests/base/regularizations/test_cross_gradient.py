import re
import pytest
import unittest

import numpy as np

from discretize import TensorMesh, TreeMesh
from simpeg import (
    maps,
    regularization,
)

from simpeg.regularization.cross_gradient import _expand_2d_vector


class CrossGradientTensor2D(unittest.TestCase):
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

        cros_grad = regularization.CrossGradient(
            mesh,
            wire_map=wires,
            active_cells=actv,
        )

        self.mesh = mesh
        self.cross_grad = cros_grad

    def test_order_approximate_hessian(self):
        """

        Test deriv and deriv2 matrix of cross-gradient with approx_hessian=True

        """
        cross_grad = self.cross_grad
        cross_grad.approx_hessian = True
        self.assertTrue(cross_grad.test(random_seed=10))

    def test_order_full_hessian(self):
        """

        Test deriv and deriv2 matrix of cross-gradient with approx_hessian=True

        """
        cross_grad = self.cross_grad
        cross_grad.approx_hessian = False
        self.assertTrue(cross_grad._test_deriv(random_seed=10))
        self.assertTrue(cross_grad._test_deriv2(random_seed=10, expectedOrder=2))

    def test_deriv2_no_arg(self):
        np.random.seed(10)
        m = np.random.randn(2 * len(self.mesh))

        cross_grad = self.cross_grad

        v = np.random.rand(len(m))

        cross_grad.approx_hessian = False
        W = cross_grad.deriv2(m)
        Wv = cross_grad.deriv2(m, v)
        np.testing.assert_allclose(Wv, W @ v)

        cross_grad.approx_hessian = True
        W = cross_grad.deriv2(m)
        Wv = cross_grad.deriv2(m, v)
        np.testing.assert_allclose(Wv, W @ v)

    def test_cross_grad_calc(self):
        mesh = self.mesh
        # get index of center point
        midx = int(mesh.shape_cells[0] / 2)
        midy = int(mesh.shape_cells[1] / 2)

        # create a model1
        m1 = np.zeros(mesh.shape_cells)
        m1[(midx - 3) : (midx + 3), (midy - 3) : (midy + 3)] = 1

        # create a model2
        m2 = np.zeros(mesh.shape_cells)
        m2[(midx - 5) : (midx + 1), (midy - 5) : (midy + 1)] = 1

        m1 = m1.reshape(-1, order="F")
        m2 = m2.reshape(-1, order="F")

        # stack the true models
        m = np.r_[m1, m2]

        cross_grad = self.cross_grad

        v1 = np.sum(np.abs(cross_grad.calculate_cross_gradient(m)))
        v2 = cross_grad(m)
        self.assertEqual(v1, v2)


class CrossGradientTensor3D(unittest.TestCase):
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

        cros_grad = regularization.CrossGradient(
            mesh,
            wire_map=wires,
            active_cells=actv,
        )

        self.mesh = mesh
        self.cross_grad = cros_grad

    def test_order_approximate_hessian(self):
        """

        Test deriv and deriv2 matrix of cross-gradient with approx_hessian=True

        """
        cross_grad = self.cross_grad
        cross_grad.approx_hessian = True
        self.assertTrue(cross_grad.test(random_seed=10))

    def test_order_full_hessian(self):
        """

        Test deriv and deriv2 matrix of cross-gradient with approx_hessian=True

        """
        cross_grad = self.cross_grad
        cross_grad.approx_hessian = False
        self.assertTrue(cross_grad._test_deriv(random_seed=10))
        self.assertTrue(cross_grad._test_deriv2(random_seed=10, expectedOrder=2))

    def test_deriv2_no_arg(self):
        np.random.seed(10)
        m = np.random.randn(2 * len(self.mesh))

        cross_grad = self.cross_grad

        v = np.random.rand(len(m))

        cross_grad.approx_hessian = False
        W = cross_grad.deriv2(m)
        Wv = cross_grad.deriv2(m, v)
        np.testing.assert_allclose(Wv, W @ v)

        cross_grad.approx_hessian = True
        W = cross_grad.deriv2(m)
        Wv = cross_grad.deriv2(m, v)
        np.testing.assert_allclose(Wv, W @ v)

    def test_cross_grad_calc(self):
        np.random.seed(10)
        m = np.random.randn(2 * len(self.mesh))
        cross_grad = self.cross_grad

        m1, m2 = cross_grad.wire_map * m
        v1 = cross_grad._calculate_gradient(m1)
        v2 = cross_grad._calculate_gradient(m1, normalized=True)
        np.testing.assert_allclose(v1 / np.linalg.norm(v1, axis=-1)[:, None], v2)

        # test that calling the calculate_cross_gradient function works in 3D
        cross_grad.calculate_cross_gradient(m)


class CrossGradientTree2D(unittest.TestCase):
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

        cross_grad = regularization.CrossGradient(
            mesh, wire_map=wires, active_cells=actv
        )

        self.mesh = mesh
        self.cross_grad = cross_grad

    def test_order_approximate_hessian(self):
        """

        Test deriv and deriv2 matrix of cross-gradient with approx_hessian=True

        """
        cross_grad = self.cross_grad
        cross_grad.approx_hessian = True
        self.assertTrue(cross_grad.test(random_seed=10))

    def test_order_full_hessian(self):
        """

        Test deriv and deriv2 matrix of cross-gradient with approx_hessian=True

        """
        cross_grad = self.cross_grad
        cross_grad.approx_hessian = False
        self.assertTrue(cross_grad._test_deriv(random_seed=10))
        self.assertTrue(cross_grad._test_deriv2(random_seed=10, expectedOrder=2))

    def test_deriv2_no_arg(self):
        np.random.seed(10)
        m = np.random.randn(2 * len(self.mesh))

        cross_grad = self.cross_grad

        v = np.random.rand(len(m))

        cross_grad.approx_hessian = False
        W = cross_grad.deriv2(m)
        Wv = cross_grad.deriv2(m, v)
        np.testing.assert_allclose(Wv, W @ v)

        cross_grad.approx_hessian = True
        W = cross_grad.deriv2(m)
        Wv = cross_grad.deriv2(m, v)
        np.testing.assert_allclose(Wv, W @ v)


class CrossGradientTree3D(unittest.TestCase):
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

        cross_grad = regularization.CrossGradient(
            mesh, wire_map=wires, active_cells=actv
        )

        self.mesh = mesh
        self.cross_grad = cross_grad

    def test_order_approximate_hessian(self):
        """

        Test deriv and deriv2 matrix of cross-gradient with approx_hessian=True

        """
        cross_grad = self.cross_grad
        cross_grad.approx_hessian = True
        self.assertTrue(cross_grad.test(random_seed=10))

    def test_order_full_hessian(self):
        """

        Test deriv and deriv2 matrix of cross-gradient with approx_hessian=True

        """
        cross_grad = self.cross_grad
        cross_grad.approx_hessian = False
        self.assertTrue(cross_grad._test_deriv(random_seed=10))
        self.assertTrue(cross_grad._test_deriv2(random_seed=10, expectedOrder=2))

    def test_deriv2_no_arg(self):
        np.random.seed(10)
        m = np.random.randn(2 * len(self.mesh))

        cross_grad = self.cross_grad

        v = np.random.rand(len(m))

        cross_grad.approx_hessian = False
        W = cross_grad.deriv2(m)
        Wv = cross_grad.deriv2(m, v)
        np.testing.assert_allclose(Wv, W @ v)

        cross_grad.approx_hessian = True
        W = cross_grad.deriv2(m)
        Wv = cross_grad.deriv2(m, v)
        np.testing.assert_allclose(Wv, W @ v)


class TestExpand2DVector:
    """
    Test the ``_expand_2d_vector`` private function.

    .. important::

        This function was added to expand 2D vectors intended to be passed to
        :func:`numpy.cross` after Numpy removed support for 2D vectors.
    """

    def test_1dim(self):
        """Test passing a 1D array with a single vector with 2 elements."""
        a = np.array([1.0, 2.0])
        result = _expand_2d_vector(a)
        expected = np.array([1.0, 2.0, 0.0])
        np.testing.assert_allclose(expected, result, strict=True)

    @pytest.mark.parametrize(
        "a",
        [
            np.array([1]),
            np.array([1, 2, 3]),
            np.array([[1]]),
            np.array([[1, 2, 3]]),
            np.array([[1], [2]]),
            np.array([[1, 2, 3], [4, 5, 6]]),
        ],
    )
    def test_error_elements(self, a):
        """
        Test error if the passed vector(s) has not 2 elements.
        """
        msg = re.escape("Invalid array containing vectors with")
        with pytest.raises(ValueError, match=msg):
            _expand_2d_vector(a)

    @pytest.mark.parametrize(
        ("a", "expected"),
        [
            (np.array([[1.0, 2.0]]), np.array([[1.0, 2.0, 0.0]])),
            (
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]),
            ),
        ],
        ids=["single", "multi"],
    )
    def test_2dim(self, a, expected):
        """Test passing a 2D array with vector(s) with 2 elements."""
        result = _expand_2d_vector(a)
        np.testing.assert_allclose(expected, result, strict=True)

    def test_error_ndim(self):
        """Test error if array has more than 2 dimensions."""
        a = np.arange(6).reshape(3, 2, 1)
        msg = re.escape("Invalid array with '3' dimensions.")
        with pytest.raises(ValueError, match=msg):
            _expand_2d_vector(a)


if __name__ == "__main__":
    unittest.main()
