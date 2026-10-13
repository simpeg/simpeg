import re
import pytest
import unittest

import numpy as np

from discretize import TensorMesh, TreeMesh
from simpeg import (
    maps,
    regularization,
)

from simpeg.regularization.cross_gradient import _cross_product_2d


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


class TestCrossProduct2D:
    """
    Test the ``_cross_product_2d`` private function.

    .. important::

        This function was added after Numpy dropped support for 2D vectors in the
        :func:`numpy.cross` function.
    """

    def test_single_vectors(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = _cross_product_2d(a, b)

        a = np.array([1.0, 2.0, 0.0])
        b = np.array([3.0, 4.0, 0.0])
        _, _, expected = np.cross(a, b)

        assert result.ndim == 1
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("a_x_b", [True, False], ids=["a x b", "b x a"])
    def test_single_and_multiple_vectors(self, a_x_b):
        a = np.array([1.0, 2.0])
        b = np.array([[3.0, 4.0], [5.0, 6.0]])
        result = _cross_product_2d(a, b) if a_x_b else _cross_product_2d(b, a)

        a = np.array([1.0, 2.0, 0.0])
        b = np.array([[3.0, 4.0, 0.0], [5.0, 6.0, 0.0]])
        cross = np.cross(a, b) if a_x_b else np.cross(b, a)
        expected = cross[:, -1]

        assert result.ndim == 1
        np.testing.assert_allclose(result, expected)

    def test_multiple_vectors(self):
        a = np.array([[1.0, 2.0], [8.0, 9.0]])
        b = np.array([[3.0, 4.0], [5.0, 6.0]])
        result = _cross_product_2d(a, b)

        a = np.array([[1.0, 2.0, 0.0], [8.0, 9.0, 0.0]])
        b = np.array([[3.0, 4.0, 0.0], [5.0, 6.0, 0.0]])
        expected = np.cross(a, b)[:, -1]

        assert result.ndim == 1
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("a_x_b", [True, False], ids=["a x b", "b x a"])
    @pytest.mark.parametrize(
        "a",
        [
            np.array([1.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0]]),
            np.array([[1.0], [2.0]]),
            np.array([[1.0, 2.0, 3.0]]),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ],
    )
    def test_invalid_elements(self, a, a_x_b):
        b = np.array([1.0, 2.0])
        msg = re.escape(
            f"Invalid array '{'a' if a_x_b else 'b'}' containing vectors with"
        )
        if a_x_b:
            with pytest.raises(ValueError, match=msg):
                _cross_product_2d(a, b)
        else:
            with pytest.raises(ValueError, match=msg):
                _cross_product_2d(b, a)

    @pytest.mark.parametrize("a_x_b", [True, False], ids=["a x b", "b x a"])
    @pytest.mark.parametrize(
        "a",
        [np.array(1.0), np.arange(6).reshape(3, 2, 1)],
    )
    def test_invalid_dimensions(self, a, a_x_b):
        b = np.array([1.0, 2.0])
        msg = (
            re.escape(f"Invalid array '{'a' if a_x_b else 'b'}' with '")
            + "[0-9]+"
            + re.escape("' dimensions")
        )
        if a_x_b:
            with pytest.raises(ValueError, match=msg):
                _cross_product_2d(a, b)
        else:
            with pytest.raises(ValueError, match=msg):
                _cross_product_2d(b, a)


if __name__ == "__main__":
    unittest.main()
