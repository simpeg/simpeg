from __future__ import print_function
import unittest
import numpy as np
from SimPEG import utils

tol = 1e-15


class coorUtilsTest(unittest.TestCase):
    def test_rotationMatrixFromNormals(self):
        np.random.seed(0)
        v0 = np.random.rand(3)
        v0 *= 1.0 / np.linalg.norm(v0)

        np.random.seed(5)
        v1 = np.random.rand(3)
        v1 *= 1.0 / np.linalg.norm(v1)

        Rf = utils.coord_utils.rotationMatrixFromNormals(v0, v1)
        Ri = utils.coord_utils.rotationMatrixFromNormals(v1, v0)

        self.assertTrue(np.linalg.norm(utils.mkvc(Rf.dot(v0) - v1)) < tol)
        self.assertTrue(np.linalg.norm(utils.mkvc(Ri.dot(v1) - v0)) < tol)

    def test_rotatePointsFromNormals(self):
        np.random.seed(10)
        v0 = np.random.rand(3)
        v0 *= 1.0 / np.linalg.norm(v0)

        np.random.seed(15)
        v1 = np.random.rand(3)
        v1 *= 1.0 / np.linalg.norm(v1)

        v2 = utils.mkvc(
            utils.coord_utils.rotatePointsFromNormals(utils.mkvc(v0, 2).T, v0, v1)
        )

        self.assertTrue(np.linalg.norm(v2 - v1) < tol)

    def test_rotateMatrixFromNormals(self):
        np.random.seed(20)
        n0 = np.random.rand(3)
        n0 *= 1.0 / np.linalg.norm(n0)

        np.random.seed(25)
        n1 = np.random.rand(3)
        n1 *= 1.0 / np.linalg.norm(n1)

        np.random.seed(30)
        scale = np.random.rand(100, 1)
        XYZ0 = scale * n0
        XYZ1 = scale * n1

        XYZ2 = utils.coord_utils.rotatePointsFromNormals(XYZ0, n0, n1)
        self.assertTrue(
            np.linalg.norm(utils.mkvc(XYZ1) - utils.mkvc(XYZ2))
            / np.linalg.norm(utils.mkvc(XYZ1))
            < tol
        )


if __name__ == "__main__":
    unittest.main()
