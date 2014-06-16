import numpy as np
import unittest
from SimPEG import *
from TestUtils import checkDerivative


class TestInnerProductsDerivs(unittest.TestCase):

    def doTestFace(self, h, rep, fast):
        mesh = Mesh.TensorMesh(h)
        v = np.random.rand(mesh.nF)
        sig = np.random.rand(1) if rep is 0 else np.random.rand(mesh.nC*rep)
        Md = mesh.getFaceInnerProductDeriv(Utils.TensorType(mesh, sig), doFast=fast)
        def fun(sig):
            M = mesh.getFaceInnerProduct(sig)
            return M*v, Md*v
        return checkDerivative(fun, sig, num=5, plotIt=False)

    def doTestEdge(self, h, rep, fast):
        mesh = Mesh.TensorMesh(h)
        v = np.random.rand(mesh.nE)
        sig = np.random.rand(1) if rep is 0 else np.random.rand(mesh.nC*rep)
        Md = mesh.getEdgeInnerProductDeriv(Utils.TensorType(mesh, sig), doFast=fast)
        def fun(sig):
            M = mesh.getEdgeInnerProduct(sig)
            return M*v, Md*v
        return checkDerivative(fun, sig, num=5, plotIt=False)

    def test_FaceIP_1D_float(self):
        self.assertTrue(self.doTestFace([10],0, False))
    def test_FaceIP_2D_float(self):
        self.assertTrue(self.doTestFace([10, 4],0, False))
    def test_FaceIP_3D_float(self):
        self.assertTrue(self.doTestFace([10, 4, 5],0, False))
    def test_FaceIP_1D_isotropic(self):
        self.assertTrue(self.doTestFace([10],1, False))
    def test_FaceIP_2D_isotropic(self):
        self.assertTrue(self.doTestFace([10, 4],1, False))
    def test_FaceIP_3D_isotropic(self):
        self.assertTrue(self.doTestFace([10, 4, 5],1, False))
    def test_FaceIP_2D_anisotropic(self):
        self.assertTrue(self.doTestFace([10, 4],2, False))
    def test_FaceIP_3D_anisotropic(self):
        self.assertTrue(self.doTestFace([10, 4, 5],3, False))
    def test_FaceIP_2D_tensor(self):
        self.assertTrue(self.doTestFace([10, 4],3, False))
    def test_FaceIP_3D_tensor(self):
        self.assertTrue(self.doTestFace([10, 4, 5],6, False))

    def test_FaceIP_1D_float_fast(self):
        self.assertTrue(self.doTestFace([10],0, True))
    def test_FaceIP_2D_float_fast(self):
        self.assertTrue(self.doTestFace([10, 4],0, True))
    def test_FaceIP_3D_float_fast(self):
        self.assertTrue(self.doTestFace([10, 4, 5],0, True))
    def test_FaceIP_1D_isotropic_fast(self):
        self.assertTrue(self.doTestFace([10],1, True))
    def test_FaceIP_2D_isotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4],1, True))
    def test_FaceIP_3D_isotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4, 5],1, True))
    def test_FaceIP_2D_anisotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4],2, True))
    def test_FaceIP_3D_anisotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4, 5],3, True))

    def test_EdgeIP_1D_float(self):
        self.assertTrue(self.doTestEdge([10],0, False))
    def test_EdgeIP_2D_float(self):
        self.assertTrue(self.doTestEdge([10, 4],0, False))
    def test_EdgeIP_3D_float(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],0, False))
    def test_EdgeIP_1D_isotropic(self):
        self.assertTrue(self.doTestEdge([10],1, False))
    def test_EdgeIP_2D_isotropic(self):
        self.assertTrue(self.doTestEdge([10, 4],1, False))
    def test_EdgeIP_3D_isotropic(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],1, False))
    def test_EdgeIP_2D_anisotropic(self):
        self.assertTrue(self.doTestEdge([10, 4],2, False))
    def test_EdgeIP_3D_anisotropic(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],3, False))
    def test_EdgeIP_2D_tensor(self):
        self.assertTrue(self.doTestEdge([10, 4],3, False))
    def test_EdgeIP_3D_tensor(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],6, False))

    def test_EdgeIP_1D_float_fast(self):
        self.assertTrue(self.doTestEdge([10],0, True))
    def test_EdgeIP_2D_float_fast(self):
        self.assertTrue(self.doTestEdge([10, 4],0, True))
    def test_EdgeIP_3D_float_fast(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],0, True))
    def test_EdgeIP_1D_isotropic_fast(self):
        self.assertTrue(self.doTestEdge([10],1, True))
    def test_EdgeIP_2D_isotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4],1, True))
    def test_EdgeIP_3D_isotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],1, True))
    def test_EdgeIP_2D_anisotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4],2, True))
    def test_EdgeIP_3D_anisotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],3, True))


if __name__ == '__main__':
    unittest.main()
