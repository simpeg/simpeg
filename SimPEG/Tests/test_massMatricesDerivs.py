import numpy as np
import unittest
from SimPEG import *
from TestUtils import checkDerivative


class TestInnerProductsDerivs(unittest.TestCase):

    def doTestFace(self, h, rep, vec, fast):
        mesh = Mesh.TensorMesh(h)
        v = np.random.rand(mesh.nF)
        def fun(sig):
            M = mesh.getFaceInnerProduct(sig)
            if vec:
                Md = mesh.getFaceInnerProductDeriv(sig, v=v, doFast=fast)
                return M*v, Md
            Md = mesh.getFaceInnerProductDeriv(sig, doFast=fast)
            return M*v, Utils.sdiag(v)*Md
        sig = np.random.rand(mesh.nC*rep)
        return checkDerivative(fun, sig, num=5, plotIt=False)

    def doTestEdge(self, h, rep, vec, fast):
        mesh = Mesh.TensorMesh(h)
        v = np.random.rand(mesh.nE)
        def fun(sig):
            M = mesh.getEdgeInnerProduct(sig)
            if vec:
                Md = mesh.getEdgeInnerProductDeriv(sig, v=v, doFast=fast)
                return M*v, Md
            Md = mesh.getEdgeInnerProductDeriv(sig, doFast=fast)
            return M*v, Utils.sdiag(v)*Md
        sig = np.random.rand(mesh.nC*rep)
        return checkDerivative(fun, sig, num=5, plotIt=False)

    def test_FaceIP_1D_isotropic(self):
        self.assertTrue(self.doTestFace([10],1,True, False))
    def test_FaceIP_2D_isotropic(self):
        self.assertTrue(self.doTestFace([10, 4],1,True, False))
    def test_FaceIP_3D_isotropic(self):
        self.assertTrue(self.doTestFace([10, 4, 5],1,True, False))
    def test_FaceIP_2D_anisotropic(self):
        self.assertTrue(self.doTestFace([10, 4],2,True, False))
    def test_FaceIP_3D_anisotropic(self):
        self.assertTrue(self.doTestFace([10, 4, 5],3,True, False))
    def test_FaceIP_2D_tensor(self):
        self.assertTrue(self.doTestFace([10, 4],3,True, False))
    def test_FaceIP_3D_tensor(self):
        self.assertTrue(self.doTestFace([10, 4, 5],6,True, False))

    def test_FaceIP_1D_isotropic_fast(self):
        self.assertTrue(self.doTestFace([10],1, False, True))
    def test_FaceIP_2D_isotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4],1, False, True))
    def test_FaceIP_3D_isotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4, 5],1, False, True))
    def test_FaceIP_2D_anisotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4],2, False, True))
    def test_FaceIP_3D_anisotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4, 5],3, False, True))


    def test_EdgeIP_2D_isotropic(self):
        self.assertTrue(self.doTestEdge([10, 4],1,True, False))
    def test_EdgeIP_3D_isotropic(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],1,True, False))
    def test_EdgeIP_2D_anisotropic(self):
        self.assertTrue(self.doTestEdge([10, 4],2,True, False))
    def test_EdgeIP_3D_anisotropic(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],3,True, False))
    def test_EdgeIP_2D_tensor(self):
        self.assertTrue(self.doTestEdge([10, 4],3,True, False))
    def test_EdgeIP_3D_tensor(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],6,True, False))

    def test_EdgeIP_2D_isotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4],1, False, True))
    def test_EdgeIP_3D_isotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],1, False, True))
    def test_EdgeIP_2D_anisotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4],2, False, True))
    def test_EdgeIP_3D_anisotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],3, False, True))




if __name__ == '__main__':
    unittest.main()
