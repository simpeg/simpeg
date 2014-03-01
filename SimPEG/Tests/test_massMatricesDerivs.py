import numpy as np
import unittest
from SimPEG import *
from TestUtils import checkDerivative


class TestInnerProductsDerivs(unittest.TestCase):
    def setUp(self):
        pass

    def test_FaceIP_derivs_isotropic(self):
        for d in range(3):
            mesh = Mesh.TensorMesh([10,5,4][d:])
            v = np.random.rand(mesh.nF)
            def fun(sig):
                M = mesh.getFaceInnerProduct(sig)
                Md = mesh.getFaceInnerProductDeriv(sig)
                return M*v, Utils.sdiag(v)*Md
            sig = np.random.rand(mesh.nC)
            passed = checkDerivative(fun, sig, plotIt=False)
            self.assertTrue(passed)


    def test_EdgeIP_derivs_isotropic(self):
        for h in [[10,5],[10,5,4]]:
            mesh = Mesh.TensorMesh(h)
            v = np.random.rand(mesh.nE)
            def fun(sig):
                M = mesh.getEdgeInnerProduct(sig)
                Md = mesh.getEdgeInnerProductDeriv(sig)
                return M*v, Utils.sdiag(v)*Md
            sig = np.random.rand(mesh.nC)
            passed = checkDerivative(fun, sig, plotIt=False)
            self.assertTrue(passed)

    def test_FaceIP_derivs_anisotropic(self):
        for d in range(3):
            mesh = Mesh.TensorMesh([10,5,4][d:])
            v = np.random.rand(mesh.nF)
            def fun(sig):
                M = mesh.getFaceInnerProduct(sig)
                Md = mesh.getFaceInnerProductDeriv(sig)
                return M*v, Utils.sdiag(v)*Md
            sig = np.random.rand(mesh.nC*mesh.dim)
            passed = checkDerivative(fun, sig, plotIt=False)
            self.assertTrue(passed)



if __name__ == '__main__':
    unittest.main()
