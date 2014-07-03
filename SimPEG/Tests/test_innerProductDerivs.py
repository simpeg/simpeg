import numpy as np
import unittest
from SimPEG import *
from TestUtils import checkDerivative


class TestInnerProductsDerivs(unittest.TestCase):

    def doTestFace(self, h, rep, fast, meshType, invProp=False, invMat=False):
        if meshType == 'LRM':
            hRect = Utils.exampleLrmGrid(h,'rotate')
            mesh = Mesh.LogicallyRectMesh(hRect)
        elif meshType == 'Tree':
            mesh = Mesh.TreeMesh(h)
        elif meshType == 'Tensor':
            mesh = Mesh.TensorMesh(h)
        v = np.random.rand(mesh.nF)
        sig = np.random.rand(1) if rep is 0 else np.random.rand(mesh.nC*rep)
        def fun(sig):
            M  = mesh.getFaceInnerProduct(sig, invProp=invProp, invMat=invMat)
            Md = mesh.getFaceInnerProductDeriv(sig, invProp=invProp, invMat=invMat, doFast=fast)
            return M*v, Md(v)
        print meshType, 'Face', h, rep, fast, ('harmonic' if invProp and invMat else 'standard')
        return checkDerivative(fun, sig, num=5, plotIt=False)

    def doTestEdge(self, h, rep, fast, meshType, invProp=False, invMat=False):
        if meshType == 'LRM':
            hRect = Utils.exampleLrmGrid(h,'rotate')
            mesh = Mesh.LogicallyRectMesh(hRect)
        elif meshType == 'Tree':
            mesh = Mesh.TreeMesh(h)
        elif meshType == 'Tensor':
            mesh = Mesh.TensorMesh(h)
        v = np.random.rand(mesh.nE)
        sig = np.random.rand(1) if rep is 0 else np.random.rand(mesh.nC*rep)
        def fun(sig):
            M  = mesh.getEdgeInnerProduct(sig, invProp=invProp, invMat=invMat)
            Md = mesh.getEdgeInnerProductDeriv(sig, invProp=invProp, invMat=invMat, doFast=fast)
            return M*v, Md(v)
        print meshType, 'Edge', h, rep, fast, ('harmonic' if invProp and invMat else 'standard')
        return checkDerivative(fun, sig, num=5, plotIt=False)

    def test_FaceIP_1D_float(self):
        self.assertTrue(self.doTestFace([10],0, False, 'Tensor'))
    def test_FaceIP_2D_float(self):
        self.assertTrue(self.doTestFace([10, 4],0, False, 'Tensor'))
    def test_FaceIP_3D_float(self):
        self.assertTrue(self.doTestFace([10, 4, 5],0, False, 'Tensor'))
    def test_FaceIP_1D_isotropic(self):
        self.assertTrue(self.doTestFace([10],1, False, 'Tensor'))
    def test_FaceIP_2D_isotropic(self):
        self.assertTrue(self.doTestFace([10, 4],1, False, 'Tensor'))
    def test_FaceIP_3D_isotropic(self):
        self.assertTrue(self.doTestFace([10, 4, 5],1, False, 'Tensor'))
    def test_FaceIP_2D_anisotropic(self):
        self.assertTrue(self.doTestFace([10, 4],2, False, 'Tensor'))
    def test_FaceIP_3D_anisotropic(self):
        self.assertTrue(self.doTestFace([10, 4, 5],3, False, 'Tensor'))
    def test_FaceIP_2D_tensor(self):
        self.assertTrue(self.doTestFace([10, 4],3, False, 'Tensor'))
    def test_FaceIP_3D_tensor(self):
        self.assertTrue(self.doTestFace([10, 4, 5],6, False, 'Tensor'))

    def test_FaceIP_1D_float_fast(self):
        self.assertTrue(self.doTestFace([10],0, True, 'Tensor'))
    def test_FaceIP_2D_float_fast(self):
        self.assertTrue(self.doTestFace([10, 4],0, True, 'Tensor'))
    def test_FaceIP_3D_float_fast(self):
        self.assertTrue(self.doTestFace([10, 4, 5],0, True, 'Tensor'))
    def test_FaceIP_1D_isotropic_fast(self):
        self.assertTrue(self.doTestFace([10],1, True, 'Tensor'))
    def test_FaceIP_2D_isotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4],1, True, 'Tensor'))
    def test_FaceIP_3D_isotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4, 5],1, True, 'Tensor'))
    def test_FaceIP_2D_anisotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4],2, True, 'Tensor'))
    def test_FaceIP_3D_anisotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4, 5],3, True, 'Tensor'))

    def test_EdgeIP_1D_float(self):
        self.assertTrue(self.doTestEdge([10],0, False, 'Tensor'))
    def test_EdgeIP_2D_float(self):
        self.assertTrue(self.doTestEdge([10, 4],0, False, 'Tensor'))
    def test_EdgeIP_3D_float(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],0, False, 'Tensor'))
    def test_EdgeIP_1D_isotropic(self):
        self.assertTrue(self.doTestEdge([10],1, False, 'Tensor'))
    def test_EdgeIP_2D_isotropic(self):
        self.assertTrue(self.doTestEdge([10, 4],1, False, 'Tensor'))
    def test_EdgeIP_3D_isotropic(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],1, False, 'Tensor'))
    def test_EdgeIP_2D_anisotropic(self):
        self.assertTrue(self.doTestEdge([10, 4],2, False, 'Tensor'))
    def test_EdgeIP_3D_anisotropic(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],3, False, 'Tensor'))
    def test_EdgeIP_2D_tensor(self):
        self.assertTrue(self.doTestEdge([10, 4],3, False, 'Tensor'))
    def test_EdgeIP_3D_tensor(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],6, False, 'Tensor'))

    def test_EdgeIP_1D_float_fast(self):
        self.assertTrue(self.doTestEdge([10],0, True, 'Tensor'))
    def test_EdgeIP_2D_float_fast(self):
        self.assertTrue(self.doTestEdge([10, 4],0, True, 'Tensor'))
    def test_EdgeIP_3D_float_fast(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],0, True, 'Tensor'))
    def test_EdgeIP_1D_isotropic_fast(self):
        self.assertTrue(self.doTestEdge([10],1, True, 'Tensor'))
    def test_EdgeIP_2D_isotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4],1, True, 'Tensor'))
    def test_EdgeIP_3D_isotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],1, True, 'Tensor'))
    def test_EdgeIP_2D_anisotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4],2, True, 'Tensor'))
    def test_EdgeIP_3D_anisotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],3, True, 'Tensor'))



    def test_FaceIP_1D_float_fast_harmonic(self):
        self.assertTrue(self.doTestFace([10],0, True, 'Tensor', invProp=True, invMat=True))
    def test_FaceIP_2D_float_fast_harmonic(self):
        self.assertTrue(self.doTestFace([10, 4],0, True, 'Tensor', invProp=True, invMat=True))
    def test_FaceIP_3D_float_fast_harmonic(self):
        self.assertTrue(self.doTestFace([10, 4, 5],0, True, 'Tensor', invProp=True, invMat=True))
    def test_FaceIP_1D_isotropic_fast_harmonic(self):
        self.assertTrue(self.doTestFace([10],1, True, 'Tensor', invProp=True, invMat=True))
    def test_FaceIP_2D_isotropic_fast_harmonic(self):
        self.assertTrue(self.doTestFace([10, 4],1, True, 'Tensor', invProp=True, invMat=True))
    def test_FaceIP_3D_isotropic_fast_harmonic(self):
        self.assertTrue(self.doTestFace([10, 4, 5],1, True, 'Tensor', invProp=True, invMat=True))
    def test_FaceIP_2D_anisotropic_fast_harmonic(self):
        self.assertTrue(self.doTestFace([10, 4],2, True, 'Tensor', invProp=True, invMat=True))
    def test_FaceIP_3D_anisotropic_fast_harmonic(self):
        self.assertTrue(self.doTestFace([10, 4, 5],3, True, 'Tensor', invProp=True, invMat=True))



    def test_FaceIP_2D_float_LRM(self):
        self.assertTrue(self.doTestFace([10, 4],0, False, 'LRM'))
    def test_FaceIP_3D_float_LRM(self):
        self.assertTrue(self.doTestFace([10, 4, 5],0, False, 'LRM'))
    def test_FaceIP_2D_isotropic_LRM(self):
        self.assertTrue(self.doTestFace([10, 4],1, False, 'LRM'))
    def test_FaceIP_3D_isotropic_LRM(self):
        self.assertTrue(self.doTestFace([10, 4, 5],1, False, 'LRM'))
    def test_FaceIP_2D_anisotropic_LRM(self):
        self.assertTrue(self.doTestFace([10, 4],2, False, 'LRM'))
    def test_FaceIP_3D_anisotropic_LRM(self):
        self.assertTrue(self.doTestFace([10, 4, 5],3, False, 'LRM'))
    def test_FaceIP_2D_tensor_LRM(self):
        self.assertTrue(self.doTestFace([10, 4],3, False, 'LRM'))
    def test_FaceIP_3D_tensor_LRM(self):
        self.assertTrue(self.doTestFace([10, 4, 5],6, False, 'LRM'))

    def test_FaceIP_2D_float_fast_LRM(self):
        self.assertTrue(self.doTestFace([10, 4],0, True, 'LRM'))
    def test_FaceIP_3D_float_fast_LRM(self):
        self.assertTrue(self.doTestFace([10, 4, 5],0, True, 'LRM'))
    def test_FaceIP_2D_isotropic_fast_LRM(self):
        self.assertTrue(self.doTestFace([10, 4],1, True, 'LRM'))
    def test_FaceIP_3D_isotropic_fast_LRM(self):
        self.assertTrue(self.doTestFace([10, 4, 5],1, True, 'LRM'))
    def test_FaceIP_2D_anisotropic_fast_LRM(self):
        self.assertTrue(self.doTestFace([10, 4],2, True, 'LRM'))
    def test_FaceIP_3D_anisotropic_fast_LRM(self):
        self.assertTrue(self.doTestFace([10, 4, 5],3, True, 'LRM'))

    def test_EdgeIP_2D_float_LRM(self):
        self.assertTrue(self.doTestEdge([10, 4],0, False, 'LRM'))
    def test_EdgeIP_3D_float_LRM(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],0, False, 'LRM'))
    def test_EdgeIP_2D_isotropic_LRM(self):
        self.assertTrue(self.doTestEdge([10, 4],1, False, 'LRM'))
    def test_EdgeIP_3D_isotropic_LRM(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],1, False, 'LRM'))
    def test_EdgeIP_2D_anisotropic_LRM(self):
        self.assertTrue(self.doTestEdge([10, 4],2, False, 'LRM'))
    def test_EdgeIP_3D_anisotropic_LRM(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],3, False, 'LRM'))
    def test_EdgeIP_2D_tensor_LRM(self):
        self.assertTrue(self.doTestEdge([10, 4],3, False, 'LRM'))
    def test_EdgeIP_3D_tensor_LRM(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],6, False, 'LRM'))

    def test_EdgeIP_2D_float_fast_LRM(self):
        self.assertTrue(self.doTestEdge([10, 4],0, True, 'LRM'))
    def test_EdgeIP_3D_float_fast_LRM(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],0, True, 'LRM'))
    def test_EdgeIP_2D_isotropic_fast_LRM(self):
        self.assertTrue(self.doTestEdge([10, 4],1, True, 'LRM'))
    def test_EdgeIP_3D_isotropic_fast_LRM(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],1, True, 'LRM'))
    def test_EdgeIP_2D_anisotropic_fast_LRM(self):
        self.assertTrue(self.doTestEdge([10, 4],2, True, 'LRM'))
    def test_EdgeIP_3D_anisotropic_fast_LRM(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],3, True, 'LRM'))




    def test_FaceIP_2D_float_Tree(self):
        self.assertTrue(self.doTestFace([10, 4],0, False, 'Tree'))
    def test_FaceIP_3D_float_Tree(self):
        self.assertTrue(self.doTestFace([10, 4, 5],0, False, 'Tree'))
    def test_FaceIP_2D_isotropic_Tree(self):
        self.assertTrue(self.doTestFace([10, 4],1, False, 'Tree'))
    def test_FaceIP_3D_isotropic_Tree(self):
        self.assertTrue(self.doTestFace([10, 4, 5],1, False, 'Tree'))
    def test_FaceIP_2D_anisotropic_Tree(self):
        self.assertTrue(self.doTestFace([10, 4],2, False, 'Tree'))
    def test_FaceIP_3D_anisotropic_Tree(self):
        self.assertTrue(self.doTestFace([10, 4, 5],3, False, 'Tree'))
    def test_FaceIP_2D_tensor_Tree(self):
        self.assertTrue(self.doTestFace([10, 4],3, False, 'Tree'))
    def test_FaceIP_3D_tensor_Tree(self):
        self.assertTrue(self.doTestFace([10, 4, 5],6, False, 'Tree'))

    def test_FaceIP_2D_float_fast_Tree(self):
        self.assertTrue(self.doTestFace([10, 4],0, True, 'Tree'))
    def test_FaceIP_3D_float_fast_Tree(self):
        self.assertTrue(self.doTestFace([10, 4, 5],0, True, 'Tree'))
    def test_FaceIP_2D_isotropic_fast_Tree(self):
        self.assertTrue(self.doTestFace([10, 4],1, True, 'Tree'))
    def test_FaceIP_3D_isotropic_fast_Tree(self):
        self.assertTrue(self.doTestFace([10, 4, 5],1, True, 'Tree'))
    def test_FaceIP_2D_anisotropic_fast_Tree(self):
        self.assertTrue(self.doTestFace([10, 4],2, True, 'Tree'))
    def test_FaceIP_3D_anisotropic_fast_Tree(self):
        self.assertTrue(self.doTestFace([10, 4, 5],3, True, 'Tree'))

    def test_EdgeIP_2D_float_Tree(self):
        self.assertTrue(self.doTestEdge([10, 4],0, False, 'Tree'))
    def test_EdgeIP_3D_float_Tree(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],0, False, 'Tree'))
    def test_EdgeIP_2D_isotropic_Tree(self):
        self.assertTrue(self.doTestEdge([10, 4],1, False, 'Tree'))
    def test_EdgeIP_3D_isotropic_Tree(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],1, False, 'Tree'))
    def test_EdgeIP_2D_anisotropic_Tree(self):
        self.assertTrue(self.doTestEdge([10, 4],2, False, 'Tree'))
    def test_EdgeIP_3D_anisotropic_Tree(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],3, False, 'Tree'))
    def test_EdgeIP_2D_tensor_Tree(self):
        self.assertTrue(self.doTestEdge([10, 4],3, False, 'Tree'))
    def test_EdgeIP_3D_tensor_Tree(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],6, False, 'Tree'))

    def test_EdgeIP_2D_float_fast_Tree(self):
        self.assertTrue(self.doTestEdge([10, 4],0, True, 'Tree'))
    def test_EdgeIP_3D_float_fast_Tree(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],0, True, 'Tree'))
    def test_EdgeIP_2D_isotropic_fast_Tree(self):
        self.assertTrue(self.doTestEdge([10, 4],1, True, 'Tree'))
    def test_EdgeIP_3D_isotropic_fast_Tree(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],1, True, 'Tree'))
    def test_EdgeIP_2D_anisotropic_fast_Tree(self):
        self.assertTrue(self.doTestEdge([10, 4],2, True, 'Tree'))
    def test_EdgeIP_3D_anisotropic_fast_Tree(self):
        self.assertTrue(self.doTestEdge([10, 4, 5],3, True, 'Tree'))

if __name__ == '__main__':
    unittest.main()
