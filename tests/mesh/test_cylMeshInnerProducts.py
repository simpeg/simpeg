from __future__ import print_function
from SimPEG import Mesh, Utils, Tests
import numpy as np
import sympy
from sympy.abc import r, t, z
import unittest
import scipy.sparse as sp

TOL = 1e-1
TOLD = 0.7  # tolerance on deriv checks


class FaceInnerProductFctsIsotropic(object):
    """ Some made up face functions to test the face inner product """
    def fcts(self):
        j = sympy.Matrix([
            r**2 * z,
            r * z**2
        ])

        # Create an isotropic sigma vector
        Sig = sympy.Matrix([
            [420/(sympy.pi)*(r*z)**2, 0],
            [0, 420/(sympy.pi)*(r*z)**2],
        ])

        return j, Sig

    def sol(self):
        # Do the inner product! - we are in cyl coordinates!
        j, Sig = self.fcts()
        jTSj = j.T*Sig*j
        # we are integrating in cyl coordinates
        ans  = sympy.integrate(sympy.integrate(sympy.integrate(r * jTSj,
                                                               (r, 0, 1)),
                                               (t, 0, 2*sympy.pi)),
                                (z, 0, 1))[0] # The `[0]` is to make it an int.

        return ans

    def vectors(self, mesh):
        """ Get Vectors sig, sr. jx from sympy"""
        j, Sig = self.fcts()

        f_jr = sympy.lambdify((r, z), j[0], 'numpy')
        f_jz = sympy.lambdify((r, z), j[1], 'numpy')
        f_sigr = sympy.lambdify((r, z), Sig[0], 'numpy')
        # f_sigz = sympy.lambdify((r,z), Sig[1], 'numpy')

        jr = f_jr(mesh.gridFx[:, 0], mesh.gridFx[:, 2])
        jz = f_jz(mesh.gridFz[:, 0], mesh.gridFz[:, 2])
        sigr = f_sigr(mesh.gridCC[:, 0], mesh.gridCC[:, 2])

        return sigr, np.r_[jr, jz]


class FaceInnerProductFunctionsDiagAnisotropic(FaceInnerProductFctsIsotropic):
    """
        Some made up face functions to test the diagonally anisotropic face
        inner product
    """

    def fcts(self):
        j = sympy.Matrix([
            r**2 * z,
            r * z**2
        ])

        # Create an isotropic sigma vector
        Sig = sympy.Matrix([
            [120/(sympy.pi)*(r*z)**2, 0],
            [0, 420/(sympy.pi)*(r*z)**2],
        ])

        return j, Sig

    def vectors(self, mesh):
        """ Get Vectors sig, sr. jx from sympy"""
        j, Sig = self.fcts()

        f_jr = sympy.lambdify((r, z), j[0], 'numpy')
        f_jz = sympy.lambdify((r, z), j[1], 'numpy')
        f_sigr = sympy.lambdify((r, z), Sig[0], 'numpy')
        f_sigz = sympy.lambdify((r, z), Sig[3], 'numpy')

        jr = f_jr(mesh.gridFx[:, 0], mesh.gridFx[:, 2])
        jz = f_jz(mesh.gridFz[:, 0], mesh.gridFz[:, 2])
        sigr = f_sigr(mesh.gridCC[:, 0], mesh.gridCC[:, 2])
        sigz = f_sigz(mesh.gridCC[:, 0], mesh.gridCC[:, 2])

        return np.c_[sigr, sigr, sigz], np.r_[jr, jz]


class EdgeInnerProductFctsIsotropic(object):
    """ Some made up edge functions to test the edge inner product """

    def fcts(self):
        h = sympy.Matrix([r**2 * z])

        # Create an isotropic sigma vector
        Sig = sympy.Matrix([200/(sympy.pi)*(r*z)**2])

        return h, Sig

    def sol(self):
        h, Sig = self.fcts()
        # Do the inner product! - we are in cyl coordinates!
        hTSh = h.T*Sig*h
        ans  = sympy.integrate(sympy.integrate(sympy.integrate(r * hTSh,
                                                               (r, 0, 1)),
                                               (t, 0, 2*sympy.pi)),
                                (z, 0, 1))[0] # The `[0]` is to make it an int.
        return ans

    def vectors(self, mesh):
        """ Get Vectors sig, sr. jx from sympy"""
        h, Sig = self.fcts()

        f_h = sympy.lambdify((r, z), h[0], 'numpy')
        f_sig = sympy.lambdify((r, z), Sig[0], 'numpy')

        ht = f_h(mesh.gridEy[:, 0], mesh.gridEy[:, 2])
        sig = f_sig(mesh.gridCC[:, 0], mesh.gridCC[:, 2])

        return sig, np.r_[ht]


class EdgeInnerProductFunctionsDiagAnisotropic(EdgeInnerProductFctsIsotropic):
    """
        Some made up edge functions to test the diagonally anisotropic edge
        inner product
    """

    def vectors(self, mesh):
        h, Sig = self.fcts()

        f_h = sympy.lambdify((r, z), h[0], 'numpy')
        f_sig = sympy.lambdify((r, z), Sig[0], 'numpy')

        ht = f_h(mesh.gridEy[:, 0], mesh.gridEy[:, 2])
        sig = f_sig(mesh.gridCC[:, 0], mesh.gridCC[:, 2])

        return np.c_[sig, sig, sig], np.r_[ht]


class TestCylInnerProducts_simple(unittest.TestCase):

    def setUp(self):
        n = 100.
        self.mesh = Mesh.CylMesh([n, 1, n])

    def test_FaceInnerProductIsotropic(self):
        # Here we will make up some j vectors that vary in space
        # j = [j_r, j_z] - to test face inner products

        fcts = FaceInnerProductFctsIsotropic()
        sig, jv = fcts.vectors(self.mesh)
        MfSig = self.mesh.getFaceInnerProduct(sig)
        numeric_ans = jv.T.dot(MfSig.dot(jv))

        ans = fcts.sol()

        print('------ Testing Face Inner Product-----------')
        print(' Analytic: {analytic}, Numeric: {numeric}, '
              'ratio (num/ana): {ratio}'.format(
               analytic=ans, numeric=numeric_ans,
               ratio=float(numeric_ans)/ans))
        assert(np.abs(ans-numeric_ans) < TOL)

    def test_FaceInnerProductDiagAnisotropic(self):
        # Here we will make up some j vectors that vary in space
        # j = [j_r, j_z] - to test face inner products

        fcts = FaceInnerProductFunctionsDiagAnisotropic()
        sig, jv = fcts.vectors(self.mesh)
        MfSig = self.mesh.getFaceInnerProduct(sig)
        numeric_ans = jv.T.dot(MfSig.dot(jv))

        ans = fcts.sol()

        print('------ Testing Face Inner Product Anisotropic -----------')
        print(' Analytic: {analytic}, Numeric: {numeric}, '
              'ratio (num/ana): {ratio}'.format(
               analytic=ans, numeric=numeric_ans,
               ratio=float(numeric_ans)/ans))
        assert(np.abs(ans-numeric_ans) < TOL)

    def test_EdgeInnerProduct(self):
        # Here we will make up some j vectors that vary in space
        # h = [h_t] - to test edge inner products

        fcts = EdgeInnerProductFctsIsotropic()
        sig, hv = fcts.vectors(self.mesh)
        MeSig = self.mesh.getEdgeInnerProduct(sig)
        numeric_ans = hv.T.dot(MeSig.dot(hv))

        ans = fcts.sol()

        print('------ Testing Edge Inner Product-----------')
        print(' Analytic: {analytic}, Numeric: {numeric}, '
              'ratio (num/ana): {ratio}'.format(
               analytic=ans, numeric=numeric_ans,
               ratio=float(numeric_ans)/ans))
        assert(np.abs(ans-numeric_ans) < TOL)

    def test_EdgeInnerProductDiagAnisotropic(self):
        # Here we will make up some j vectors that vary in space
        # h = [h_t] - to test edge inner products

        fcts = EdgeInnerProductFunctionsDiagAnisotropic()

        sig, hv = fcts.vectors(self.mesh)
        MeSig = self.mesh.getEdgeInnerProduct(sig)
        numeric_ans = hv.T.dot(MeSig.dot(hv))

        ans = fcts.sol()

        print('------ Testing Edge Inner Product Anisotropic -----------')
        print(' Analytic: {analytic}, Numeric: {numeric}, '
              'ratio (num/ana): {ratio}'.format(
               analytic=ans, numeric=numeric_ans,
               ratio=float(numeric_ans)/ans))
        assert(np.abs(ans-numeric_ans) < TOL)


class TestCylFaceInnerProducts_Order(Tests.OrderTest):

    meshTypes = ['uniformCylMesh']
    meshDimension = 2

    def getError(self):
        fct = FaceInnerProductFctsIsotropic()
        sig, jv = fct.vectors(self.M)
        Msig = self.M.getFaceInnerProduct(sig)
        return float(fct.sol()) - jv.T.dot(Msig.dot(jv))

    def test_order(self):
        self.orderTest()


class TestCylEdgeInnerProducts_Order(Tests.OrderTest):

    meshTypes = ['uniformCylMesh']
    meshDimension = 2

    def getError(self):
        fct = EdgeInnerProductFctsIsotropic()
        sig, ht = fct.vectors(self.M)
        Msig = self.M.getEdgeInnerProduct(sig)
        return float(fct.sol()) - ht.T.dot(Msig.dot(ht))

    def test_order(self):
        self.orderTest()


class TestCylFaceInnerProductsDiagAnisotropic_Order(Tests.OrderTest):

    meshTypes = ['uniformCylMesh']
    meshDimension = 2

    def getError(self):
        fct = FaceInnerProductFunctionsDiagAnisotropic()
        sig, jv = fct.vectors(self.M)
        Msig = self.M.getFaceInnerProduct(sig)
        return float(fct.sol()) - jv.T.dot(Msig.dot(jv))

    def test_order(self):
        self.orderTest()


class TestCylEdgeInnerProducts_Order(Tests.OrderTest):

    meshTypes = ['uniformCylMesh']
    meshDimension = 2

    def getError(self):
        fct = EdgeInnerProductFunctionsDiagAnisotropic()
        sig, ht = fct.vectors(self.M)
        Msig = self.M.getEdgeInnerProduct(sig)
        return float(fct.sol()) - ht.T.dot(Msig.dot(ht))

    def test_order(self):
        self.orderTest()


class TestCylInnerProducts_Deriv(unittest.TestCase):

    def setUp(self):
        n = 2
        self.mesh = Mesh.CylMesh([n, 1, n])
        self.face_vec = np.random.rand(self.mesh.nF)
        self.edge_vec = np.random.rand(self.mesh.nE)
        # make up a smooth function
        self.x0 = 2*self.mesh.gridCC[:, 0]**2 + self.mesh.gridCC[:, 2]**4

    def test_FaceInnerProductIsotropicDeriv(self):

        def fun(x):
            MfSig = self.mesh.getFaceInnerProduct(x)
            MfSigDeriv = self.mesh.getFaceInnerProductDeriv(self.x0)
            return MfSig*self.face_vec, MfSigDeriv(self.face_vec)

        print('Testing FaceInnerProduct Isotropic')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                               tolerance=TOLD, plotIt=False))

    def test_FaceInnerProductIsotropicDerivInvProp(self):

        def fun(x):
            MfSig = self.mesh.getFaceInnerProduct(x, invProp=True)
            MfSigDeriv = self.mesh.getFaceInnerProductDeriv(self.x0,
                                                            invProp=True)
            return MfSig*self.face_vec, MfSigDeriv(self.face_vec)

        print('Testing FaceInnerProduct Isotropic InvProp')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                                                     tolerance=TOLD,
                                                     plotIt=False))

    def test_FaceInnerProductIsotropicDerivInvMat(self):

        def fun(x):
            MfSig = self.mesh.getFaceInnerProduct(x, invMat=True)
            MfSigDeriv = self.mesh.getFaceInnerProductDeriv(self.x0,
                                                            invMat=True)
            return MfSig*self.face_vec, MfSigDeriv(self.face_vec)

        print('Testing FaceInnerProduct Isotropic InvMat')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                                                     tolerance=TOLD,
                                                     plotIt=False))


    def test_FaceInnerProductIsotropicDerivInvPropInvMat(self):

        def fun(x):
            MfSig = self.mesh.getFaceInnerProduct(x, invProp=True, invMat=True)
            MfSigDeriv = self.mesh.getFaceInnerProductDeriv(self.x0,
                                                            invProp=True,
                                                            invMat=True)
            return MfSig*self.face_vec, MfSigDeriv(self.face_vec)

        print('Testing FaceInnerProduct Isotropic InvProp InvMat')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                                                     tolerance=TOLD,
                                                     plotIt=False))

    def test_EdgeInnerProductIsotropicDeriv(self):

        def fun(x):
            MeSig = self.mesh.getEdgeInnerProduct(x)
            MeSigDeriv = self.mesh.getEdgeInnerProductDeriv(self.x0)
            return MeSig*self.edge_vec, MeSigDeriv(self.edge_vec)

        print('Testing EdgeInnerProduct Isotropic')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                                                     tolerance=TOLD,
                                                     plotIt=False))

    def test_EdgeInnerProductIsotropicDerivInvProp(self):

        def fun(x):
            MeSig = self.mesh.getEdgeInnerProduct(x, invProp=True)
            MeSigDeriv = self.mesh.getEdgeInnerProductDeriv(self.x0,
                                                            invProp=True)
            return MeSig*self.edge_vec, MeSigDeriv(self.edge_vec)

        print('Testing EdgeInnerProduct Isotropic InvProp')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                                                     tolerance=TOLD,
                                                     plotIt=False))

    def test_EdgeInnerProductIsotropicDerivInvMat(self):

        def fun(x):
            MeSig = self.mesh.getEdgeInnerProduct(x, invMat=True)
            MeSigDeriv = self.mesh.getEdgeInnerProductDeriv(self.x0,
                                                            invMat=True)
            return MeSig*self.edge_vec, MeSigDeriv(self.edge_vec)

        print('Testing EdgeInnerProduct Isotropic InvMat')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                                                     tolerance=TOLD,
                                                     plotIt=False))

    def test_EdgeInnerProductIsotropicDerivInvPropInvMat(self):

        def fun(x):
            MeSig = self.mesh.getEdgeInnerProduct(x, invProp=True, invMat=True)
            MeSigDeriv = self.mesh.getEdgeInnerProductDeriv(self.x0,
                                                            invProp=True,
                                                            invMat=True)
            return MeSig*self.edge_vec, MeSigDeriv(self.edge_vec)

        print('Testing EdgeInnerProduct Isotropic InvProp InvMat')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                                                     tolerance=TOLD,
                                                     plotIt=False))


class TestCylInnerProductsAnisotropic_Deriv(unittest.TestCase):

    def setUp(self):
        n = 60
        self.mesh = Mesh.CylMesh([n, 1, n])
        self.face_vec = np.random.rand(self.mesh.nF)
        self.edge_vec = np.random.rand(self.mesh.nE)
        # make up a smooth function
        self.x0 = np.array([2*self.mesh.gridCC[:, 0]**2 + self.mesh.gridCC[:, 2]**4
                             ])

    def test_FaceInnerProductAnisotropicDeriv(self):

        def fun(x):
            # fake anisotropy (testing anistropic implementation with isotropic
            # vector). First order behavior expected for fully anisotropic
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            Zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            Eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([Eye, Zero, Eye])])

            MfSig = self.mesh.getFaceInnerProduct(x)
            MfSigDeriv = self.mesh.getFaceInnerProductDeriv(x0)
            return MfSig*self.face_vec ,  MfSigDeriv(self.face_vec) * P.T

        print('Testing FaceInnerProduct Anisotropic')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                               tolerance=TOLD, plotIt=False))

    def test_FaceInnerProductAnisotropicDerivInvProp(self):

        def fun(x):
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            Zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            Eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([Eye, Zero, Eye])])

            MfSig = self.mesh.getFaceInnerProduct(x, invProp=True)
            MfSigDeriv = self.mesh.getFaceInnerProductDeriv(x0,
                                                            invProp=True)
            return MfSig*self.face_vec, MfSigDeriv(self.face_vec) * P.T

        print('Testing FaceInnerProduct Anisotropic InvProp')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                                                     tolerance=TOLD,
                                                     plotIt=False))

    def test_FaceInnerProductAnisotropicDerivInvMat(self):

        def fun(x):
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            Zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            Eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([Eye, Zero, Eye])])

            MfSig = self.mesh.getFaceInnerProduct(x, invMat=True)
            MfSigDeriv = self.mesh.getFaceInnerProductDeriv(x0, invMat=True)
            return MfSig*self.face_vec, MfSigDeriv(self.face_vec) * P.T

        print('Testing FaceInnerProduct Anisotropic InvMat')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                                                     tolerance=TOLD,
                                                     plotIt=False))

    def test_FaceInnerProductAnisotropicDerivInvPropInvMat(self):

        def fun(x):
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            Zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            Eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([Eye, Zero, Eye])])

            MfSig = self.mesh.getFaceInnerProduct(x, invProp=True, invMat=True)
            MfSigDeriv = self.mesh.getFaceInnerProductDeriv(x0,
                                                            invProp=True,
                                                            invMat=True)
            return MfSig*self.face_vec, MfSigDeriv(self.face_vec) * P.T

        print('Testing FaceInnerProduct Anisotropic InvProp InvMat')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                                                     tolerance=TOLD,
                                                     plotIt=False))

    def test_EdgeInnerProductAnisotropicDeriv(self):

        def fun(x):
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            Zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            Eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([Zero, Eye, Zero])])

            MeSig = self.mesh.getEdgeInnerProduct(x.reshape(self.mesh.nC, 3))
            MeSigDeriv = self.mesh.getEdgeInnerProductDeriv(x0)
            return MeSig*self.edge_vec, MeSigDeriv(self.edge_vec) * P.T

        print('Testing EdgeInnerProduct Anisotropic')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                                                     tolerance=TOLD,
                                                     plotIt=False))

    def test_EdgeInnerProductAnisotropicDerivInvProp(self):

        def fun(x):
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            Zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            Eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([Zero, Eye, Zero])])

            MeSig = self.mesh.getEdgeInnerProduct(x, invProp=True)
            MeSigDeriv = self.mesh.getEdgeInnerProductDeriv(x0, invProp=True)
            return MeSig*self.edge_vec, MeSigDeriv(self.edge_vec) * P.T

        print('Testing EdgeInnerProduct Anisotropic InvProp')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                                                     tolerance=TOLD,
                                                     plotIt=False))

    def test_EdgeInnerProductAnisotropicDerivInvMat(self):

        def fun(x):
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            Zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            Eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([Zero, Eye, Zero])])

            MeSig = self.mesh.getEdgeInnerProduct(x, invMat=True)
            MeSigDeriv = self.mesh.getEdgeInnerProductDeriv(x0, invMat=True)
            return MeSig*self.edge_vec, MeSigDeriv(self.edge_vec) * P.T

        print('Testing EdgeInnerProduct Anisotropic InvMat')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                                                     tolerance=TOLD,
                                                     plotIt=False))

    def test_EdgeInnerProductAnisotropicDerivInvPropInvMat(self):

        def fun(x):
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            Zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            Eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([Zero, Eye, Zero])])

            MeSig = self.mesh.getEdgeInnerProduct(x, invProp=True, invMat=True)
            MeSigDeriv = self.mesh.getEdgeInnerProductDeriv(x0,
                                                            invProp=True,
                                                            invMat=True)
            return MeSig*self.edge_vec, MeSigDeriv(self.edge_vec) * P.T

        print('Testing EdgeInnerProduct Anisotropic InvProp InvMat')
        return self.assertTrue(Tests.checkDerivative(fun, self.x0, num=7,
                                                     tolerance=TOLD,
                                                     plotIt=False))


if __name__ == '__main__':
    unittest.main()
