import numpy as np
import scipy.sparse as sp
import unittest
from SimPEG.mesh import TensorMesh
from TestUtils import OrderTest, checkDerivative
from scipy.sparse.linalg import dsolve
from SimPEG.forward import Richards


TOL = 1E-8

class RichardsTests(unittest.TestCase):

    def setUp(self):
        M = TensorMesh([np.ones(40)])
        Ks = 9.4400e-03
        E = Richards.Haverkamp(Ks=np.log(Ks), A=1.1750e+06, gamma=4.74, alpha=1.6110e+06, theta_s=0.287, theta_r=0.075, beta=3.96)

        bc = np.array([-61.5,-20.7])
        h = np.zeros(M.nC) + bc[0]
        prob = Richards.RichardsProblem(M,E, timeStep=30, timeEnd=360, boundaryConditions=bc, initialConditions=h, doNewton=False, method='mixed')

        q = sp.csr_matrix((np.ones(4),(np.arange(4),np.array([20, 30, 35, 38]))),shape=(4,M.nCx))
        P = sp.kron(sp.identity(prob.numIts),q)
        prob.P = P


        self.h0 = h
        self.M = M
        self.Ks = Ks
        self.prob = prob

    def test_VanGenuchten_moistureContent(self):
        vanG = Richards.VanGenuchten()
        def wrapper(x):
            return vanG.moistureContent(x), vanG.moistureContentDeriv(x)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)

    def test_VanGenuchten_hydraulicConductivity(self):
        hav = Richards.VanGenuchten()
        def wrapper(x):
            return hav.hydraulicConductivity(x), hav.hydraulicConductivityDeriv(x)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)

    def test_VanGenuchten_hydraulicConductivity_FullKs(self):
        n = 50
        hav = Richards.VanGenuchten(Ks=np.random.rand(n))
        def wrapper(x):
            return hav.hydraulicConductivity(x), hav.hydraulicConductivityDeriv(x)
        passed = checkDerivative(wrapper, np.random.randn(n), plotIt=False)
        self.assertTrue(passed,True)

    def test_Haverkamp_moistureContent(self):
        hav = Richards.Haverkamp()
        def wrapper(x):
            return hav.moistureContent(x), hav.moistureContentDeriv(x)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)

    def test_Haverkamp_hydraulicConductivity(self):
        hav = Richards.Haverkamp()
        def wrapper(x):
            return hav.hydraulicConductivity(x), hav.hydraulicConductivityDeriv(x)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)

    def test_Haverkamp_hydraulicConductivity_FullKs(self):
        n = 50
        hav = Richards.Haverkamp(Ks=np.random.rand(n))
        def wrapper(x):
            return hav.hydraulicConductivity(x), hav.hydraulicConductivityDeriv(x)
        passed = checkDerivative(wrapper, np.random.randn(n), plotIt=False)
        self.assertTrue(passed,True)

    def test_Richards_getResidual_Newton(self):
        self.prob.doNewton = True
        passed = checkDerivative(lambda hn1: self.prob.getResidual(self.h0,hn1), self.h0, plotIt=False)
        self.assertTrue(passed,True)

    def test_Richards_getResidual_Picard(self):
        self.prob.doNewton = False
        passed = checkDerivative(lambda hn1: self.prob.getResidual(self.h0,hn1), self.h0, plotIt=False, expectedOrder=1)
        self.assertTrue(passed,True)

    def test_Adjoint_PressureHead(self):
        self.prob.dataType = 'pressureHead'
        Ks = self.Ks
        v = np.random.rand(self.prob.P.shape[0])
        z = np.random.rand(self.M.nC)
        Hs = self.prob.field(np.log(Ks))
        vJz = v.dot(self.prob.J(np.log(Ks),z,u=Hs))
        zJv = z.dot(self.prob.Jt(np.log(Ks),v,u=Hs))
        tol = TOL*(10**int(np.log10(zJv)))
        passed = np.abs(vJz - zJv) < tol
        print 'Richards Adjoint Test - PressureHead'
        print '%4.4e === %4.4e, diff=%4.4e < %4.e'%(vJz, zJv,np.abs(vJz - zJv),tol)
        self.assertTrue(passed,True)


    def test_Adjoint_Saturation(self):
        self.prob.dataType = 'saturation'
        Ks = self.Ks
        v = np.random.rand(self.prob.P.shape[0])
        z = np.random.rand(self.M.nC)
        Hs = self.prob.field(np.log(Ks))
        vJz = v.dot(self.prob.J(np.log(Ks),z,u=Hs))
        zJv = z.dot(self.prob.Jt(np.log(Ks),v,u=Hs))
        tol = TOL*(10**int(np.log10(zJv)))
        passed = np.abs(vJz - zJv) < tol
        print 'Richards Adjoint Test - Saturation'
        print '%4.4e === %4.4e, diff=%4.4e < %4.e'%(vJz, zJv,np.abs(vJz - zJv),tol)
        self.assertTrue(passed,True)





if __name__ == '__main__':
    unittest.main()
