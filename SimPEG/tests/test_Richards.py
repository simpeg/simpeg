import numpy as np
import unittest
from SimPEG.mesh import TensorMesh
from TestUtils import OrderTest, checkDerivative
from scipy.sparse.linalg import dsolve
from SimPEG.forward import Richards


class RichardsTests(unittest.TestCase):

    def setUp(self):
        M = TensorMesh([np.ones(40)])
        Ks = 9.4400e-03
        E = Richards.Haverkamp(Ks=np.log(Ks), A=1.1750e+06, gamma=4.74, alpha=1.6110e+06, theta_s=0.287, theta_r=0.075, beta=3.96)

        prob = Richards.RichardsProblem(M,E)
        prob.timeStep = 1
        prob.boundaryConditions = np.array([-61.5,-20.7])
        prob.doNewton = True
        prob.method = 'mixed'

        h = np.zeros(M.nC) + prob.boundaryConditions[0]

        self.h0 = h
        self.M = M
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
        checkDerivative(lambda hn1: self.prob.getResidual(self.h0,hn1), self.h0, plotIt=False)

    def test_Richards_getResidual_Picard(self):
        self.prob.doNewton = False
        checkDerivative(lambda hn1: self.prob.getResidual(self.h0,hn1), self.h0, plotIt=False, expectedOrder=1)



if __name__ == '__main__':
    unittest.main()
