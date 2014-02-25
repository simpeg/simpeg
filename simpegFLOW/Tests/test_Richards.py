import unittest
from SimPEG import *
from SimPEG.Tests.TestUtils import OrderTest, checkDerivative
from scipy.sparse.linalg import dsolve
import simpegFLOW.Richards

TOL = 1E-8

class EmpiricalRelations(unittest.TestCase):


    def test_BaseHaverkamp_Theta(self):
        mesh = Mesh.TensorMesh([50])
        hav = Richards.BaseHaverkamp_theta(mesh)
        m = np.random.randn(50)
        def wrapper(u):
            return hav.transform(u, m), hav.transformDerivU(u, m)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)


    def test_BaseHaverkamp_k(self):
        mesh = Mesh.TensorMesh([50])
        hav = Richards.BaseHaverkamp_k(mesh)
        m = np.random.randn(50)
        def wrapper(u):
            return hav.transform(u, m), hav.transformDerivU(u, m)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)

        hav = Richards.BaseHaverkamp_k(mesh)
        u = np.random.randn(50)
        def wrapper(m):
            return hav.transform(u, m), hav.transformDerivM(u, m)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)

    # def test_Haverkamp_hydraulicConductivity(self):
    #     print 'Haverkamp_hydraulicConductivity'
    #     hav = Richards.Haverkamp()
    #     def wrapper(x):
    #         return hav.hydraulicConductivity(x), hav.hydraulicConductivityDeriv(x)
    #     passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
    #     self.assertTrue(passed,True)

    # def test_Haverkamp_hydraulicConductivity_FullKs(self):
    #     print 'Haverkamp_hydraulicConductivity_FullKs'
    #     n = 50
    #     hav = Richards.Haverkamp(Ks=np.random.rand(n))
    #     def wrapper(x):
    #         return hav.hydraulicConductivity(x), hav.hydraulicConductivityDeriv(x)
    #     passed = checkDerivative(wrapper, np.random.randn(n), plotIt=False)
    #     self.assertTrue(passed,True)

if __name__ == '__main__':
    unittest.main()
