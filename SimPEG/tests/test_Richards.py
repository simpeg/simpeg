import numpy as np
import unittest
from SimPEG.mesh import TensorMesh
from TestUtils import OrderTest, checkDerivative
from scipy.sparse.linalg import dsolve
from SimPEG.forward import Richards


class RichardsTests(unittest.TestCase):

    def setUp(self):
        pass
        # a = np.array([1, 1, 1])
        # b = np.array([1, 2])
        # c = np.array([1, 4])
        # self.mesh2 = TensorMesh([a, b], np.array([3, 5]))
        # self.mesh3 = TensorMesh([a, b, c])

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



if __name__ == '__main__':
    unittest.main()
