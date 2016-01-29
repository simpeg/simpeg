import numpy as np
import unittest
from SimPEG import *
from scipy.sparse.linalg import dsolve
import inspect


DO_NOT_TEST_REG2 = ['Simple']
DO_NOT_TEST_REG3 = []

class RegularizationTests(unittest.TestCase):

    def setUp(self):
        self.mesh2 = Mesh.TensorMesh([3, 2])

    def test_regularization(self):
        for R in dir(Regularization):
            r = getattr(Regularization, R)
            if not inspect.isclass(r): continue
            if not issubclass(r, Regularization.BaseRegularization):
                continue
            if r.__name__ in DO_NOT_TEST_REG2: continue
            mapping = r.mapPair(self.mesh2)
            reg = r(self.mesh2, mapping=mapping)
            m = np.random.rand(mapping.nP)
            reg.mref = m[:]*np.mean(m)

            print 'Check:', R
            passed = Tests.checkDerivative(lambda m : [reg.eval(m), reg.evalDeriv(m)], m, plotIt=False)
            self.assertTrue(passed)
            print 'Check 2 Deriv:', R
            passed = Tests.checkDerivative(lambda m : [reg.evalDeriv(m), reg.eval2Deriv(m)], m, plotIt=False)
            self.assertTrue(passed)


class RegularizationTests3D(unittest.TestCase):

    def setUp(self):
        self.mesh3 = Mesh.TensorMesh([3, 2, 5])

    def test_regularization(self):
        for R in dir(Regularization):
            r = getattr(Regularization, R)
            if not inspect.isclass(r): continue
            if not issubclass(r, Regularization.BaseRegularization):
                continue
            if r.__name__ in DO_NOT_TEST_REG3: continue
            mapping = r.mapPair(self.mesh3)
            reg = r(self.mesh3, mapping=mapping)
            m = np.random.rand(mapping.nP)
            reg.mref = m[:]*np.mean(m)

            print 'Check:', R
            passed = Tests.checkDerivative(lambda m : [reg.eval(m), reg.evalDeriv(m)], m, plotIt=False)
            self.assertTrue(passed)
            print 'Check 2 Deriv:', R
            passed = Tests.checkDerivative(lambda m : [reg.evalDeriv(m), reg.eval2Deriv(m)], m, plotIt=False)
            self.assertTrue(passed)


if __name__ == '__main__':
    unittest.main()
