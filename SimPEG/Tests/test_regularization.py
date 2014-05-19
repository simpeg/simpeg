import numpy as np
import unittest
from SimPEG import *
from TestUtils import checkDerivative
from scipy.sparse.linalg import dsolve
import inspect


class RegularizationTests(unittest.TestCase):

    def setUp(self):
        self.mesh2 = Mesh.TensorMesh([3, 2])

    def test_regularization(self):
        for R in dir(Regularization):
            r = getattr(Regularization, R)
            if not inspect.isclass(r): continue
            if not issubclass(r, Regularization.BaseRegularization):
                continue
            # if 'Regularization' not in R: continue
            print 'Check:', R
            mapping = r.mapPair(self.mesh2)
            reg = r(self.mesh2, mapping=mapping)
            m = np.random.rand(mapping.nP)
            reg.mref = m[:]*0
            passed = checkDerivative(lambda m : [reg.eval(m), reg.evalDeriv(m)], m, plotIt=False)
            self.assertTrue(passed)


if __name__ == '__main__':
    unittest.main()
