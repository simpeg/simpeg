import numpy as np
import unittest
from SimPEG import *
from TestUtils import checkDerivative
from scipy.sparse.linalg import dsolve


class ModelTests(unittest.TestCase):

    def setUp(self):

        a = np.array([1, 1, 1])
        b = np.array([1, 2])
        c = np.array([1, 4])
        self.mesh2 = Mesh.TensorMesh([a, b], np.array([3, 5]))

    def test_modelTransforms(self):
        print 'SimPEG.Model.BaseModel: Testing Model Transform'
        for M in dir(Model):
            if 'Model' not in M: continue
            model = getattr(Model, M)()
            m = model.example(self.mesh2)
            passed = checkDerivative(lambda m : [model.transform(m), model.transformDeriv(m)], m, plotIt=False)
            self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
