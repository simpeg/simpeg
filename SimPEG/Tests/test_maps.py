import numpy as np
import unittest
from SimPEG import *
from TestUtils import checkDerivative
from scipy.sparse.linalg import dsolve


class MapTests(unittest.TestCase):

    def setUp(self):

        a = np.array([1, 1, 1])
        b = np.array([1, 2])
        self.mesh2 = Mesh.TensorMesh([a, b], x0=np.array([3, 5]))
        self.mesh22 = Mesh.TensorMesh([b, a], x0=np.array([3, 5]))

    def test_modelTransforms(self):
        for M in dir(Maps):
            try:
                model = getattr(Maps, M)(self.mesh2)
                assert isinstance(model, Maps.BaseModel)
            except Exception, e:
                continue
            self.assertTrue(model.test())

    def test_Mesh2MeshMap(self):
        model = Maps.Mesh2Mesh([self.mesh22, self.mesh2])
        self.assertTrue(model.test())

    def test_comboMaps(self):
        combos = [(Maps.ExpMap, Maps.Vertical1DMap)]
        for combo in combos:
            model = Maps.ComboMap(self.mesh2, combo)
            self.assertTrue(model.test())


if __name__ == '__main__':
    unittest.main()
