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

    def test_transforms(self):
        for M in dir(Maps):
            try:
                maps = getattr(Maps, M)(self.mesh2)
                assert isinstance(maps, Maps.BaseModel)
            except Exception, e:
                continue
            self.assertTrue(maps.test())

    def test_Mesh2MeshMap(self):
        maps = Maps.Mesh2Mesh([self.mesh22, self.mesh2])
        self.assertTrue(maps.test())

    def test_comboMaps(self):
        combos = [(Maps.ExpMap, Maps.Vertical1DMap)]
        for combo in combos:
            maps = Maps.ComboMap(self.mesh2, combo)
            self.assertTrue(maps.test())


if __name__ == '__main__':
    unittest.main()
