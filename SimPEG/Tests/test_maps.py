import numpy as np
import unittest
from SimPEG import *
from TestUtils import checkDerivative
from scipy.sparse.linalg import dsolve

TOL = 1e-14

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
                assert isinstance(maps, Maps.IdentityMap)
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

    def test_mapMultiplication(self):
        M = Mesh.TensorMesh([2,3])
        expMap = Maps.ExpMap(M)
        vertMap = Maps.Vertical1DMap(M)
        combo = expMap*vertMap
        m = np.arange(3.0)
        t = combo * m
        t_true = np.exp(np.r_[0,0,1,1,2,2.])
        self.assertLess(np.linalg.norm(t-t_true,np.inf),TOL)
        #Try making a model
        mod = Maps.Model(m,mapping=combo)
        # print mod.transform
        # import matplotlib.pyplot as plt
        # plt.colorbar(M.plotImage(mod.transform)[0])
        # plt.show()
        self.assertLess(np.linalg.norm(mod.transform-t_true,np.inf),TOL)

        self.assertTrue(mod.test(plotIt=False))

        self.assertRaises(Exception,Maps.Model,np.r_[1.0],mapping=combo)

    def test_activeCells(self):
        M = Mesh.TensorMesh([2,4],'0C')
        actMap = Maps.ActiveCells(M, M.vectorCCy <=0, 10, nC=M.nCy)
        vertMap = Maps.Vertical1DMap(M)
        mod = Maps.Model(np.r_[1,2.],vertMap * actMap)
        # import matplotlib.pyplot as plt
        # plt.colorbar(M.plotImage(mod.transform)[0])
        # plt.show()
        self.assertLess(np.linalg.norm(mod.transform - np.r_[1,1,2,2,10,10,10,10.]), TOL)
        self.assertTrue(mod.test())


if __name__ == '__main__':
    unittest.main()
