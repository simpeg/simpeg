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

    def test_mapMultiplication(self):
        M = Mesh.TensorMesh([2,3])
        expMap = Maps.ExpMap(M)
        vertMap = Maps.Vertical1DMap(M)
        combo = expMap*vertMap
        m = np.arange(3.0)
        t_true = np.exp(np.r_[0,0,1,1,2,2.])
        self.assertLess(np.linalg.norm((combo * m)-t_true,np.inf),TOL)
        self.assertLess(np.linalg.norm((expMap * vertMap * m)-t_true,np.inf),TOL)
        self.assertLess(np.linalg.norm(expMap * (vertMap * m)-t_true,np.inf),TOL)
        self.assertLess(np.linalg.norm((expMap * vertMap) * m-t_true,np.inf),TOL)
        #Try making a model
        mod = Models.Model(m, mapping=combo)
        # print mod.transform
        # import matplotlib.pyplot as plt
        # plt.colorbar(M.plotImage(mod.transform)[0])
        # plt.show()
        self.assertLess(np.linalg.norm(mod.transform-t_true,np.inf),TOL)

        self.assertRaises(Exception,Models.Model,np.r_[1.0],mapping=combo)

        self.assertRaises(ValueError, lambda: combo * (vertMap * expMap))
        self.assertRaises(ValueError, lambda: (combo * vertMap) * expMap)
        self.assertRaises(ValueError, lambda: vertMap * expMap)
        self.assertRaises(ValueError, lambda: expMap * np.ones(100))
        self.assertRaises(ValueError, lambda: expMap * np.ones((100.0,1)))
        self.assertRaises(ValueError, lambda: expMap * np.ones((100.0,5)))
        self.assertRaises(ValueError, lambda: combo * np.ones(100))
        self.assertRaises(ValueError, lambda: combo * np.ones((100.0,1)))
        self.assertRaises(ValueError, lambda: combo * np.ones((100.0,5)))

    def test_activeCells(self):
        M = Mesh.TensorMesh([2,4],'0C')
        expMap = Maps.ExpMap(M)
        actMap = Maps.ActiveCells(M, M.vectorCCy <=0, 10, nC=M.nCy)
        vertMap = Maps.Vertical1DMap(M)
        combo = vertMap * actMap
        m = np.r_[1,2.]
        mod = Models.Model(m,combo)
        # import matplotlib.pyplot as plt
        # plt.colorbar(M.plotImage(mod.transform)[0])
        # plt.show()
        self.assertLess(np.linalg.norm(mod.transform - np.r_[1,1,2,2,10,10,10,10.]), TOL)
        self.assertLess((mod.transformDeriv - combo.deriv(m)).toarray().sum(), TOL)

    def test_tripleMultiply(self):
        M = Mesh.TensorMesh([2,4],'0C')
        expMap = Maps.ExpMap(M)
        vertMap = Maps.Vertical1DMap(M)
        actMap = Maps.ActiveCells(M, M.vectorCCy <=0, 10, nC=M.nCy)
        m = np.r_[1,2.]
        t_true = np.exp(np.r_[1,1,2,2,10,10,10,10.])
        self.assertLess(np.linalg.norm((expMap * vertMap * actMap * m)-t_true,np.inf),TOL)
        self.assertLess(np.linalg.norm(((expMap * vertMap * actMap) * m)-t_true,np.inf),TOL)
        self.assertLess(np.linalg.norm((expMap * vertMap * (actMap * m))-t_true,np.inf),TOL)
        self.assertLess(np.linalg.norm((expMap * (vertMap * actMap) * m)-t_true,np.inf),TOL)
        self.assertLess(np.linalg.norm(((expMap * vertMap) * actMap * m)-t_true,np.inf),TOL)

        self.assertRaises(ValueError, lambda: expMap * actMap * vertMap )
        self.assertRaises(ValueError, lambda: actMap * vertMap * expMap )


if __name__ == '__main__':
    unittest.main()
