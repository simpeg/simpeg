import unittest
from SimPEG import *
from scipy.constants import mu_0


class MyPropMap(Maps.PropMap):
    sigma = Maps.Property("Electrical Conductivity", defaultInvProp=True)
    mu    = Maps.Property("Mu", defaultVal=mu_0)

class MyReciprocalPropMap(Maps.PropMap):
    sigma = Maps.Property("Electrical Conductivity", defaultInvProp=True, propertyLink=('rho',   Maps.ReciprocalMap))
    rho   = Maps.Property("Electrical Resistivity",                       propertyLink=('sigma', Maps.ReciprocalMap))
    mu    = Maps.Property("Mu", defaultVal=mu_0, propertyLink=('mui',   Maps.ReciprocalMap))
    mui    = Maps.Property("Mu", defaultVal=1./mu_0, propertyLink=('mu',   Maps.ReciprocalMap))


class TestPropMaps(unittest.TestCase):

    def setUp(self):
        pass

    def test_setup(self):
        expMap = Maps.ExpMap(Mesh.TensorMesh((3,)))
        assert expMap.nP == 3

        PM1 = MyPropMap(expMap)
        PM2 = MyPropMap([('sigma', expMap)])
        PM3 = MyPropMap({'maps':[('sigma', expMap)], 'slices':{'sigma':slice(0,3)}})

        for PM in [PM1,PM2,PM3]:
            assert PM.defaultInvProp == 'sigma'
            assert PM.sigmaMap is not None
            assert PM.sigmaMap is expMap
            assert PM.sigmaIndex == slice(0,3)
            assert getattr(PM, 'sigma', None) is None
            assert PM.muMap is None
            assert PM.muIndex is None

            m = PM(np.r_[1.,2,3])
            assert m.mu == mu_0
            assert m.muModel is None
            assert m.muMap is None
            assert m.muDeriv is None

            assert np.all(m.sigmaModel == np.r_[1.,2,3])
            assert m.sigmaMap is expMap
            assert np.all(m.sigma == np.exp(np.r_[1.,2,3]))
            assert m.sigmaDeriv is not None

            assert m.nP == 3

    def test_slices(self):
        expMap = Maps.ExpMap(Mesh.TensorMesh((3,)))
        PM = MyPropMap({'maps':[('sigma', expMap)], 'slices':{'sigma':[2,1,0]}})
        assert PM.sigmaIndex == [2,1,0]
        m = PM(np.r_[1.,2,3])
        assert np.all(m.sigmaModel == np.r_[3,2,1])
        assert np.all(m.sigma == np.exp(np.r_[3,2,1]))

    def test_multiMap(self):
        m = Mesh.TensorMesh((3,))
        expMap = Maps.ExpMap(m)
        iMap = Maps.IdentityMap(m)
        PM = MyPropMap([('sigma', expMap), ('mu', iMap)])

        pm = PM(np.r_[1.,2,3,4,5,6])

        assert pm.nP == 6

        assert np.all(pm.sigmaModel == [1.,2,3])
        assert np.all(pm.sigma == np.exp([1.,2,3]))
        assert np.all(pm.muModel == [4.,5,6])
        assert np.all(pm.mu == [4.,5,6])


    def test_multiMapCompressed(self):
        m = Mesh.TensorMesh((3,))
        expMap = Maps.ExpMap(m)
        iMap = Maps.IdentityMap(m)
        PM = MyPropMap({'maps':[('sigma', expMap), ('mu', iMap)],'slices':{'mu':[0,1,2]}})

        pm = PM(np.r_[1,2.,3])

        assert pm.nP == 3

        assert np.all(pm.sigmaModel == [1,2,3])
        assert np.all(pm.sigma == np.exp([1,2,3]))
        assert np.all(pm.muModel == [1,2,3])
        assert np.all(pm.mu == [1,2,3])

    def test_Links(self):
        m = Mesh.TensorMesh((3,))
        expMap = Maps.ExpMap(m)
        iMap = Maps.IdentityMap(m)
        PM = MyReciprocalPropMap([('sigma', iMap)])
        pm = PM(np.r_[1,2.,3])
        # print pm.sigma
        # print pm.sigmaMap
        assert np.all(pm.sigma == [1,2,3])
        assert np.all(pm.rho == 1./np.r_[1,2,3])
        assert pm.sigmaMap is iMap
        assert pm.rhoMap is None
        assert pm.sigmaDeriv is not None
        assert pm.rhoDeriv is not None

        assert pm.mu == mu_0
        assert pm.mui == 1.0/mu_0
        assert pm.muMap is None
        assert pm.muDeriv is None
        assert pm.muiMap is None
        assert pm.muiDeriv is None

        PM = MyReciprocalPropMap([('rho', iMap)])
        pm = PM(np.r_[1,2.,3])
        # print pm.sigma
        # print pm.sigmaMap
        assert np.all(pm.sigma == 1./np.r_[1,2,3])
        assert np.all(pm.rho == [1,2,3])
        assert pm.sigmaMap is None
        assert pm.rhoMap is iMap
        assert pm.sigmaDeriv is not None
        assert pm.rhoDeriv is not None

        self.assertRaises(AssertionError, MyReciprocalPropMap, [('rho', iMap), ('sigma', iMap)])
        self.assertRaises(AssertionError, MyReciprocalPropMap, [('sigma', iMap), ('rho', iMap)])

        MyReciprocalPropMap([('sigma', iMap), ('mu', iMap)]) # This should be fine

if __name__ == '__main__':
    unittest.main()


