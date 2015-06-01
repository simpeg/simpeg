import unittest
from SimPEG import *



class MyPropMap(Maps.PropMap):
    sigma = Maps.Property("Electrical Conductivity", defaultInvProp=True)
    mu    = Maps.Property("Electrical Conductivity", defaultVal=4e-10)


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

            m = PM(np.r_[1,2,3])
            assert m.mu == 4e-10
            assert m.muModel is None
            assert m.muMap is None
            assert m.muDeriv is None

            assert np.all(m.sigmaModel == np.r_[1,2,3])
            assert m.sigmaMap is expMap
            assert np.all(m.sigma == np.exp(np.r_[1,2,3]))
            assert m.sigmaDeriv is not None

    def test_slices(self):
        expMap = Maps.ExpMap(Mesh.TensorMesh((3,)))
        PM = MyPropMap({'maps':[('sigma', expMap)], 'slices':{'sigma':[2,1,0]}})
        assert PM.sigmaIndex == [2,1,0]
        m = PM(np.r_[1,2,3])
        assert np.all(m.sigmaModel == np.r_[3,2,1])
        assert np.all(m.sigma == np.exp(np.r_[3,2,1]))



if __name__ == '__main__':
    unittest.main()


