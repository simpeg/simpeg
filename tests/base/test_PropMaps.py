from __future__ import print_function
import unittest
import numpy as np
from SimPEG import Mesh, Maps
from scipy.constants import mu_0
from SimPEG import Tests


class MyPropMap(Maps.PropMap):
    sigma = Maps.Property("Electrical Conductivity", defaultInvProp=True)
    mu    = Maps.Property("Mu", defaultVal=mu_0)

class MyReciprocalPropMap(Maps.PropMap):
    sigma  = Maps.Property("Electrical Conductivity", defaultInvProp=True, propertyLink=('rho',   Maps.ReciprocalMap))
    rho    = Maps.Property("Electrical Resistivity",                       propertyLink=('sigma', Maps.ReciprocalMap))
    mu     = Maps.Property("Mu", defaultVal=mu_0,    propertyLink=('mui',  Maps.ReciprocalMap))
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

            assert 'sigma' in PM
            assert 'mu' not in PM
            assert 'mui' not in PM

            m = PM(np.r_[1.,2,3])

            assert 'sigma' in m
            assert 'mu' not in m
            assert 'mui' not in m

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

        assert 'sigma' in PM
        assert 'mu' in PM
        assert 'mui' not in PM

        assert 'sigma' in pm
        assert 'mu' in pm
        assert 'mui' not in pm

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

        assert 'sigma' in PM
        assert 'mu' in PM
        assert 'mui' not in PM

        assert 'sigma' in pm
        assert 'mu' in pm
        assert 'mui' not in pm

        assert np.all(pm.sigmaModel == [1,2,3])
        assert np.all(pm.sigma == np.exp([1,2,3]))
        assert np.all(pm.muModel == [1,2,3])
        assert np.all(pm.mu == [1,2,3])

    def test_Projections(self):
        m = Mesh.TensorMesh((3,))
        iMap = Maps.IdentityMap(m)
        PM = MyReciprocalPropMap([('sigma', iMap)])
        v = np.r_[1,2.,3]
        pm = PM(v)

        assert pm.sigmaProj is not None
        assert pm.rhoProj   is None
        assert pm.muProj    is None
        assert pm.muiProj   is None

        assert np.all(pm.sigmaProj * v == pm.sigmaModel)

    def test_Links(self):
        m = Mesh.TensorMesh((3,))
        expMap = Maps.ExpMap(m)
        iMap = Maps.IdentityMap(m)
        PM = MyReciprocalPropMap([('sigma', iMap)])
        pm = PM(np.r_[1,2.,3])
        # print(pm.sigma)
        # print(pm.sigmaMap)
        assert np.all(pm.sigma == [1,2,3])
        assert np.all(pm.rho == 1./np.r_[1,2,3])
        assert pm.sigmaMap is iMap
        assert pm.rhoMap is None
        assert pm.sigmaDeriv is not None
        assert pm.rhoDeriv is not None

        assert 'sigma' in PM
        assert 'rho' not in PM
        assert 'mu' not in PM
        assert 'mui' not in PM


        assert 'sigma' in pm
        assert 'rho' not in pm
        assert 'mu' not in pm
        assert 'mui' not in pm

        assert pm.mu == mu_0
        assert pm.mui == 1.0/mu_0
        assert pm.muMap is None
        assert pm.muDeriv is None
        assert pm.muiMap is None
        assert pm.muiDeriv is None

        PM = MyReciprocalPropMap([('rho', iMap)])
        pm = PM(np.r_[1,2.,3])
        # print(pm.sigma)
        # print(pm.sigmaMap)
        assert np.all(pm.sigma == 1./np.r_[1,2,3])
        assert np.all(pm.rho == [1,2,3])
        assert pm.sigmaMap is None
        assert pm.rhoMap is iMap
        assert pm.sigmaDeriv is not None
        assert pm.rhoDeriv is not None

        assert 'sigma' not in PM
        assert 'rho' in PM
        assert 'mu' not in PM
        assert 'mui' not in PM


        assert 'sigma' not in pm
        assert 'rho' in pm
        assert 'mu' not in pm
        assert 'mui' not in pm

        self.assertRaises(AssertionError, MyReciprocalPropMap, [('rho', iMap), ('sigma', iMap)])
        self.assertRaises(AssertionError, MyReciprocalPropMap, [('sigma', iMap), ('rho', iMap)])

        MyReciprocalPropMap([('sigma', iMap), ('mu', iMap)]) # This should be fine

    def test_linked_derivs_sigma(self):
        mesh = Mesh.TensorMesh([4,5], x0='CC')

        mapping = Maps.ExpMap(mesh)
        propmap = MyReciprocalPropMap([('rho', mapping)])

        x0 = np.random.rand(mesh.nC)
        m  = propmap(x0)

        # test Sigma
        testme = lambda v: [1./(m.rhoMap*v), m.sigmaDeriv]
        print('Testing Rho from Sigma')
        Tests.checkDerivative(testme, x0, dx=0.01*x0, num=5, plotIt=False)

    def test_linked_derivs_rho(self):
        mesh = Mesh.TensorMesh([4,5], x0='CC')

        mapping = Maps.ExpMap(mesh)
        propmap = MyReciprocalPropMap([('sigma', mapping)])

        x0 = np.random.rand(mesh.nC)
        m  = propmap(x0)

        # test Sigma
        testme = lambda v: [1./(m.sigmaMap*v), m.rhoDeriv]
        print('Testing Rho from Sigma')
        Tests.checkDerivative(testme, x0, dx=0.01*x0, num=5, plotIt=False)

if __name__ == '__main__':
    unittest.main()


