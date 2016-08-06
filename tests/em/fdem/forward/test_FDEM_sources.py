import unittest
from SimPEG.EM import FDEM, Analytics, mu_0
import numpy as np
from SimPEG import Mesh, Maps, Utils
import warnings

TOL = 0.5 # relative tolerance (to norm of soln)
plotIt = False

if plotIt is True:
    import matplotlib.pyplot as plt


class TestSimpleSourcePropertiesTensor(unittest.TestCase):

    def setUp(self):
        cs = 20.
        ncx, ncy, ncz = 80., 80., 80.
        npad = 20.
        hx = [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        self.mesh = Mesh.TensorMesh([hx, hy, hz], 'CCC')
        mapping = Maps.ExpMap(self.mesh)

        self.freq = 1.

        self.prob_e = FDEM.Problem3D_e(self.mesh, mapping=mapping)
        self.prob_b = FDEM.Problem3D_b(self.mesh, mapping=mapping)
        self.prob_h = FDEM.Problem3D_h(self.mesh, mapping=mapping)
        self.prob_j = FDEM.Problem3D_j(self.mesh, mapping=mapping)


    def test_MagDipole(self):

        print('\ntesting MagDipole assignments')


        for orient in ['x', 'y', 'z', 'X', 'Y', 'Z']:
            src = FDEM.Src.MagDipole([], freq=self.freq, loc=np.r_[0.,0.,0.], orientation=orient)

            # test assignments
            assert np.all(src.loc == np.r_[0.,0.,0.])
            assert src.freq == self.freq

            if orient.upper() == 'X':
                orient_vec = np.r_[1., 0., 0.]
            elif orient.upper() == 'Y':
                orient_vec = np.r_[0., 1., 0.]
            elif orient.upper() == 'Z':
                orient_vec = np.r_[0., 0., 1.]

            print ('  {0} component. src: {1}, expected: {2}'.format(orient, src.orientation, orient_vec))
            assert np.all(src.orientation == orient_vec)

    def test_MagDipoleSimpleFail(self):

        print('\ntesting MagDipole error handling')
        with self.assertRaises(Exception):
            FDEM.Src.MagDipole([], freq=self.freq, loc=np.r_[0.,0.,0.], orientation=np.r_[2.,3.,2.])

        with warnings.catch_warnings(record=True):
            FDEM.Src.MagDipole([], freq=self.freq, loc=np.r_[0.,0.,0.], orientation=np.r_[1.,0.,0.])

    def test_bPrimary(self):
        print('\ntesting bPrimary')
        loc = np.r_[0.,0.,0.]
        loc = Utils.mkvc(self.mesh.gridCC[Utils.closestPoints(self.mesh, loc, 'CC'),:])
        src = FDEM.Src.MagDipole([], freq=self.freq, loc=loc, orientation='Z', mu=mu_0)

        for solType in ['e', 'b', 'h', 'j']:
            prob = getattr(self, 'prob_{}'.format(solType))

            bPrimary = src.bPrimary(prob)

            def ana_sol(XYZ):
                return Analytics.FDEM.MagneticDipoleWholeSpace(XYZ, src.loc, 0., 0., moment=1., orientation='Z', mu=src.mu)


            if solType in ['e', 'b']:
                # TODO: clean up how we call analytics
                bx, _, _ = ana_sol(self.mesh.gridFx)
                _, by, _ = ana_sol(self.mesh.gridFy)
                _, _, bz = ana_sol(self.mesh.gridFz)

                # remove the z faces right next to the source
                ignore_these = ((-self.mesh.hx.min()+src.loc[0] <= self.mesh.gridFz[:,0]) & (self.mesh.gridFz[:,0] <= self.mesh.hx.min()+src.loc[0]) &
                                (-self.mesh.hy.min()+src.loc[1] <= self.mesh.gridFz[:,1]) & (self.mesh.gridFz[:,1] <= self.mesh.hy.min()+src.loc[1]) &
                                (-self.mesh.hz.min()+src.loc[2] <= self.mesh.gridFz[:,2]) & (self.mesh.gridFz[:,2] <= self.mesh.hz.min()+src.loc[2]))

                look_at_these = np.ones(self.mesh.nFx + self.mesh.nFy, dtype=bool)

            elif solType in ['h', 'j']:
                # TODO: clean up how we call analytics
                bx, _, _ = ana_sol(self.mesh.gridEx)
                _, by, _ = ana_sol(self.mesh.gridEy)
                _, _, bz = ana_sol(self.mesh.gridEz)

                # remove the z faces right next to the source
                ignore_these = ((-self.mesh.hx.min()+src.loc[0] <= self.mesh.gridEz[:,0]) & (self.mesh.gridEz[:,0] <= self.mesh.hx.min()+src.loc[0]) &
                                (-self.mesh.hy.min()+src.loc[1] <= self.mesh.gridEz[:,1]) & (self.mesh.gridEz[:,1] <= self.mesh.hy.min()+src.loc[1]) &
                                (-self.mesh.hz.min()+src.loc[2] <= self.mesh.gridEz[:,2]) & (self.mesh.gridEz[:,2] <= self.mesh.hz.min()+src.loc[2]))

                look_at_these = np.ones(self.mesh.nEx + self.mesh.nEy, dtype=bool)

            look_at_these = np.hstack([look_at_these, np.array(ignore_these==False, dtype=bool)])
            bPrimary_ana = Utils.mkvc(np.vstack([bx, by, bz]))
            bPrimary = bPrimary

            check = np.linalg.norm(bPrimary[look_at_these] - bPrimary_ana[look_at_these], np.inf)
            tol = np.linalg.norm(bPrimary[look_at_these]) * TOL
            passed = check < tol

            print('  {}, num: {}, ana: {}, num - ana: {}, < {} ? {}'.format(
                  solType,
                  np.linalg.norm(bPrimary[look_at_these], np.inf),
                  np.linalg.norm(bPrimary_ana[look_at_these], np.inf),
                  check,
                  tol,
                  passed))

            if plotIt is True:

                print self.mesh.vnF

                fig, ax = plt.subplots(1,2)
                ax[0].semilogy(np.absolute(bPrimary))
                ax[0].semilogy(np.absolute(bPrimary_ana))
                ax[0].legend(['|num|', '|ana|'])
                ax[0].set_ylim([tol, bPrimary.max()*2])

                ax[1].semilogy(np.absolute(bPrimary-bPrimary_ana))
                ax[1].legend(['num - ana'])
                ax[1].set_ylim([tol, np.absolute(bPrimary-bPrimary_ana).max()*2])

                plt.show()

            assert passed

if __name__ == '__main__':
    unittest.main()
