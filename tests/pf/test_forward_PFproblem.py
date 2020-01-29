import unittest
#from SimPEG import Mesh, Utils, PF
import discretize
from SimPEG import utils, maps
from SimPEG.utils.ModelBuilder import getIndicesSphere
from SimPEG.potential_fields import magnetics as mag
import numpy as np
from pymatsolver import Pardiso

import matplotlib.pyplot as plt


class MagFwdProblemTests(unittest.TestCase):

    def setUp(self):

        Inc = 45.
        Dec = 45.
        Btot = 51000
        H0 = (Btot, Inc, Dec)

        self.b0 = mag.analytics.IDTtoxyz(-Inc, Dec, Btot)

        cs = 25.
        hxind = [(cs, 5, -1.3), (cs/2.0, 41), (cs, 5, 1.3)]
        hyind = [(cs, 5, -1.3), (cs/2.0, 41), (cs, 5, 1.3)]
        hzind = [(cs, 5, -1.3), (cs/2.0, 40), (cs, 5, 1.3)]
        M = discretize.TensorMesh([hxind, hyind, hzind], 'CCC')

        chibkg = 0.
        chiblk = 0.01
        chi = np.ones(M.nC)*chibkg

        rad = 150
        sph_ind = getIndicesSphere([0., 0., 0.], rad, M.gridCC)
        chi[sph_ind] = chiblk

        xr = np.linspace(-300, 300, 41)
        yr = np.linspace(-300, 300, 41)
        X, Y = np.meshgrid(xr, yr)
        Z = np.ones((xr.size, yr.size))*150
        components = ['bx', 'by', 'bz']
        self.xr = xr
        self.yr = yr
        self.rxLoc = np.c_[utils.mkvc(X), utils.mkvc(Y), utils.mkvc(Z)]
        receivers = mag.point_receiver(self.rxLoc, components=components)
        srcField = mag.SourceField([receivers], parameters=H0)

        self.survey = mag.MagneticSurvey(srcField)

        #model = PF.BaseMag.BaseMagMap(M)
        #prob = PF.Magnetics.Problem3D_DiffSecondary(M, muMap=model)
        self.sim = mag.simulation.DifferentialEquationSimulation(
            M,
            survey=self.survey,
            muMap=maps.ChiMap(M),
            solver=Pardiso,
        )
        self.M = M
        self.chi = chi

    def test_ana_forward(self):

        u = self.sim.fields(self.chi)
        dpred = self.sim.projectFields(u)

        bxa, bya, bza = mag.analytics.MagSphereAnaFunA(
            self.rxLoc[:, 0], self.rxLoc[:, 1], self.rxLoc[:, 2],
            100., 0., 0., 0., 0.01, self.b0, 'secondary')

        n_obs, n_comp = self.rxLoc.shape[0], len(self.survey.components)
        dx, dy, dz = dpred.reshape(n_comp, n_obs)

        nx = len(self.xr)
        ny = len(self.yr)

        plt.figure()
        plt.subplot(1,3,1)
        plt.pcolormesh(self.xr, self.yr, dx.reshape(nx, ny).T)
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.pcolormesh(self.xr, self.yr, dy.reshape(nx, ny).T)
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.pcolormesh(self.xr, self.yr, dz.reshape(nx, ny).T)
        plt.colorbar()

        plt.figure()
        plt.subplot(1,3,1)
        plt.pcolormesh(self.xr, self.yr, bxa.reshape(nx, ny).T)
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.pcolormesh(self.xr, self.yr, bya.reshape(nx, ny).T)
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.pcolormesh(self.xr, self.yr, bza.reshape(nx, ny).T)
        plt.colorbar()

        plt.show()
        err = np.linalg.norm(dpred-np.r_[bxa, bya, bza])/np.linalg.norm(np.r_[bxa, bya, bza])
        self.assertTrue(err < 0.08)


if __name__ == '__main__':
    unittest.main()
