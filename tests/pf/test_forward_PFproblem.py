import unittest
from SimPEG import Mesh, Utils, PF
import numpy as np


class MagFwdProblemTests(unittest.TestCase):

    def setUp(self):

        cs = 25.
        hxind = [(cs, 5, -1.3), (cs/2.0, 41), (cs, 5, 1.3)]
        hyind = [(cs, 5, -1.3), (cs/2.0, 41), (cs, 5, 1.3)]
        hzind = [(cs, 5, -1.3), (cs/2.0, 40), (cs, 5, 1.3)]
        M = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

        chibkg = 0.
        chiblk = 0.01
        chi = np.ones(M.nC)*chibkg
        sph_ind = PF.MagAnalytics.spheremodel(M, 0., 0., 0., 100)
        chi[sph_ind] = chiblk
        model = PF.BaseMag.BaseMagMap(M)
        prob = PF.Magnetics.Problem3D_DiffSecondary(M, muMap=model)
        self.prob = prob
        self.M = M
        self.chi = chi

    def test_ana_forward(self):

        survey = PF.BaseMag.BaseMagSurvey()

        Inc = 45.
        Dec = 45.
        Btot = 51000

        b0 = PF.MagAnalytics.IDTtoxyz(Inc, Dec, Btot)
        survey.setBackgroundField(Inc, Dec, Btot)
        xr = np.linspace(-300, 300, 41)
        yr = np.linspace(-300, 300, 41)
        X, Y = np.meshgrid(xr, yr)
        Z = np.ones((xr.size, yr.size))*150
        rxLoc = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]
        survey.rxLoc = rxLoc

        self.prob.pair(survey)
        u = self.prob.fields(self.chi)
        B = u['B']

        bxa, bya, bza = PF.MagAnalytics.MagSphereAnaFunA(rxLoc[:, 0], rxLoc[:, 1], rxLoc[:, 2], 100., 0., 0., 0., 0.01, b0, 'secondary')

        dpred = survey.projectFieldsAsVector(B)
        err = np.linalg.norm(dpred-np.r_[bxa, bya, bza])/np.linalg.norm(np.r_[bxa, bya, bza])
        self.assertTrue(err < 0.08)


if __name__ == '__main__':
    unittest.main()
