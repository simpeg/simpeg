import unittest
from SimPEG import Mesh, Utils, PF, Maps, Problem, Survey, mkvc
import numpy as np
import matplotlib.pyplot as plt


class MagFwdProblemTests(unittest.TestCase):

    def setUp(self):

        # Define inducing field and sphere parameters
        H0 = (50000., 60., 270.)
        self.b0 = PF.MagAnalytics.IDTtoxyz(-H0[1], H0[2], H0[0])
        self.rad = 2.
        self.chi = 0.01

        # Define a mesh
        cs = 0.2
        hxind = [(cs, 21)]
        hyind = [(cs, 21)]
        hzind = [(cs, 21)]
        mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

        # Get cells inside the sphere
        sph_ind = PF.MagAnalytics.spheremodel(mesh, 0., 0., 0., self.rad)

        # Adjust susceptibility for volume difference
        Vratio = (4./3.*np.pi*self.rad**3.) / (np.sum(sph_ind)*cs**3.)
        model = np.ones(mesh.nC)*self.chi*Vratio
        self.model = model[sph_ind]

        # Creat reduced identity map for Linear Pproblem
        idenMap = Maps.IdentityMap(nP=int(sum(sph_ind)))

        # Create plane of observations
        xr = np.linspace(-20, 20, 21)
        yr = np.linspace(-20, 20, 21)
        X, Y = np.meshgrid(xr, yr)

        # Move obs plane 2 radius away from sphere
        Z = np.ones((xr.size, yr.size))*2.*self.rad
        self.locXyz = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]
        rxLoc = PF.BaseMag.RxObs(self.locXyz)
        srcField = PF.BaseMag.SrcField([rxLoc], param=H0)
        self.survey = PF.BaseMag.LinearSurvey(srcField)

        self.prob_xyz = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap,
                                                      actInd=sph_ind,
                                                      forwardOnly=True,
                                                      rtype='xyz')

        self.prob_tmi = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap,
                                                      actInd=sph_ind,
                                                      forwardOnly=True,
                                                      rtype='tmi')

    def test_ana_forward(self):

        # Compute 3-component mag data
        self.survey.pair(self.prob_xyz)
        d = self.prob_xyz.fields(self.model)

        ndata = self.locXyz.shape[0]
        dbx = d[0:ndata]
        dby = d[ndata:2*ndata]
        dbz = d[2*ndata:]

        # Compute tmi mag data
        self.survey.unpair()
        self.survey.pair(self.prob_tmi)
        dtmi = self.prob_tmi.fields(self.model)

        # Compute analytical response from a magnetized sphere
        bxa, bya, bza = PF.MagAnalytics.MagSphereFreeSpace(self.locXyz[:, 0],
                                                           self.locXyz[:, 1],
                                                           self.locXyz[:, 2],
                                                           self.rad, 0, 0, 0,
                                                           self.chi, self.b0)

        # Projection matrix
        Ptmi = mkvc(self.b0)/np.sqrt(np.sum(self.b0**2.))

        btmi = mkvc(Ptmi.dot(np.vstack((bxa, bya, bza))))

        err_xyz = (np.linalg.norm(d-np.r_[bxa, bya, bza]) /
                   np.linalg.norm(np.r_[bxa, bya, bza]))

        err_tmi = np.linalg.norm(dtmi-btmi)/np.linalg.norm(btmi)
        self.assertTrue(err_xyz < 0.005 and err_tmi < 0.005)


if __name__ == '__main__':
    unittest.main()
