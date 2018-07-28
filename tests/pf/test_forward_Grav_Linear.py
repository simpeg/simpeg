import unittest
from SimPEG import Mesh, Utils, PF, Maps
import numpy as np


class GravFwdProblemTests(unittest.TestCase):

    def setUp(self):

        # Define sphere parameters
        self.rad = 2.
        self.rho = 0.1

        # Define a mesh
        cs = 0.2
        hxind = [(cs, 21)]
        hyind = [(cs, 21)]
        hzind = [(cs, 21)]
        mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

        # Get cells inside the sphere
        sph_ind = PF.MagAnalytics.spheremodel(mesh, 0., 0., 0., self.rad)

        # Adjust density for volume difference
        Vratio = (4./3.*np.pi*self.rad**3.) / (np.sum(sph_ind)*cs**3.)
        model = np.ones(mesh.nC)*self.rho*Vratio
        self.model = model[sph_ind]

        # Create reduced identity map for Linear Pproblem
        idenMap = Maps.IdentityMap(nP=int(sum(sph_ind)))

        # Create plane of observations
        xr = np.linspace(-20, 20, 21)
        yr = np.linspace(-20, 20, 21)
        X, Y = np.meshgrid(xr, yr)

        # Move obs plane 2 radius away from sphere
        Z = np.ones((xr.size, yr.size))*2.*self.rad
        self.locXyz = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]
        rxLoc = PF.BaseGrav.RxObs(self.locXyz)
        srcField = PF.BaseGrav.SrcField([rxLoc])
        self.survey = PF.BaseGrav.LinearSurvey(srcField)

        self.prob_x = PF.Gravity.GravityIntegral(mesh, rhoMap=idenMap,
                                                 actInd=sph_ind,
                                                 forwardOnly=True,
                                                 rxType='x')

        self.prob_y = PF.Gravity.GravityIntegral(mesh, rhoMap=idenMap,
                                                 actInd=sph_ind,
                                                 forwardOnly=True,
                                                 rxType='y')

        self.prob_z = PF.Gravity.GravityIntegral(mesh, rhoMap=idenMap,
                                                 actInd=sph_ind,
                                                 forwardOnly=True,
                                                 rxType='z')

    def test_ana_forward(self):

        # Compute gz data only
        self.survey.pair(self.prob_x)
        data_x = self.prob_x.fields(self.model)

        # Compute gz data only
        self.survey.unpair()
        self.survey.pair(self.prob_y)
        data_y = self.prob_y.fields(self.model)

        # Compute gz data only
        self.survey.unpair()
        self.survey.pair(self.prob_z)
        data_z = self.prob_z.fields(self.model)

        # Compute analytical response from mass sphere
        gxa, gya, gza = PF.GravAnalytics.GravSphereFreeSpace(self.locXyz[:, 0],
                                                             self.locXyz[:, 1],
                                                             self.locXyz[:, 2],
                                                             self.rad, 0, 0, 0,
                                                             self.rho)

        # Compute residual
        err_xyz = (np.linalg.norm(
            np.r_[data_x, data_y, data_z] -
            np.r_[gxa, gya, gza]) /
                   np.linalg.norm(np.r_[gxa, gya, gza])
        )

        err_tmi = np.linalg.norm(data_z-gza)/np.linalg.norm(gza)

        self.assertTrue(err_xyz < 0.005 and err_tmi < 0.005)


if __name__ == '__main__':
    unittest.main()
