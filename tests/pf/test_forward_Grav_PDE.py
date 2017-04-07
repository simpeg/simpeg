import unittest
from SimPEG import Mesh, Utils, PF, Maps, Problem, Survey, mkvc
import numpy as np
import matplotlib.pyplot as plt


class GravFwdProblemTests(unittest.TestCase):

    def setUp(self):

        # Define sphere parameters
        self.rad = 10.
        self.rho = 0.1

        # Define a mesh
        cs = 1.
        nC = 21

        hxind = [(cs, 15, -1.3), (cs, nC), (cs, 15, 1.3)]
        hyind = [(cs, 15, -1.3), (cs, nC), (cs, 15, 1.3)]
        hzind = [(cs, 15, -1.3), (cs, 2*nC), (cs, 15, 1.3)]
        mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

        # Get cells inside the sphere
        sph_ind = PF.MagAnalytics.spheremodel(mesh, 0., 0., 0., self.rad)

        # Adjust density for volume difference
        Vratio = (4./3.*np.pi*self.rad**3.) / (np.sum(sph_ind)*cs**3.)
        self.model = np.zeros(mesh.nC)
        self.model[sph_ind] = self.rho*Vratio

        # Create reduced identity map for Linear Pproblem
        idenMap = Maps.IdentityMap(nP=int(sum(sph_ind)))

        # Create plane of observations
        xr = np.linspace(-cs*nC/2., cs*nC/2., 11)
        yr = np.linspace(-cs*nC/2., cs*nC/2., 11)
        X, Y = np.meshgrid(xr, yr)

        # Move obs plane 2 radius away from sphere
        Z = np.ones((xr.size, yr.size))*2.*self.rad
        self.locXyz = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]

        # Rho map
        m_rho = PF.BaseGrav.BaseGravMap(mesh)

        # Grav PDE problem
        self.prob = PF.Gravity.Problem3D_PDE(mesh, rhoMap=m_rho)

        rxLoc = PF.BaseGrav.RxObs(self.locXyz)
        srcField = PF.BaseGrav.SrcField([rxLoc])
        self.survey = PF.BaseGrav.LinearSurvey(srcField)

    def test_ana_forward(self):

        # Compute gravity fields
        u = self.prob.fields(self.model)

        # Pair problem and survey
        self.prob.pair(self.survey)

        # Get data grav and grav.grad.
        d = self.survey.projectFields(u)

        # Compute analytical response from mass sphere
        AnaSphere = PF.GravAnalytics.GravSphereFreeSpace(self.locXyz[:, 0],
                                                         self.locXyz[:, 1],
                                                         self.locXyz[:, 2],
                                                         self.rad, 0, 0, 0.,
                                                         self.rho)

        # Compute residual on all components of gravity and grav.grad. and
        # check is residual is smaller than treshold on all
        var = 0
        for key in d:

            res = (np.linalg.norm(d[key] - AnaSphere[key]) /
                   np.linalg.norm(AnaSphere[key]))

            if res < 0.02:
                var += 1

        self.assertTrue(var == 9)


if __name__ == '__main__':
    unittest.main()
