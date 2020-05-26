import unittest
import discretize
from SimPEG import utils, maps
from SimPEG.utils.model_builder import getIndicesSphere
from SimPEG.potential_fields import magnetics as mag

import numpy as np

nx = 5
ny = 5


class MagFwdProblemTests(unittest.TestCase):
    def setUp(self):

        # Define inducing field and sphere parameters
        H0 = (50000.0, 60.0, 250.0)
        # H0 = (50000., 90., 0.)
        self.b0 = mag.analytics.IDTtoxyz(-H0[1], H0[2], H0[0])
        self.rad = 2.0
        self.chi = 0.01

        # Define a mesh
        cs = 0.2
        hxind = [(cs, 21)]
        hyind = [(cs, 21)]
        hzind = [(cs, 21)]
        mesh = discretize.TensorMesh([hxind, hyind, hzind], "CCC")

        # Get cells inside the sphere
        sph_ind = getIndicesSphere([0.0, 0.0, 0.0], self.rad, mesh.gridCC)

        # Adjust susceptibility for volume difference
        Vratio = (4.0 / 3.0 * np.pi * self.rad ** 3.0) / (np.sum(sph_ind) * cs ** 3.0)
        model = np.ones(mesh.nC) * self.chi * Vratio
        self.model = model[sph_ind]

        # Creat reduced identity map for Linear Pproblem
        idenMap = maps.IdentityMap(nP=int(sum(sph_ind)))

        # Create plane of observations
        xr = np.linspace(-20, 20, nx)
        yr = np.linspace(-20, 20, ny)
        self.xr = xr
        self.yr = yr
        X, Y = np.meshgrid(xr, yr)
        components = ["bx", "by", "bz", "tmi"]

        # Move obs plane 2 radius away from sphere
        Z = np.ones((xr.size, yr.size)) * 2.0 * self.rad
        self.locXyz = np.c_[utils.mkvc(X), utils.mkvc(Y), utils.mkvc(Z)]
        rxLoc = mag.Point(self.locXyz, components=components)
        srcField = mag.SourceField([rxLoc], parameters=H0)
        self.survey = mag.Survey(srcField)

        self.sim = mag.Simulation3DIntegral(
            mesh,
            survey=self.survey,
            chiMap=idenMap,
            actInd=sph_ind,
            store_sensitivities="forward_only",
        )

    def test_ana_forward(self):

        # Compute 3-component mag data
        data = self.sim.dpred(self.model)
        d_x = data[0::4]
        d_y = data[1::4]
        d_z = data[2::4]
        d_t = data[3::4]

        # Compute analytical response from a magnetized sphere
        bxa, bya, bza, btmi = mag.analytics.MagSphereFreeSpace(
            self.locXyz[:, 0],
            self.locXyz[:, 1],
            self.locXyz[:, 2],
            self.rad,
            0,
            0,
            0,
            self.chi,
            self.b0,
        )

        err_x = np.linalg.norm(d_x - bxa) / np.linalg.norm(bxa)
        err_y = np.linalg.norm(d_y - bya) / np.linalg.norm(bya)
        err_z = np.linalg.norm(d_z - bza) / np.linalg.norm(bza)
        err_t = np.linalg.norm(d_t - btmi) / np.linalg.norm(btmi)

        self.assertLess(err_x, 0.005)
        self.assertLess(err_y, 0.005)
        self.assertLess(err_z, 0.005)
        self.assertLess(err_t, 0.005)


if __name__ == "__main__":
    unittest.main()
