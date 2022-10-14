import unittest
import discretize
from SimPEG import utils, maps
from SimPEG.utils.model_builder import getIndicesSphere
from SimPEG.potential_fields import gravity
import numpy as np
import shutil

nx = 5
ny = 5


class GravFwdProblemTests(unittest.TestCase):
    def setUp(self):

        # Define sphere parameters
        self.rad = 2.0
        self.rho = 0.1

        # Define a mesh
        cs = 0.2
        hxind = [(cs, 21)]
        hyind = [(cs, 21)]
        hzind = [(cs, 21)]
        mesh = discretize.TensorMesh([hxind, hyind, hzind], "CCC")

        # Get cells inside the sphere
        sph_ind = getIndicesSphere([0.0, 0.0, 0.0], self.rad, mesh.cell_centers)

        # Adjust density for volume difference
        Vratio = (4.0 / 3.0 * np.pi * self.rad ** 3.0) / (np.sum(sph_ind) * cs ** 3.0)
        model = np.ones(mesh.nC) * self.rho * Vratio
        self.model = model[sph_ind]

        # Create reduced identity map for Linear Pproblem
        idenMap = maps.IdentityMap(nP=int(sum(sph_ind)))

        # Create plane of observations
        xr = np.linspace(-20, 20, nx)
        yr = np.linspace(-20, 20, ny)
        X, Y = np.meshgrid(xr, yr)
        self.xr = xr
        self.yr = yr

        components = ["gx", "gy", "gz"]

        # Move obs plane 3 radius away from sphere
        Z = np.ones((xr.size, yr.size)) * 3.0 * self.rad
        self.locXyz = np.c_[utils.mkvc(X), utils.mkvc(Y), utils.mkvc(Z)]
        receivers = gravity.Point(self.locXyz, components=components)
        sources = gravity.SourceField([receivers])
        self.survey = gravity.Survey(sources)

        self.sim = gravity.Simulation3DIntegral(
            mesh,
            survey=self.survey,
            rhoMap=idenMap,
            actInd=sph_ind,
            store_sensitivities="disk",
        )

    def test_ana_grav_forward(self):

        # Compute 3-component grav data

        data = self.sim.dpred(self.model)

        # Compute analytical response from mass sphere
        gxa, gya, gza = gravity.analytics.GravSphereFreeSpace(
            self.locXyz[:, 0],
            self.locXyz[:, 1],
            self.locXyz[:, 2],
            self.rad,
            0,
            0,
            0,
            self.rho,
        )

        d_x = data[0::3]
        d_y = data[1::3]
        d_z = data[2::3]
        # Compute residual
        err_x = np.linalg.norm(d_x - gxa) / np.linalg.norm(gxa)
        err_y = np.linalg.norm(d_y - gya) / np.linalg.norm(gya)
        err_z = np.linalg.norm(d_z - gza) / np.linalg.norm(gza)
        self.assertLess(err_x, 0.005)
        self.assertLess(err_y, 0.005)
        self.assertLess(err_z, 0.005)

    def tearDown(self):
        # Clean up the working directory
        try:
            if self.sim.store_sensitivities == "disk":
                shutil.rmtree(self.sim.sensitivity_path)
        except FileNotFoundError:
            pass


class GravityGradientFwdProblemTests(unittest.TestCase):
    def setUp(self):

        # Define sphere parameters
        self.rad = 2.0
        self.rho = 0.1

        # Define a mesh
        cs = 0.2
        hxind = [(cs, 21)]
        hyind = [(cs, 21)]
        hzind = [(cs, 21)]
        mesh = discretize.TensorMesh([hxind, hyind, hzind], "CCC")

        # Get cells inside the sphere
        sph_ind = getIndicesSphere([0.0, 0.0, 0.0], self.rad, mesh.cell_centers)

        # Adjust density for volume difference
        Vratio = (4.0 / 3.0 * np.pi * self.rad ** 3.0) / (np.sum(sph_ind) * cs ** 3.0)
        model = np.ones(mesh.nC) * self.rho * Vratio
        self.model = model[sph_ind]

        # Create reduced identity map for Linear Pproblem
        idenMap = maps.IdentityMap(nP=int(sum(sph_ind)))

        # Create plane of observations
        xr = np.linspace(-20, 20, nx)
        yr = np.linspace(-20, 20, ny)
        X, Y = np.meshgrid(xr, yr)
        self.xr = xr
        self.yr = yr

        components = ["gxx", "gxy", "gxz", "gyy", "gyz", "gzz"]

        # Move obs plane 2 radius away from sphere
        Z = np.ones((xr.size, yr.size)) * 3.0 * self.rad
        self.locXyz = np.c_[utils.mkvc(X), utils.mkvc(Y), utils.mkvc(Z)]
        receivers = gravity.Point(self.locXyz, components=components)
        sources = gravity.SourceField([receivers])
        self.survey = gravity.Survey(sources)

        self.sim = gravity.Simulation3DIntegral(
            mesh,
            survey=self.survey,
            rhoMap=idenMap,
            actInd=sph_ind,
            store_sensitivities="forward_only",
        )

    def test_ana_gg_forward(self):

        # Compute 3-component grav data
        data = self.sim.dpred(self.model)

        # Compute analytical response from mass sphere
        out = gravity.analytics.GravityGradientSphereFreeSpace(
            self.locXyz[:, 0],
            self.locXyz[:, 1],
            self.locXyz[:, 2],
            self.rad,
            0,
            0,
            0,
            self.rho,
        )
        gxxa, gxya, gxza, gyya, gyza, gzza = out

        d_xx = data[0::6]
        d_xy = data[1::6]
        d_xz = data[2::6]
        d_yy = data[3::6]
        d_yz = data[4::6]
        d_zz = data[5::6]

        # Compute residual
        err_xx = np.linalg.norm(d_xx - gxxa) / np.linalg.norm(gxxa)
        err_xy = np.linalg.norm(d_xy - gxya) / np.linalg.norm(gxya)
        err_xz = np.linalg.norm(d_xz - gxza) / np.linalg.norm(gxza)
        err_yy = np.linalg.norm(d_yy - gyya) / np.linalg.norm(gyya)
        err_yz = np.linalg.norm(d_yz - gyza) / np.linalg.norm(gyza)
        err_zz = np.linalg.norm(d_zz - gzza) / np.linalg.norm(gzza)
        self.assertLess(err_xx, 0.005)
        self.assertLess(err_xy, 0.005)
        self.assertLess(err_xz, 0.005)
        self.assertLess(err_yy, 0.005)
        self.assertLess(err_yz, 0.005)
        self.assertLess(err_zz, 0.005)


if __name__ == "__main__":
    unittest.main()
