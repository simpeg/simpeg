import unittest
import discretize
from discretize.utils import active_from_xyz
from SimPEG import (
    utils,
    maps,
    regularization,
    data_misfit,
    optimization,
    inverse_problem,
    directives,
    inversion,
)
import numpy as np

# import SimPEG.PF as PF
from SimPEG.potential_fields import magnetics as mag
import shutil


class MagInvLinProblemTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

        # Define the inducing field parameter
        h0_amplitude, h0_inclination, h0_declination = (50000, 90, 0)

        # Create a mesh
        dx = 5.0
        hxind = [(dx, 5, -1.3), (dx, 5), (dx, 5, 1.3)]
        hyind = [(dx, 5, -1.3), (dx, 5), (dx, 5, 1.3)]
        hzind = [(dx, 5, -1.3), (dx, 6)]
        self.mesh = discretize.TensorMesh([hxind, hyind, hzind], "CCC")

        # Get index of the center
        midx = int(self.mesh.shape_cells[0] / 2)
        midy = int(self.mesh.shape_cells[1] / 2)

        # Lets create a simple Gaussian topo and set the active cells
        [xx, yy] = np.meshgrid(self.mesh.nodes_x, self.mesh.nodes_y)
        zz = -np.exp((xx**2 + yy**2) / 75**2) + self.mesh.nodes_z[-1]

        # Go from topo to actv cells
        topo = np.c_[utils.mkvc(xx), utils.mkvc(yy), utils.mkvc(zz)]
        actv = active_from_xyz(self.mesh, topo, "N")

        # Create active map to go from reduce space to full
        self.actvMap = maps.InjectActiveCells(self.mesh, actv, -100)
        nC = int(actv.sum())

        # Create and array of observation points
        xr = np.linspace(-20.0, 20.0, 20)
        yr = np.linspace(-20.0, 20.0, 20)
        X, Y = np.meshgrid(xr, yr)

        # Move the observation points 5m above the topo
        Z = -np.exp((X**2 + Y**2) / 75**2) + self.mesh.nodes_z[-1] + 5.0

        # Create a MAGsurvey
        rxLoc = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]
        rxLoc = mag.Point(rxLoc)
        srcField = mag.UniformBackgroundField(
            receiver_list=[rxLoc],
            amplitude=h0_amplitude,
            inclination=h0_inclination,
            declination=h0_declination,
        )
        survey = mag.Survey(srcField)

        # We can now create a susceptibility model and generate data
        # Here a simple block in half-space
        model = np.zeros(self.mesh.shape_cells)
        model[(midx - 2) : (midx + 2), (midy - 2) : (midy + 2), -6:-2] = 0.02
        model = utils.mkvc(model)
        self.model = model[actv]

        # Create active map to go from reduce set to full
        self.actvMap = maps.InjectActiveCells(self.mesh, actv, -100)

        # Creat reduced identity map
        idenMap = maps.IdentityMap(nP=nC)

        # Create the forward model operator
        sim = mag.Simulation3DIntegral(
            self.mesh,
            survey=survey,
            chiMap=idenMap,
            ind_active=actv,
            store_sensitivities="disk",
            n_processes=None,
        )
        self.sim = sim

        # Compute linear forward operator and compute some data
        data = sim.make_synthetic_data(
            self.model,
            relative_error=0.0,
            noise_floor=1.0,
            add_noise=True,
            random_seed=2,
        )

        # Create a regularization
        reg = regularization.Sparse(self.mesh, active_cells=actv, mapping=idenMap)
        reg.norms = [0, 0, 0, 0]
        reg.gradient_type = "components"

        # Data misfit function
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)

        # Add directives to the inversion
        opt = optimization.ProjectedGNCG(
            maxIter=100, lower=0.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
        )

        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
        betaest = directives.BetaEstimate_ByEig()

        # Here is where the norms are applied
        IRLS = directives.Update_IRLS(f_min_change=1e-4, minGNiter=1)
        update_Jacobi = directives.UpdatePreconditioner()
        sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)
        self.inv = inversion.BaseInversion(
            invProb, directiveList=[IRLS, sensitivity_weights, betaest, update_Jacobi]
        )

    def test_mag_inverse(self):
        # Run the inversion
        mrec = self.inv.run(self.model)
        residual = np.linalg.norm(mrec - self.model) / np.linalg.norm(self.model)

        # plt.figure()
        # ax = plt.subplot(1, 2, 1)
        # midx = int(self.mesh.shape_cells[0]/2)
        # self.mesh.plot_slice(self.actvMap*mrec, ax=ax, normal='Y', ind=midx,
        #                grid=True, clim=(0, 0.02))

        # ax = plt.subplot(1, 2, 2)
        # midx = int(self.mesh.shape_cells[0]/2)
        # self.mesh.plot_slice(self.actvMap*self.model, ax=ax, normal='Y', ind=midx,
        #                grid=True, clim=(0, 0.02))
        # plt.show()

        self.assertTrue(residual < 0.05)

    def tearDown(self):
        # Clean up the working directory
        if self.sim.store_sensitivities == "disk":
            shutil.rmtree(self.sim.sensitivity_path)


if __name__ == "__main__":
    unittest.main()
