from __future__ import print_function
import unittest
import numpy as np
import discretize
import dask
import SimPEG.dask
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
from SimPEG.potential_fields import gravity

import shutil

np.random.seed(43)


class GravInvLinProblemTest(unittest.TestCase):
    def setUp(self):

        # Create a self.mesh
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
        zz = -np.exp((xx ** 2 + yy ** 2) / 75 ** 2) + self.mesh.nodes_z[-1]

        # Go from topo to actv cells
        topo = np.c_[utils.mkvc(xx), utils.mkvc(yy), utils.mkvc(zz)]
        actv = utils.surface2ind_topo(self.mesh, topo, "N")

        # Create active map to go from reduce space to full
        self.actvMap = maps.InjectActiveCells(self.mesh, actv, -100)
        nC = int(actv.sum())

        # Create and array of observation points
        xr = np.linspace(-20.0, 20.0, 20)
        yr = np.linspace(-20.0, 20.0, 20)
        X, Y = np.meshgrid(xr, yr)

        # Move the observation points 5m above the topo
        Z = -np.exp((X ** 2 + Y ** 2) / 75 ** 2) + self.mesh.vectorNz[-1] + 5.0

        # Create a MAGsurvey
        locXYZ = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]
        rxLoc = gravity.Point(locXYZ)
        srcField = gravity.SourceField([rxLoc])
        survey = gravity.Survey(srcField)

        # We can now create a density model and generate data
        # Here a simple block in half-space
        model = np.zeros(self.mesh.shape_cells)
        model[(midx - 2) : (midx + 2), (midy - 2) : (midy + 2), -6:-2] = 0.5
        model = utils.mkvc(model)
        self.model = model[actv]

        # Create reduced identity map
        idenMap = maps.IdentityMap(nP=nC)

        # Create the forward model operator
        sim = gravity.Simulation3DIntegral(
            self.mesh,
            survey=survey,
            rhoMap=idenMap,
            actInd=actv,
            store_sensitivities="ram",
        )

        # Compute linear forward operator and compute some data
        # computing sensitivities to ram is best using dask processes
        with dask.config.set(scheduler="processes"):
            data = sim.make_synthetic_data(
                self.model, relative_error=0.0, noise_floor=0.0005, add_noise=True
            )
        # Create a regularization
        reg = regularization.Sparse(self.mesh, active_cells=actv, mapping=idenMap)
        reg.norms = [0, 0, 0, 0]
        reg.gradient_type = "components"

        # Data misfit function
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)

        # Add directives to the inversion
        opt = optimization.ProjectedGNCG(
            maxIter=100, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-4
        )
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e2)

        # Here is where the norms are applied
        IRLS = directives.Update_IRLS(max_irls_iterations=20, chifact_start=2.0)
        update_Jacobi = directives.UpdatePreconditioner()
        sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)
        self.inv = inversion.BaseInversion(
            invProb, directiveList=[IRLS, sensitivity_weights, update_Jacobi]
        )
        self.sim = sim

    def test_grav_inverse(self):
        # Run the inversion
        mrec = self.inv.run(self.model)
        residual = np.linalg.norm(mrec - self.model) / np.linalg.norm(self.model)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # ax = plt.subplot(1, 2, 1)
        # self.mesh.plotSlice(self.actvMap*mrec, ax=ax, clim=(0, 0.5))
        # ax = plt.subplot(1, 2, 2)
        # self.mesh.plotSlice(self.actvMap*self.model, ax=ax, clim=(0, 0.5))
        # plt.show()
        print(f"RESIDUAL !! {residual}")
        self.assertTrue(residual < 0.05)

    def tearDown(self):
        # Clean up the working directory
        if self.sim.store_sensitivities == "disk":
            # Clean up the working directory
            try:
                shutil.rmtree(self.sim.sensitivity_path)
            except:
                pass


if __name__ == "__main__":
    unittest.main()
