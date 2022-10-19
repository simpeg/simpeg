from __future__ import print_function
import unittest
import SimPEG.dask
from SimPEG import (
    directives,
    maps,
    inverse_problem,
    optimization,
    data_misfit,
    inversion,
    utils,
    regularization,
)

from discretize.utils import mesh_utils

import shutil

from SimPEG.potential_fields import magnetics as mag
import numpy as np


class MagInvLinProblemTest(unittest.TestCase):
    def setUp(self):

        np.random.seed(0)

        # First we need to define the direction of the inducing field
        # As a simple case, we pick a vertical inducing field of magnitude
        # 50,000nT.
        # From old convention, field orientation is given as an
        # azimuth from North (positive clockwise)
        # and dip from the horizontal (positive downward).
        H0 = (50000.0, 90.0, 0.0)

        # Create a mesh
        h = [5, 5, 5]
        padDist = np.ones((3, 2)) * 100
        nCpad = [2, 4, 2]

        # Create grid of points for topography
        # Lets create a simple Gaussian topo and set the active cells
        [xx, yy] = np.meshgrid(
            np.linspace(-200.0, 200.0, 50), np.linspace(-200.0, 200.0, 50)
        )

        b = 100
        A = 50
        zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))

        # We would usually load a topofile
        topo = np.c_[utils.mkvc(xx), utils.mkvc(yy), utils.mkvc(zz)]

        # Create and array of observation points
        xr = np.linspace(-100.0, 100.0, 20)
        yr = np.linspace(-100.0, 100.0, 20)
        X, Y = np.meshgrid(xr, yr)
        Z = A * np.exp(-0.5 * ((X / b) ** 2.0 + (Y / b) ** 2.0)) + 5

        # Create a MAGsurvey
        xyzLoc = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]
        rxLoc = mag.Point(xyzLoc)
        srcField = mag.SourceField([rxLoc], parameters=H0)
        survey = mag.Survey(srcField)

        self.mesh = mesh_utils.mesh_builder_xyz(
            xyzLoc,
            h,
            padding_distance=padDist,
            mesh_type="TREE",
        )

        self.mesh = mesh_utils.refine_tree_xyz(
            self.mesh,
            topo,
            method="surface",
            octree_levels=nCpad,
            octree_levels_padding=nCpad,
            finalize=True,
        )

        # Define an active cells from topo
        actv = utils.surface2ind_topo(self.mesh, topo)
        nC = int(actv.sum())

        # We can now create a susceptibility model and generate data
        # Lets start with a simple block in half-space
        self.model = utils.model_builder.addBlock(
            self.mesh.gridCC,
            np.zeros(self.mesh.nC),
            np.r_[-20, -20, -15],
            np.r_[20, 20, 20],
            0.1,
        )[actv]

        # Create active map to go from reduce set to full
        self.actvMap = maps.InjectActiveCells(self.mesh, actv, np.nan)

        # Creat reduced identity map
        idenMap = maps.IdentityMap(nP=nC)

        # Create the forward model operator
        sim = mag.Simulation3DIntegral(
            self.mesh,
            survey=survey,
            chiMap=idenMap,
            actInd=actv,
            store_sensitivities="ram",
        )
        self.sim = sim
        data = sim.make_synthetic_data(
            self.model, relative_error=0.0, noise_floor=1.0, add_noise=True
        )

        # Create a regularization
        reg = regularization.Sparse(
            self.mesh,
            active_cells=actv,
            mapping=idenMap,
            gradient_type="components",
            norms=[0, 0, 0, 0]
        )
        reg.reference_model = np.zeros(nC)

        # Data misfit function
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)

        # Add directives to the inversion
        opt = optimization.ProjectedGNCG(
            maxIter=25,
            lower=0.0,
            upper=10.0,
            maxIterLS=5,
            maxIterCG=10,
            tolCG=1e-4,
            stepOffBoundsFact=1e-4,
        )

        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e6)

        # Here is where the norms are applied
        # Use pick a treshold parameter empirically based on the distribution of
        #  model parameters
        IRLS = directives.Update_IRLS()
        update_Jacobi = directives.UpdatePreconditioner()
        sensitivity_weights = directives.UpdateSensitivityWeights()
        self.inv = inversion.BaseInversion(
            invProb, directiveList=[IRLS, sensitivity_weights, update_Jacobi]
        )

    def test_mag_inverse(self):

        # Run the inversion
        mrec = self.inv.run(self.model * 1e-4)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # ax = plt.subplot(1, 2, 1)
        # self.mesh.plot_slice(self.actvMap*mrec, ax=ax, normal="Y", grid=True)
        # ax = plt.subplot(1, 2, 2)
        # self.mesh.plot_slice(self.actvMap*self.model, ax=ax, normal="Y", grid=True)
        # plt.show()

        residual = np.linalg.norm(mrec - self.model) / np.linalg.norm(self.model)
        self.assertLess(residual, 0.5)

    def tearDown(self):
        # Clean up the working directory
        if self.sim.store_sensitivities == "disk":
            try:
                shutil.rmtree(self.sim.sensitivity_path)
            except:
                pass


if __name__ == "__main__":
    unittest.main()
