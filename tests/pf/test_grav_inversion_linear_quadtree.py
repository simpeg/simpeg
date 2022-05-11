from __future__ import print_function
import unittest
import numpy as np
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
from discretize.utils import mkvc, mesh_builder_xyz, refine_tree_xyz
from SimPEG.potential_fields import gravity

import shutil


class GravInvLinProblemTest(unittest.TestCase):
    def setUp(self):
        def simulate_topo(x, y, amplitude=50, scale_factor=100):
            # Create synthetic Gaussian topography from a function
            return amplitude * np.exp(
                -0.5 * ((x / scale_factor) ** 2.0 + (y / scale_factor) ** 2.0)
            )

        # Create grid of points for topography
        [xx, yy] = np.meshgrid(
            np.linspace(-200.0, 200.0, 50), np.linspace(-200.0, 200.0, 50)
        )
        zz = simulate_topo(xx, yy)

        topo_xyz = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

        # Create and array of observation points
        altitude = 5
        [X, Y] = np.meshgrid(
            np.linspace(-100.0, 100.0, 20), np.linspace(-100.0, 100.0, 20)
        )
        Z = simulate_topo(X, Y) + altitude

        # Create a gravity survey
        xyzLoc = np.c_[mkvc(X.T), mkvc(Y.T), mkvc(Z.T)]
        rxLoc = gravity.Point(xyzLoc)
        srcField = gravity.SourceField([rxLoc])
        survey = gravity.Survey(srcField)

        # Create a quadtree mesh. Only requires x and y coordinates and padding
        h = [5, 5]
        padDist = np.ones((2, 2)) * 100
        nCpad = [2, 4]

        mesh = mesh_builder_xyz(
            topo_xyz[:, :2], h, padding_distance=padDist, mesh_type="TREE",
        )

        mesh = refine_tree_xyz(
            mesh,
            xyzLoc[:, :2],
            method="radial",
            octree_levels=nCpad,
            octree_levels_padding=nCpad,
            finalize=True,
        )

        # elevations are Nx2 array of [bottom-southwest, top-northeast] corners
        # Set tne to topo height at cell centers
        z_tne = simulate_topo(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1])
        # Set bsw to 50 m below the lowest z_tne
        z_bsw = np.full_like(z_tne, fill_value=z_tne.min() - 50.0)
        mesh_elevations = np.c_[z_bsw, z_tne]

        # Create a density model and generate data,
        # with a block in a half space
        self.model = utils.model_builder.addBlock(
            mesh.gridCC, np.zeros(mesh.nC), np.r_[-20, -20], np.r_[20, 20], 0.3,
        )

        # Create reduced identity map. All cells are active in an quadtree
        idenMap = maps.IdentityMap(nP=mesh.nC)

        # Create the forward model operator
        self.sim = gravity.Simulation3DIntegral(
            mesh, survey=survey, rhoMap=idenMap, store_sensitivities="ram",
        )

        # Define the mesh cell heights independent from mesh
        self.sim.Zn = mesh_elevations

        data = self.sim.make_synthetic_data(
            self.model, relative_error=0.0, noise_floor=0.01, add_noise=True
        )

        # Create a regularization
        reg = regularization.Sparse(mesh, mapping=idenMap)
        reg.norms = np.c_[0, 0, 0, 0]
        reg.mref = np.zeros(mesh.nC)

        # Data misfit function
        dmis = data_misfit.L2DataMisfit(simulation=self.sim, data=data)

        # Add directives to the inversion
        opt = optimization.ProjectedGNCG(
            maxIter=15, lower=-1.0, upper=1.0, maxIterLS=5, maxIterCG=5, tolCG=1e-4,
        )

        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e3)

        # Build directives
        IRLS = directives.Update_IRLS(
            f_min_change=1e-3, max_irls_iterations=20, beta_tol=1e-1, beta_search=False
        )
        sensitivity_weights = directives.UpdateSensitivityWeights()
        update_Jacobi = directives.UpdatePreconditioner()
        self.inv = inversion.BaseInversion(
            invProb, directiveList=[IRLS, sensitivity_weights, update_Jacobi]
        )

    def test_grav_inverse(self):

        # Run the inversion
        mrec = self.inv.run(self.model)
        residual = np.linalg.norm(mrec - self.model) / np.linalg.norm(self.model)
        print(residual)

        self.assertLess(residual, 0.7)

    def tearDown(self):
        # Clean up the working directory
        if self.sim.store_sensitivities == "disk":
            shutil.rmtree(self.sim.sensitivity_path)


if __name__ == "__main__":
    unittest.main()
