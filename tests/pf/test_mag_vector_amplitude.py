import unittest
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


from discretize.utils import mesh_builder_xyz, refine_tree_xyz, active_from_xyz
import numpy as np
from SimPEG.potential_fields import magnetics as mag
import shutil


class MVIProblemTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        H0 = (50000.0, 90.0, 0.0)

        # The magnetization is set along a different
        # direction (induced + remanence)
        M = np.array([45.0, 90.0])

        # Create grid of points for topography
        # Lets create a simple Gaussian topo
        # and set the active cells
        [xx, yy] = np.meshgrid(np.linspace(-200, 200, 50), np.linspace(-200, 200, 50))
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

        # Create a mesh
        h = [5, 5, 5]
        padDist = np.ones((3, 2)) * 100

        mesh = mesh_builder_xyz(
            xyzLoc, h, padding_distance=padDist, depth_core=100, mesh_type="tree"
        )
        mesh = refine_tree_xyz(
            mesh, topo, method="surface", octree_levels=[4, 4], finalize=True
        )
        self.mesh = mesh
        # Define an active cells from topo
        actv = active_from_xyz(mesh, topo)
        nC = int(actv.sum())

        model = np.zeros((mesh.nC, 3))

        # Convert the inclination declination to vector in Cartesian
        M_xyz = utils.mat_utils.dip_azimuth2cartesian(M[0], M[1])

        # Get the indicies of the magnetized block
        ind = utils.model_builder.getIndicesBlock(
            np.r_[-20, -20, -10],
            np.r_[20, 20, 25],
            mesh.gridCC,
        )[0]

        # Assign magnetization values
        model[ind, :] = np.kron(np.ones((ind.shape[0], 1)), M_xyz * 0.05)

        # Remove air cells
        self.model = model[actv, :]

        # Create active map to go from reduce set to full
        self.actvMap = maps.InjectActiveCells(mesh, actv, np.nan)

        # Creat reduced identity map
        idenMap = maps.IdentityMap(nP=nC * 3)

        # Create the forward model operator
        sim = mag.Simulation3DIntegral(
            self.mesh,
            survey=survey,
            model_type="vector",
            chiMap=idenMap,
            ind_active=actv,
            store_sensitivities="disk",
        )
        self.sim = sim

        # Compute some data and add some random noise
        data = sim.make_synthetic_data(
            utils.mkvc(self.model), relative_error=0.0, noise_floor=5.0, add_noise=True
        )

        wires = maps.Wires(("p", nC), ("s", nC), ("t", nC))
        reg = regularization.VectorAmplitude(
            mesh,
            wires,
            active_cells=actv,
            reference_model_in_smooth=True,
            norms=[0.0, 0.0, 0.0, 0.0],
            gradient_type="components",
        )

        # Data misfit function
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)
        # dmis.W = 1./survey.std

        # Add directives to the inversion
        opt = optimization.ProjectedGNCG(
            maxIter=10, lower=-10, upper=10.0, maxIterLS=5, maxIterCG=5, tolCG=1e-4
        )

        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)

        # A list of directive to control the inverson
        betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e1)

        # Here is where the norms are applied
        # Use pick a treshold parameter empirically based on the distribution of
        #  model parameters
        IRLS = directives.Update_IRLS(
            f_min_change=1e-3, max_irls_iterations=10, beta_tol=5e-1
        )

        # Pre-conditioner
        update_Jacobi = directives.UpdatePreconditioner()
        sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)
        self.inv = inversion.BaseInversion(
            invProb, directiveList=[sensitivity_weights, IRLS, update_Jacobi, betaest]
        )

        # Run the inversion
        self.mstart = np.ones(3 * nC) * 1e-4  # Starting model

    def test_vector_amplitude_inverse(self):
        # Run the inversion
        mrec = self.inv.run(self.mstart)

        nC = int(mrec.shape[0] / 3)
        vec_xyz = mrec.reshape((nC, 3), order="F")
        residual = np.linalg.norm(vec_xyz - self.model) / np.linalg.norm(self.model)

        # import matplotlib.pyplot as plt
        # ax = plt.subplot()
        # self.mesh.plot_slice(
        #     self.actvMap * mrec.reshape((-1, 3), order="F"),
        #     v_type="CCv",
        #     view="vec",
        #     ax=ax,
        #     normal="Y",
        #     grid=True,
        #     quiver_opts={
        #         "pivot": "mid",
        #         "scale": 8 * np.abs(mrec).max(),
        #         "scale_units": "inches",
        #     },
        # )
        # plt.gca().set_aspect("equal", adjustable="box")
        #
        # plt.show()

        self.assertLess(residual, 1)
        # self.assertTrue(residual < 0.05)

    def tearDown(self):
        # Clean up the working directory
        if self.sim.store_sensitivities == "disk":
            shutil.rmtree(self.sim.sensitivity_path)


if __name__ == "__main__":
    unittest.main()
