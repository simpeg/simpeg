import shutil
import unittest
import numpy as np

from discretize import TensorMesh
from discretize.utils import mkvc, mesh_builder_xyz, refine_tree_xyz
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
from SimPEG.potential_fields import gravity, magnetics

np.random.seed(44)


class QuadTreeLinProblemTest(unittest.TestCase):
    def setUp(self):
        def generate_topo(x, y, amplitude=50, scale_factor=100):
            # Create synthetic Gaussian topography from a function
            return amplitude * np.exp(
                -0.5 * ((x / scale_factor) ** 2.0 + (y / scale_factor) ** 2.0)
            )

        def create_xyz_points_flat(x_range, y_range, spacing, altitude=0.0):
            [xx, yy] = np.meshgrid(
                np.linspace(x_range[0], x_range[1], spacing),
                np.linspace(y_range[0], y_range[1], spacing),
            )
            zz = np.zeros_like(xx) + altitude

            return np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

        def create_xyz_points(x_range, y_range, spacing, altitude=0.0):
            [xx, yy] = np.meshgrid(
                np.linspace(x_range[0], x_range[1], spacing),
                np.linspace(y_range[0], y_range[1], spacing),
            )
            zz = generate_topo(xx, yy) + altitude

            return np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

        def create_mesh(h, padDist, nCpad, topo, data, min_height=1.0):
            # Create a quadtree mesh. Only requires x and y coordinates and padding
            h = [5, 5]
            padDist = np.ones((2, 2)) * 100
            nCpad = [2, 4]

            mesh = mesh_builder_xyz(
                topo[:, :2],
                h,
                padding_distance=padDist,
                mesh_type="TREE",
            )

            self.mesh = refine_tree_xyz(
                mesh,
                data[:, :2],
                method="radial",
                octree_levels=nCpad,
                octree_levels_padding=nCpad,
                finalize=True,
            )

            # elevations are Nx2 array of [bottom-southwest, top-northeast] corners
            # Set tne to topo height at cell centers
            z_tne = generate_topo(
                self.mesh.cell_centers[:, 0], self.mesh.cell_centers[:, 1]
            )
            # Set bsw to 50 m below the lowest z_tne
            z_bsw = np.full_like(z_tne, fill_value=z_tne.min() - min_height)
            self.z_bsw = z_bsw
            self.z_tne = z_tne

        def create_gravity_sim_flat(self, block_value=1.0, noise_floor=0.01):
            # Create a gravity survey
            grav_rxLoc = gravity.Point(data_xyz_flat)
            grav_srcField = gravity.SourceField([grav_rxLoc])
            grav_survey = gravity.Survey(grav_srcField)

            # Create the gravity forward model operator
            self.grav_sim_flat = gravity.SimulationEquivalentSourceLayer(
                self.mesh,
                0.0,
                -5.0,
                survey=grav_survey,
                rhoMap=self.idenMap,
                store_sensitivities="ram",
            )

            self.grav_model = block_value * self.model

            self.grav_data_flat = self.grav_sim_flat.make_synthetic_data(
                self.grav_model,
                relative_error=0.0,
                noise_floor=noise_floor,
                add_noise=True,
            )

        def create_magnetics_sim_flat(self, block_value=1.0, noise_floor=0.01):
            # Create a magnetic survey
            H0 = (50000.0, 90.0, 0.0)
            mag_rxLoc = magnetics.Point(data_xyz_flat)
            mag_srcField = magnetics.SourceField([mag_rxLoc], parameters=H0)
            mag_survey = magnetics.Survey(mag_srcField)

            # Create the magnetics forward model operator
            self.mag_sim_flat = magnetics.SimulationEquivalentSourceLayer(
                self.mesh,
                0.0,
                -5.0,
                survey=mag_survey,
                chiMap=self.idenMap,
                store_sensitivities="ram",
            )

            # Define the mesh cell heights independent from mesh
            self.mag_model = block_value * self.model

            self.mag_data_flat = self.mag_sim_flat.make_synthetic_data(
                self.mag_model,
                relative_error=0.0,
                noise_floor=noise_floor,
                add_noise=True,
            )

        def create_gravity_sim(self, block_value=1.0, noise_floor=0.01):
            # Create a gravity survey
            grav_rxLoc = gravity.Point(data_xyz)
            grav_srcField = gravity.SourceField([grav_rxLoc])
            grav_survey = gravity.Survey(grav_srcField)

            # Create the gravity forward model operator
            self.grav_sim = gravity.SimulationEquivalentSourceLayer(
                self.mesh,
                self.z_tne,
                self.z_bsw,
                survey=grav_survey,
                rhoMap=self.idenMap,
                store_sensitivities="ram",
            )

            # Already defined
            self.grav_model = block_value * self.model

            self.grav_data = self.grav_sim.make_synthetic_data(
                self.grav_model,
                relative_error=0.0,
                noise_floor=noise_floor,
                add_noise=True,
            )

        def create_magnetics_sim(self, block_value=1.0, noise_floor=0.01):
            # Create a magnetic survey
            H0 = (50000.0, 90.0, 0.0)
            mag_rxLoc = magnetics.Point(data_xyz)
            mag_srcField = magnetics.SourceField([mag_rxLoc], parameters=H0)
            mag_survey = magnetics.Survey(mag_srcField)

            # Create the magnetics forward model operator
            self.mag_sim = magnetics.SimulationEquivalentSourceLayer(
                self.mesh,
                self.z_tne,
                self.z_bsw,
                survey=mag_survey,
                chiMap=self.idenMap,
                store_sensitivities="ram",
            )

            # Already defined
            self.mag_model = block_value * self.model

            self.mag_data = self.mag_sim.make_synthetic_data(
                self.mag_model,
                relative_error=0.0,
                noise_floor=noise_floor,
                add_noise=True,
            )

        def create_inversion(self, sim, data, beta=1e3):

            # Create a regularization
            reg = regularization.Sparse(self.mesh, mapping=self.idenMap)
            reg.norms = [0, 0, 0]
            reg.gradient_type = "components"
            reg.reference_model = np.zeros(self.mesh.nC)

            # Data misfit function
            dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)

            # Add directives to the inversion
            opt = optimization.ProjectedGNCG(
                maxIter=40,
                lower=-1.0,
                upper=1.0,
                maxIterLS=5,
                maxIterCG=10,
                tolCG=1e-4,
            )

            invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=beta)

            # Build directives
            IRLS = directives.Update_IRLS(
                f_min_change=1e-3,
                max_irls_iterations=30,
                beta_tol=1e-1,
                beta_search=False,
            )
            sensitivity_weights = directives.UpdateSensitivityWeights()
            update_Jacobi = directives.UpdatePreconditioner()
            inv = inversion.BaseInversion(
                invProb, directiveList=[IRLS, sensitivity_weights, update_Jacobi]
            )

            return inv

        # Create grid of points for topography
        topo_xyz = create_xyz_points(
            x_range=(-200, 200), y_range=(-200, 200), spacing=50, altitude=0
        )

        # Create and array of observation points
        data_xyz = create_xyz_points(
            x_range=(-100, 100), y_range=(-100, 100), spacing=20, altitude=5
        )

        data_xyz_flat = create_xyz_points_flat(
            x_range=(-100, 100), y_range=(-100, 100), spacing=20, altitude=5
        )

        # Create the mesh
        create_mesh(
            h=[5, 5],
            padDist=np.ones((2, 2)) * 100,
            nCpad=[2, 4],
            topo=topo_xyz,
            data=data_xyz,
            min_height=50.0,
        )

        # Create a density model and generate data,
        # with a block in a half space
        self.model = utils.model_builder.addBlock(
            self.mesh.cell_centers,
            np.zeros(self.mesh.nC),
            np.r_[-20, -20],
            np.r_[20, 20],
            1.0,
        )

        # Create reduced identity map. All cells are active in an quadtree
        self.idenMap = maps.IdentityMap(nP=self.mesh.nC)

        # create_gravity_sim_flat(self, block_value=0.3, noise_floor=0.01)
        # create_magnetics_sim_flat(self, block_value=0.3, noise_floor=0.01)

        create_gravity_sim(self, block_value=0.3, noise_floor=0.001)
        self.grav_inv = create_inversion(self, self.grav_sim, self.grav_data, beta=1e3)

        create_magnetics_sim(self, block_value=0.03, noise_floor=3.0)
        self.mag_inv = create_inversion(self, self.mag_sim, self.mag_data, beta=1e3)

    def test_instantiation_failures(self):

        # Ensure simulation can't be instantiated with 3D mesh.
        dh = 5.0
        hx = [(dh, 5, -1.3), (dh, 10), (dh, 5, 1.3)]
        hy = [(dh, 5, -1.3), (dh, 10), (dh, 5, 1.3)]
        hz = [(dh, 1)]
        mesh3D = TensorMesh([hx, hy, hz], "CCN")

        def create_xyz_points_flat(x_range, y_range, spacing, altitude=0.0):
            [xx, yy] = np.meshgrid(
                np.linspace(x_range[0], x_range[1], spacing),
                np.linspace(y_range[0], y_range[1], spacing),
            )
            zz = np.zeros_like(xx) + altitude

            return np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

        data_xyz_flat = create_xyz_points_flat(
            x_range=(-100, 100), y_range=(-100, 100), spacing=20, altitude=5
        )

        grav_rxLoc = gravity.Point(data_xyz_flat)
        grav_srcField = gravity.SourceField([grav_rxLoc])
        grav_survey = gravity.Survey(grav_srcField)

        self.assertRaises(
            AttributeError,
            gravity.SimulationEquivalentSourceLayer,
            mesh3D,
            0.0,
            -5.0,
            survey=grav_survey,
            rhoMap=self.idenMap,
        )

        print("3D MESH ERROR TEST PASSED.")

        self.assertRaises(
            AttributeError,
            gravity.SimulationEquivalentSourceLayer,
            self.mesh,
            np.zeros(5),
            -5.0 * np.ones(5),
            survey=grav_survey,
            rhoMap=self.idenMap,
        )

        print("Z_TOP OR Z_BOTTOM LENGTH MATCHING NCELLS ERROR TEST PASSED.")

    def test_quadtree_grav_inverse(self):

        # Run the inversion from a zero starting model
        mrec = self.grav_inv.run(np.zeros(self.mesh.nC))

        # Compute predicted data
        dpred = self.grav_sim.dpred(self.grav_model)

        # Check models match well enough (allowing for random noise)
        model_residual = np.linalg.norm(mrec - self.grav_model) / np.linalg.norm(
            self.grav_model
        )
        self.assertAlmostEqual(model_residual, 0.2, delta=0.1)

        # Check data converged to less than 10% of target misfit
        data_misfit = 2.0 * self.grav_inv.invProb.dmisfit(self.grav_model)
        self.assertLess(data_misfit, dpred.shape[0] * 1.1)

    def test_quadtree_mag_inverse(self):

        # Run the inversion from a zero starting model
        mrec = self.mag_inv.run(np.zeros(self.mesh.nC))

        # Compute predicted data
        dpred = self.mag_sim.dpred(self.mag_model)

        # Check models match well enough (allowing for random noise)
        model_residual = np.linalg.norm(mrec - self.mag_model) / np.linalg.norm(
            self.mag_model
        )
        self.assertAlmostEqual(model_residual, 0.03, delta=0.05)

        # Check data converged to less than 10% of target misfit
        data_misfit = 2.0 * self.mag_inv.invProb.dmisfit(self.mag_model)
        self.assertLess(data_misfit, dpred.shape[0] * 1.1)

    def tearDown(self):
        # Clean up the working directory
        if self.grav_sim.store_sensitivities == "disk":
            shutil.rmtree(self.grav_sim.sensitivity_path)

        if self.mag_sim.store_sensitivities == "disk":
            shutil.rmtree(self.mag_sim.sensitivity_path)


if __name__ == "__main__":
    unittest.main()
