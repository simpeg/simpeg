import pytest

import numpy as np
from discretize.utils import mesh_builder_xyz, mkvc, refine_tree_xyz

import simpeg
from simpeg.optimization import ProjectedGNCG
from simpeg.potential_fields import gravity


def create_grid(x_range, y_range, spacing):
    """Create a 2D horizontal coordinates grid."""
    x_start, x_end = x_range
    y_start, y_end = y_range
    x, y = np.meshgrid(
        np.linspace(x_start, x_end, spacing), np.linspace(y_start, y_end, spacing)
    )
    return x, y


@pytest.fixture
def tree_mesh(coordinates):
    """Sample 2D mesh to use with equivalent sources."""
    # Define parameters for building the mesh
    h = [5, 5]
    padDist = np.ones((2, 2)) * 100
    nCpad = [2, 4]
    # Build mesh
    mesh = mesh_builder_xyz(
        coordinates[:, :2], h, padding_distance=padDist, mesh_type="TREE"
    )
    # Refine mesh
    mesh = refine_tree_xyz(
        mesh,
        coordinates[:, :2],
        method="radial",
        octree_levels=nCpad,
        octree_levels_padding=nCpad,
        finalize=True,
    )
    return mesh


@pytest.fixture
def mesh_top():
    return -20.0


@pytest.fixture
def mesh_bottom():
    return -50.0


@pytest.fixture
def coordinates():
    """Synthetic observation points grid."""
    x, y = create_grid(x_range=(-100, 100), y_range=(-100, 100), spacing=50)
    z = np.full_like(x, fill_value=5.0)
    return np.c_[mkvc(x), mkvc(y), mkvc(z)]


@pytest.fixture
def block_model(tree_mesh):
    """Build a block model."""
    model = simpeg.utils.model_builder.add_block(
        tree_mesh.cell_centers,
        np.zeros(tree_mesh.n_cells),
        np.r_[-20, -20],
        np.r_[20, 20],
        1.0,
    )
    return model


class TestGravityEquivalentSources:

    @pytest.fixture
    def survey(self, coordinates):
        """
        Sample survey for the gravity equivalent sources.
        """
        receivers = gravity.Point(coordinates)
        source_field = gravity.SourceField([receivers])
        survey = gravity.Survey(source_field)
        return survey

    @pytest.fixture
    def model(self, block_model):
        """
        Sample model the gravity equivalent sources.
        """
        density = 2.67  # in g/cc
        return density * block_model

    @pytest.fixture
    def mapping(self, tree_mesh):
        return simpeg.maps.IdentityMap(nP=tree_mesh.n_cells)

    @pytest.fixture(params=("geoana", "choclo"))
    def simulation(self, tree_mesh, mesh_bottom, mesh_top, survey, mapping, request):
        engine = request.param
        simulation = gravity.SimulationEquivalentSourceLayer(
            tree_mesh,
            mesh_top,
            mesh_bottom,
            survey=survey,
            rhoMap=mapping,
            engine=engine,
        )
        return simulation

    @pytest.fixture
    def synthetic_data(self, simulation, model):
        data = simulation.make_synthetic_data(
            model,
            relative_error=0.0,
            noise_floor=1e-3,
            add_noise=True,
            random_seed=1,
        )
        return data

    def build_inversion(self, tree_mesh, simulation, synthetic_data):
        """Build inversion problem."""
        # Build data misfit and regularization terms
        data_misfit = simpeg.data_misfit.L2DataMisfit(
            simulation=simulation, data=synthetic_data
        )
        regularization = simpeg.regularization.WeightedLeastSquares(mesh=tree_mesh)
        # Choose optimization
        optimization = ProjectedGNCG(
            maxIterLS=5,
            maxIterCG=20,
            tolCG=1e-4,
        )
        # Build inverse problem
        inverse_problem = simpeg.inverse_problem.BaseInvProblem(
            data_misfit, regularization, optimization
        )
        # Define directives
        starting_beta = simpeg.directives.BetaEstimate_ByEig(beta0_ratio=1e-1, seed=42)
        beta_schedule = simpeg.directives.BetaSchedule(coolingFactor=3, coolingRate=1)
        update_jacobi = simpeg.directives.UpdatePreconditioner()
        target_misfit = simpeg.directives.TargetMisfit(chifact=1)
        sensitivity_weights = simpeg.directives.UpdateSensitivityWeights(
            every_iteration=False
        )
        directives = [
            sensitivity_weights,
            starting_beta,
            beta_schedule,
            update_jacobi,
            target_misfit,
        ]
        # Define inversion
        inversion = simpeg.inversion.BaseInversion(inverse_problem, directives)
        return inversion

    def test_gravity_equivalent_sources(self, tree_mesh, simulation, synthetic_data):
        """Test gravity equivalent sources against block model."""
        # Build inversion
        inversion = self.build_inversion(tree_mesh, simulation, synthetic_data)
        # Run inversion
        starting_model = np.zeros(tree_mesh.n_cells)
        recovered_model = inversion.run(starting_model)
        # Predict data
        prediction = simulation.dpred(recovered_model)
        # Check if prediction is close to the synthetic data
        atol, rtol = 0.004, 1e-5
        np.testing.assert_allclose(
            prediction, synthetic_data.dobs, atol=atol, rtol=rtol
        )
