import pytest

import numpy as np
from discretize.utils import mesh_builder_xyz, mkvc

import simpeg
from simpeg.optimization import ProjectedGNCG
from simpeg.potential_fields import gravity, magnetics

COMPONENTS = ["gx", "gy", "gz", "gxx", "gyy", "gzz", "gxy", "gxz", "gyz", "guv"]


def create_grid(x_range, y_range, size):
    """Create a 2D horizontal coordinates grid."""
    x_start, x_end = x_range
    y_start, y_end = y_range
    x, y = np.meshgrid(
        np.linspace(x_start, x_end, size), np.linspace(y_start, y_end, size)
    )
    return x, y


@pytest.fixture()
def mesh_params():
    """Parameters for building the sample meshes."""
    h = [5, 5]
    padding_distances = np.ones((2, 2)) * 50
    return h, padding_distances


@pytest.fixture()
def tensor_mesh(mesh_params, coordinates):
    """Sample 2D tensor mesh to use with equivalent sources."""
    mesh_type = "tensor"
    h, padding_distance = mesh_params
    mesh = mesh_builder_xyz(
        coordinates[:, :2], h, padding_distance=padding_distance, mesh_type=mesh_type
    )
    return mesh


@pytest.fixture()
def tree_mesh(mesh_params, coordinates):
    """Sample 2D tensor mesh to use with equivalent sources."""
    mesh_type = "tree"
    h, padding_distance = mesh_params
    mesh = mesh_builder_xyz(
        coordinates[:, :2], h, padding_distance=padding_distance, mesh_type=mesh_type
    )
    mesh.refine_points(coordinates[:, :2], padding_cells_by_level=[2, 4])
    return mesh


@pytest.fixture(params=["tensor", "tree"])
def mesh(tensor_mesh, tree_mesh, request):
    """Sample 2D mesh to use with equivalent sources."""
    mesh_type = request.param
    if mesh_type == "tree":
        return tree_mesh
    elif mesh_type == "tensor":
        return tensor_mesh
    else:
        raise ValueError(f"Invalid mesh type: '{mesh_type}'")


@pytest.fixture
def mesh_top():
    """Top boundary of the mesh."""
    return -20.0


@pytest.fixture
def mesh_bottom():
    """Bottom boundary of the mesh."""
    return -50.0


@pytest.fixture
def coordinates():
    """Synthetic observation points grid."""
    x, y = create_grid(x_range=(-50, 50), y_range=(-50, 50), size=11)
    z = np.full_like(x, fill_value=5.0)
    return np.c_[mkvc(x), mkvc(y), mkvc(z)]


def get_block_model(mesh, phys_property: float):
    """Build a block model."""
    model = simpeg.utils.model_builder.add_block(
        mesh.cell_centers,
        np.zeros(mesh.n_cells),
        np.r_[-20, -20],
        np.r_[20, 20],
        phys_property,
    )
    return model


def get_mapping(mesh):
    """Get an identity map for the given mesh."""
    return simpeg.maps.IdentityMap(nP=mesh.n_cells)


class TestGravityEquivalentSources:

    @pytest.fixture
    def survey(self, coordinates):
        """
        Sample survey for the gravity equivalent sources.
        """
        return self._build_survey(coordinates, components="gz")

    def _get_mesh_top_bottom(self, mesh, array=False):
        """Build the top and bottom boundaries of the mesh.

        If array is True, the outputs are going to be arrays, otherwise they'll
        be floats.
        """
        top, bottom = -20.0, -50.0
        if array:
            rng = np.random.default_rng(seed=42)
            mesh_top = np.full(mesh.n_cells, fill_value=top) + rng.normal(
                scale=0.5, size=mesh.n_cells
            )
            mesh_bottom = np.full(mesh.n_cells, fill_value=bottom) + rng.normal(
                scale=0.5, size=mesh.n_cells
            )
        else:
            mesh_top, mesh_bottom = top, bottom
        return mesh_top, mesh_bottom

    def _build_survey(self, coordinates, components):
        """
        Build a gravity survey.
        """
        receivers = gravity.Point(coordinates, components=components)
        source_field = gravity.SourceField([receivers])
        survey = gravity.Survey(source_field)
        return survey

    def build_synthetic_data(self, simulation, model):
        data = simulation.make_synthetic_data(
            model,
            relative_error=0.0,
            noise_floor=1e-3,
            add_noise=True,
            random_seed=1,
        )
        return data

    def build_inversion(self, mesh, simulation, synthetic_data):
        """Build inversion problem."""
        # Build data misfit and regularization terms
        data_misfit = simpeg.data_misfit.L2DataMisfit(
            simulation=simulation, data=synthetic_data
        )
        regularization = simpeg.regularization.WeightedLeastSquares(mesh=mesh)
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

    @pytest.mark.parametrize("store_sensitivities", ("ram", "forward_only"))
    def test_forward_geoana_choclo(
        self, mesh, mesh_bottom, mesh_top, survey, store_sensitivities
    ):
        """Test forward using geoana and choclo varying how to store sensitivity."""
        # Build simulations
        mapping = get_mapping(mesh)
        kwargs = dict(
            mesh=mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=survey,
            rhoMap=mapping,
            store_sensitivities=store_sensitivities,
        )
        sim_geoana = gravity.SimulationEquivalentSourceLayer(engine="geoana", **kwargs)
        sim_choclo = gravity.SimulationEquivalentSourceLayer(engine="choclo", **kwargs)
        model = get_block_model(mesh, 2.67)
        np.testing.assert_allclose(sim_geoana.dpred(model), sim_choclo.dpred(model))

    def test_forward_geoana_choclo_on_disk(
        self, mesh, mesh_bottom, mesh_top, survey, tmp_path
    ):
        """Test forward using geoana and choclo varying how to store sensitivity."""
        # Build simulations
        mapping = get_mapping(mesh)
        kwargs = dict(
            mesh=mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=survey,
            rhoMap=mapping,
            store_sensitivities="disk",
        )
        # Create directory for geoana
        sensitivity_dir = tmp_path / "sensitivities_geoana"
        sensitivity_dir.mkdir()
        # Define fname for choclo
        sensitivity_fname = tmp_path / "sensitivities_choclo"
        sim_geoana = gravity.SimulationEquivalentSourceLayer(
            engine="geoana", sensitivity_path=str(sensitivity_dir), **kwargs
        )
        sim_choclo = gravity.SimulationEquivalentSourceLayer(
            engine="choclo", sensitivity_path=str(sensitivity_fname), **kwargs
        )
        model = get_block_model(mesh, 2.67)
        np.testing.assert_allclose(sim_geoana.dpred(model), sim_choclo.dpred(model))

    @pytest.mark.parametrize("components", COMPONENTS + [["gz", "gzz"]])
    def test_forward_geoana_choclo_with_components(
        self, coordinates, mesh, mesh_bottom, mesh_top, components
    ):
        """Test forward using geoana and choclo with different components."""
        # Build survey
        survey = self._build_survey(coordinates, components)
        # Build simulations
        mapping = get_mapping(mesh)
        kwargs = dict(
            mesh=mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=survey,
            rhoMap=mapping,
        )
        sim_geoana = gravity.SimulationEquivalentSourceLayer(engine="geoana", **kwargs)
        sim_choclo = gravity.SimulationEquivalentSourceLayer(engine="choclo", **kwargs)
        model = get_block_model(mesh, 2.67)
        np.testing.assert_allclose(sim_geoana.dpred(model), sim_choclo.dpred(model))

    def test_forward_geoana_choclo_active_cells(
        self, mesh, mesh_bottom, mesh_top, survey
    ):
        """Test forward using geoana and choclo passing active cells."""
        model = get_block_model(mesh, 2.67)
        # Define some inactive cells inside the block
        block_cells_indices = np.indices(model.shape).ravel()[model != 0]
        inactive_indices = block_cells_indices[
            : block_cells_indices.size // 2
        ]  # mark half of the cells in the block as inactive
        active_cells = np.ones_like(model, dtype=bool)
        active_cells[inactive_indices] = False
        assert not np.all(active_cells)  # check we do have inactive cells
        # Build simulations
        mapping = get_mapping(mesh)
        kwargs = dict(
            mesh=mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=survey,
            rhoMap=mapping,
        )
        sim_geoana = gravity.SimulationEquivalentSourceLayer(engine="geoana", **kwargs)
        sim_choclo = gravity.SimulationEquivalentSourceLayer(engine="choclo", **kwargs)
        np.testing.assert_allclose(sim_geoana.dpred(model), sim_choclo.dpred(model))

    def test_forward_choclo_serial_parallel(self, mesh, mesh_bottom, mesh_top, survey):
        """Test forward using choclo in serial and in parallel."""
        # Build simulations
        mapping = get_mapping(mesh)
        kwargs = dict(
            mesh=mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=survey,
            rhoMap=mapping,
            engine="choclo",
        )
        sim_parallel = gravity.SimulationEquivalentSourceLayer(
            numba_parallel=True, **kwargs
        )
        sim_serial = gravity.SimulationEquivalentSourceLayer(
            numba_parallel=False, **kwargs
        )
        model = get_block_model(mesh, 2.67)
        np.testing.assert_allclose(sim_parallel.dpred(model), sim_serial.dpred(model))

    @pytest.mark.parametrize("engine", ("geoana", "choclo"))
    def test_predictions_on_data_points(
        self,
        tree_mesh,
        mesh_top,
        mesh_bottom,
        survey,
        engine,
    ):
        """
        Test eq sources predictions on the same data points.

        The equivalent sources should be able to reproduce the same data with
        which they were trained.
        """
        # Build simulation
        mapping = get_mapping(tree_mesh)
        simulation = gravity.SimulationEquivalentSourceLayer(
            tree_mesh,
            mesh_top,
            mesh_bottom,
            survey=survey,
            rhoMap=mapping,
            engine=engine,
        )
        # Generate synthetic data
        model = get_block_model(tree_mesh, 2.67)
        synthetic_data = self.build_synthetic_data(simulation, model)
        # Build inversion
        inversion = self.build_inversion(tree_mesh, simulation, synthetic_data)
        # Run inversion
        starting_model = np.zeros(tree_mesh.n_cells)
        recovered_model = inversion.run(starting_model)
        # Predict data
        prediction = simulation.dpred(recovered_model)
        # Check if prediction is close to the synthetic data
        atol, rtol = 0.005, 1e-5
        np.testing.assert_allclose(
            prediction, synthetic_data.dobs, atol=atol, rtol=rtol
        )


class TestMagneticEquivalentSources:
    @pytest.fixture
    def survey(self, coordinates):
        """
        Sample survey for the gravity equivalent sources.
        """
        return self._build_survey(coordinates, components="tmi")

    def _build_survey(self, coordinates, components):
        """
        Build a magnetic survey.
        """
        receivers = magnetics.Point(coordinates, components=components)
        source_field = magnetics.UniformBackgroundField(
            [receivers], amplitude=50_000, inclination=35, declination=12
        )
        survey = magnetics.Survey(source_field)
        return survey

    def test_choclo_not_implemented(self, tensor_mesh, mesh_top, mesh_bottom, survey):
        """
        Test if error is raised when passing "choclo" to magnetic eq sources.
        """
        msg = (
            "Magnetic equivalent sources with choclo as engine has not been"
            " implemented yet. Use 'geoana' instead."
        )
        mapping = simpeg.maps.IdentityMap(nP=tensor_mesh.n_cells)
        with pytest.raises(NotImplementedError, match=msg):
            magnetics.SimulationEquivalentSourceLayer(
                tensor_mesh,
                mesh_top,
                mesh_bottom,
                survey=survey,
                rhoMap=mapping,
                engine="choclo",
            )
