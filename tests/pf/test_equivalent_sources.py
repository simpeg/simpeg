import pytest

from collections.abc import Iterable
import numpy as np
from discretize import TensorMesh
from discretize.utils import mesh_builder_xyz, mkvc
import simpeg
from simpeg.optimization import ProjectedGNCG
from simpeg.potential_fields import gravity, magnetics, base

GRAVITY_COMPONENTS = ["gx", "gy", "gz", "gxx", "gyy", "gzz", "gxy", "gxz", "gyz", "guv"]
MAGNETIC_COMPONENTS = [
    "tmi",
    "bx",
    "by",
    "bz",
    "bxx",
    "byy",
    "bzz",
    "bxy",
    "bxz",
    "byz",
    "tmi_x",
    "tmi_y",
    "tmi_z",
]


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
    """Sample 2D tree mesh to use with equivalent sources."""
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


def get_block_model(mesh, phys_property: float | tuple):
    """
    Build a block model.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        Mesh.
    phys_property : float or tuple of floats
        Pass a tuple of floats if you want to generate a vector model.

    Returns
    -------
    model : np.ndarray
    """
    if not isinstance(phys_property, Iterable):
        model = simpeg.utils.model_builder.add_block(
            mesh.cell_centers,
            np.zeros(mesh.n_cells),
            np.r_[-20, -20],
            np.r_[20, 20],
            phys_property,
        )
    else:
        models = tuple(
            simpeg.utils.model_builder.add_block(
                mesh.cell_centers,
                np.zeros(mesh.n_cells),
                np.r_[-20, -20],
                np.r_[20, 20],
                p,
            )
            for p in phys_property
        )
        model = np.hstack(models)
    return model


def get_mapping(mesh):
    """Get an identity map for the given mesh."""
    return simpeg.maps.IdentityMap(nP=mesh.n_cells)


def get_mesh_3d(mesh, top: float, bottom: float):
    """
    Build a 3D mesh analogous to the 2D mesh + the top and bottom bounds.
    """
    origin = (*mesh.origin, bottom)
    h = (*mesh.h, np.array([top - bottom], dtype=np.float64))
    mesh_3d = TensorMesh(h=h, origin=origin)
    return mesh_3d


@pytest.fixture
def gravity_survey(coordinates):
    """
    Sample survey for the gravity equivalent sources.
    """
    return build_gravity_survey(coordinates, components="gz")


@pytest.fixture
def magnetic_survey(coordinates):
    """
    Sample survey for the magnetic equivalent sources.
    """
    return build_magnetic_survey(coordinates, components="tmi")


def build_gravity_survey(coordinates, components):
    """
    Build a gravity survey.
    """
    receivers = gravity.Point(coordinates, components=components)
    source_field = gravity.SourceField([receivers])
    survey = gravity.Survey(source_field)
    return survey


def build_magnetic_survey(
    coordinates, components, amplitude=51_000.0, inclination=71.0, declination=12.0
):
    """
    Build a magnetic survey.
    """
    receivers = magnetics.Point(coordinates, components=components)
    source_field = magnetics.UniformBackgroundField(
        [receivers],
        amplitude=amplitude,
        inclination=inclination,
        declination=declination,
    )
    survey = magnetics.Survey(source_field)
    return survey


class Test3DMeshError:
    """
    Test if error is raised after passing a 3D mesh to equivalent sources.
    """

    @pytest.fixture
    def mesh_3d(self):
        mesh = TensorMesh([2, 3, 4])
        return mesh

    @pytest.mark.parametrize("engine", ("geoana", "choclo"))
    def test_error_on_gravity(self, mesh_3d, engine):
        """
        Test error is raised after passing a 3D mesh to gravity eq source class.
        """
        msg = "SimulationEquivalentSourceLayer mesh must be 2D, received a 3D mesh."
        with pytest.raises(ValueError, match=msg):
            gravity.SimulationEquivalentSourceLayer(
                mesh=mesh_3d, cell_z_top=0.0, cell_z_bottom=-2.0, engine=engine
            )

    @pytest.mark.parametrize("engine", ("geoana", "choclo"))
    def test_error_on_mag(self, mesh_3d, engine):
        """
        Test error is raised after passing a 3D mesh to magnetic eq source class.
        """
        msg = "SimulationEquivalentSourceLayer mesh must be 2D, received a 3D mesh."
        with pytest.raises(ValueError, match=msg):
            magnetics.SimulationEquivalentSourceLayer(
                mesh=mesh_3d, cell_z_top=0.0, cell_z_bottom=-2.0, engine=engine
            )

    def test_error_on_base_class(self, mesh_3d):
        """
        Test error is raised after passing a 3D mesh to the eq source base class.
        """
        msg = "BaseEquivalentSourceLayerSimulation mesh must be 2D, received a 3D mesh."
        with pytest.raises(ValueError, match=msg):
            base.BaseEquivalentSourceLayerSimulation(
                mesh=mesh_3d, cell_z_top=0.0, cell_z_bottom=-2.0
            )


class TestGravityEquivalentSourcesForward:
    """
    Test the forward capabilities of the gravity equivalent sources.
    """

    @pytest.mark.parametrize("engine", ("geoana", "choclo"))
    @pytest.mark.parametrize("store_sensitivities", ("ram", "forward_only"))
    def test_forward_vs_simulation(
        self,
        tensor_mesh,
        mesh_bottom,
        mesh_top,
        gravity_survey,
        engine,
        store_sensitivities,
    ):
        """
        Test forward of the eq sources vs. using the integral 3d simulation.
        """
        # Build 3D mesh that is analogous to the 2D mesh with bottom and top
        mesh_3d = get_mesh_3d(tensor_mesh, top=mesh_top, bottom=mesh_bottom)
        # Build simulations
        mapping = get_mapping(tensor_mesh)
        sim_3d = gravity.Simulation3DIntegral(
            survey=gravity_survey, mesh=mesh_3d, rhoMap=mapping
        )
        eq_sources = gravity.SimulationEquivalentSourceLayer(
            mesh=tensor_mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=gravity_survey,
            rhoMap=mapping,
            engine=engine,
            store_sensitivities=store_sensitivities,
        )
        # Compare predictions of both simulations
        model = get_block_model(tensor_mesh, 2.67)
        np.testing.assert_allclose(
            sim_3d.dpred(model), eq_sources.dpred(model), atol=1e-7
        )

    @pytest.mark.parametrize("engine", ("geoana", "choclo"))
    @pytest.mark.parametrize("store_sensitivities", ("ram", "forward_only"))
    @pytest.mark.parametrize("components", GRAVITY_COMPONENTS + [["gz", "gzz"]])
    def test_forward_vs_simulation_with_components(
        self,
        coordinates,
        tensor_mesh,
        mesh_bottom,
        mesh_top,
        engine,
        store_sensitivities,
        components,
    ):
        """
        Test forward vs simulation using different gravity components.
        """
        # Build survey
        survey = build_gravity_survey(coordinates, components)
        # Build 3D mesh that is analogous to the 2D mesh with bottom and top
        mesh_3d = get_mesh_3d(tensor_mesh, top=mesh_top, bottom=mesh_bottom)
        # Build simulations
        mapping = get_mapping(tensor_mesh)
        sim_3d = gravity.Simulation3DIntegral(
            survey=survey, mesh=mesh_3d, rhoMap=mapping
        )
        eq_sources = gravity.SimulationEquivalentSourceLayer(
            mesh=tensor_mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=survey,
            rhoMap=mapping,
            engine=engine,
            store_sensitivities=store_sensitivities,
        )
        # Compare predictions of both simulations
        model = get_block_model(tensor_mesh, 2.67)
        np.testing.assert_allclose(
            sim_3d.dpred(model), eq_sources.dpred(model), atol=5e-6
        )

    @pytest.mark.parametrize("engine", ("geoana", "choclo"))
    def test_forward_vs_simulation_on_disk(
        self,
        tensor_mesh,
        mesh_bottom,
        mesh_top,
        gravity_survey,
        engine,
        tmp_path,
    ):
        """
        Test forward vs simulation storing sensitivities on disk.
        """
        # Build 3D mesh that is analogous to the 2D mesh with bottom and top
        mesh_3d = get_mesh_3d(tensor_mesh, top=mesh_top, bottom=mesh_bottom)
        # Define sensitivity_dir
        if engine == "geoana":
            sensitivity_path = tmp_path / "sensitivities_geoana"
            sensitivity_path.mkdir()
        elif engine == "choclo":
            sensitivity_path = tmp_path / "sensitivities_choclo"
        # Build simulations
        mapping = get_mapping(tensor_mesh)
        sim_3d = gravity.Simulation3DIntegral(
            survey=gravity_survey, mesh=mesh_3d, rhoMap=mapping
        )
        eq_sources = gravity.SimulationEquivalentSourceLayer(
            mesh=tensor_mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=gravity_survey,
            rhoMap=mapping,
            engine=engine,
            store_sensitivities="disk",
            sensitivity_path=str(sensitivity_path),
        )
        # Compare predictions of both simulations
        model = get_block_model(tensor_mesh, 2.67)
        np.testing.assert_allclose(sim_3d.dpred(model), eq_sources.dpred(model))

    @pytest.mark.parametrize("engine", ("geoana", "choclo"))
    @pytest.mark.parametrize("store_sensitivities", ("ram", "forward_only"))
    def test_forward_vs_simulation_with_active_cells(
        self,
        tensor_mesh,
        mesh_bottom,
        mesh_top,
        gravity_survey,
        engine,
        store_sensitivities,
    ):
        """
        Test forward vs simulation using active cells.
        """
        model = get_block_model(tensor_mesh, 2.67)

        # Define some inactive cells inside the block
        block_cells_indices = np.indices(model.shape).ravel()[model != 0]
        inactive_indices = block_cells_indices[
            : block_cells_indices.size // 2
        ]  # mark half of the cells in the block as inactive
        active_cells = np.ones_like(model, dtype=bool)
        active_cells[inactive_indices] = False
        assert not np.all(active_cells)  # check we do have inactive cells

        # Keep only values of the model in the active cells
        model = model[active_cells]

        # Build 3D mesh that is analogous to the 2D mesh with bottom and top
        mesh_3d = get_mesh_3d(tensor_mesh, top=mesh_top, bottom=mesh_bottom)

        # Build simulations
        mapping = simpeg.maps.IdentityMap(nP=model.size)
        sim_3d = gravity.Simulation3DIntegral(
            survey=gravity_survey,
            mesh=mesh_3d,
            rhoMap=mapping,
            active_cells=active_cells,
        )
        eq_sources = gravity.SimulationEquivalentSourceLayer(
            mesh=tensor_mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=gravity_survey,
            rhoMap=mapping,
            engine=engine,
            store_sensitivities=store_sensitivities,
            active_cells=active_cells,
        )
        # Compare predictions of both simulations
        np.testing.assert_allclose(
            sim_3d.dpred(model), eq_sources.dpred(model), atol=1e-7
        )

    @pytest.mark.parametrize("store_sensitivities", ("ram", "forward_only"))
    def test_forward_geoana_choclo(
        self, mesh, mesh_bottom, mesh_top, gravity_survey, store_sensitivities
    ):
        """Compare forwards using geoana and choclo."""
        # Build simulations
        mapping = get_mapping(mesh)
        kwargs = dict(
            mesh=mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=gravity_survey,
            rhoMap=mapping,
            store_sensitivities=store_sensitivities,
        )
        sim_geoana = gravity.SimulationEquivalentSourceLayer(engine="geoana", **kwargs)
        sim_choclo = gravity.SimulationEquivalentSourceLayer(engine="choclo", **kwargs)
        model = get_block_model(mesh, 2.67)
        np.testing.assert_allclose(sim_geoana.dpred(model), sim_choclo.dpred(model))

    def test_forward_choclo_serial_parallel(
        self, mesh, mesh_bottom, mesh_top, gravity_survey
    ):
        """Test forward using choclo in serial and in parallel."""
        # Build simulations
        mapping = get_mapping(mesh)
        kwargs = dict(
            mesh=mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=gravity_survey,
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


class TestMagneticEquivalentSourcesForward:
    """
    Test the forward capabilities of the magnetic equivalent sources.
    """

    @pytest.mark.parametrize("engine", ("geoana", "choclo"))
    @pytest.mark.parametrize("store_sensitivities", ("ram", "forward_only"))
    @pytest.mark.parametrize("model_type", ("scalar", "vector"))
    @pytest.mark.parametrize("components", MAGNETIC_COMPONENTS + [["tmi", "bx"]])
    def test_forward_vs_simulation(
        self,
        coordinates,
        tensor_mesh,
        mesh_bottom,
        mesh_top,
        engine,
        store_sensitivities,
        model_type,
        components,
    ):
        """
        Test forward of the eq sources vs. using the integral 3d simulation.
        """
        # Build survey
        survey = build_magnetic_survey(coordinates, components)
        # Build 3D mesh that is analogous to the 2D mesh with bottom and top
        mesh_3d = get_mesh_3d(tensor_mesh, top=mesh_top, bottom=mesh_bottom)
        # Build model and mapping
        if model_type == "scalar":
            model = get_block_model(tensor_mesh, 0.2e-3)
        else:
            model = get_block_model(tensor_mesh, (0.2e-3, -0.1e-3, 0.5e-3))
        mapping = simpeg.maps.IdentityMap(nP=model.size)
        # Build simulations
        sim_3d = magnetics.Simulation3DIntegral(
            survey=survey,
            mesh=mesh_3d,
            chiMap=mapping,
            model_type=model_type,
        )
        eq_sources = magnetics.SimulationEquivalentSourceLayer(
            mesh=tensor_mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=survey,
            chiMap=mapping,
            engine=engine,
            store_sensitivities=store_sensitivities,
            model_type=model_type,
        )
        # Compare predictions of both simulations
        np.testing.assert_allclose(
            sim_3d.dpred(model), eq_sources.dpred(model), atol=1e-7
        )

    @pytest.mark.parametrize("engine", ("geoana", "choclo"))
    def test_forward_vs_simulation_on_disk(
        self,
        tensor_mesh,
        mesh_bottom,
        mesh_top,
        magnetic_survey,
        engine,
        tmp_path,
    ):
        """
        Test forward vs simulation storing sensitivities on disk.
        """
        # Build 3D mesh that is analogous to the 2D mesh with bottom and top
        mesh_3d = get_mesh_3d(tensor_mesh, top=mesh_top, bottom=mesh_bottom)
        # Define sensitivity_dir
        if engine == "geoana":
            sensitivity_path = tmp_path / "sensitivities_geoana"
            sensitivity_path.mkdir()
        elif engine == "choclo":
            sensitivity_path = tmp_path / "sensitivities_choclo"
        # Build simulations
        mapping = get_mapping(tensor_mesh)
        sim_3d = magnetics.Simulation3DIntegral(
            survey=magnetic_survey, mesh=mesh_3d, chiMap=mapping
        )
        eq_sources = magnetics.SimulationEquivalentSourceLayer(
            mesh=tensor_mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=magnetic_survey,
            chiMap=mapping,
            engine=engine,
            store_sensitivities="disk",
            sensitivity_path=str(sensitivity_path),
        )
        # Compare predictions of both simulations
        model = get_block_model(tensor_mesh, 0.2e-3)
        np.testing.assert_allclose(sim_3d.dpred(model), eq_sources.dpred(model))

    @pytest.mark.parametrize("engine", ("geoana", "choclo"))
    @pytest.mark.parametrize("store_sensitivities", ("ram", "forward_only"))
    def test_forward_vs_simulation_with_active_cells(
        self,
        tensor_mesh,
        mesh_bottom,
        mesh_top,
        magnetic_survey,
        engine,
        store_sensitivities,
    ):
        """
        Test forward vs simulation using active cells.
        """
        model = get_block_model(tensor_mesh, 0.2e-3)

        # Define some inactive cells inside the block
        block_cells_indices = np.indices(model.shape).ravel()[model != 0]
        inactive_indices = block_cells_indices[
            : block_cells_indices.size // 2
        ]  # mark half of the cells in the block as inactive
        active_cells = np.ones_like(model, dtype=bool)
        active_cells[inactive_indices] = False
        assert not np.all(active_cells)  # check we do have inactive cells

        # Keep only values of the model in the active cells
        model = model[active_cells]

        # Build 3D mesh that is analogous to the 2D mesh with bottom and top
        mesh_3d = get_mesh_3d(tensor_mesh, top=mesh_top, bottom=mesh_bottom)

        # Build simulations
        mapping = simpeg.maps.IdentityMap(nP=model.size)
        sim_3d = magnetics.Simulation3DIntegral(
            survey=magnetic_survey,
            mesh=mesh_3d,
            chiMap=mapping,
            active_cells=active_cells,
        )
        eq_sources = magnetics.SimulationEquivalentSourceLayer(
            mesh=tensor_mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=magnetic_survey,
            chiMap=mapping,
            engine=engine,
            store_sensitivities=store_sensitivities,
            active_cells=active_cells,
        )
        # Compare predictions of both simulations
        np.testing.assert_allclose(
            sim_3d.dpred(model), eq_sources.dpred(model), atol=1e-7
        )

    @pytest.mark.parametrize("store_sensitivities", ("ram", "forward_only"))
    def test_forward_geoana_choclo(
        self, mesh, mesh_bottom, mesh_top, magnetic_survey, store_sensitivities
    ):
        """Compare forwards using geoana and choclo."""
        # Build simulations
        mapping = get_mapping(mesh)
        kwargs = dict(
            mesh=mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=magnetic_survey,
            chiMap=mapping,
            store_sensitivities=store_sensitivities,
        )
        sim_geoana = magnetics.SimulationEquivalentSourceLayer(
            engine="geoana", **kwargs
        )
        sim_choclo = magnetics.SimulationEquivalentSourceLayer(
            engine="choclo", **kwargs
        )
        model = get_block_model(mesh, 0.2e-3)
        np.testing.assert_allclose(sim_geoana.dpred(model), sim_choclo.dpred(model))

    @pytest.mark.parametrize("store_sensitivities", ("ram", "forward_only"))
    def test_forward_choclo_serial_parallel(
        self, mesh, mesh_bottom, mesh_top, magnetic_survey, store_sensitivities
    ):
        """Test forward using choclo in serial and in parallel."""
        # Build simulations
        mapping = get_mapping(mesh)
        kwargs = dict(
            mesh=mesh,
            cell_z_top=mesh_top,
            cell_z_bottom=mesh_bottom,
            survey=magnetic_survey,
            chiMap=mapping,
            engine="choclo",
            store_sensitivities=store_sensitivities,
        )
        sim_parallel = magnetics.SimulationEquivalentSourceLayer(
            numba_parallel=True, **kwargs
        )
        sim_serial = magnetics.SimulationEquivalentSourceLayer(
            numba_parallel=False, **kwargs
        )
        model = get_block_model(mesh, 0.2e-3)
        np.testing.assert_allclose(sim_parallel.dpred(model), sim_serial.dpred(model))


class BaseFittingEquivalentSources:
    """
    Base class to test the fitting of equivalent sources with synthetic data.
    """

    def get_mesh_top_bottom(self, mesh, array=False):
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

    def build_synthetic_data(self, simulation, model):
        data = simulation.make_synthetic_data(
            model,
            relative_error=0.0,
            noise_floor=1e-3,
            add_noise=True,
            random_seed=1,
        )
        return data

    def build_inversion(self, mesh, simulation, synthetic_data, max_iterations=20):
        """Build inversion problem."""
        # Build data misfit and regularization terms
        data_misfit = simpeg.data_misfit.L2DataMisfit(
            simulation=simulation, data=synthetic_data
        )
        regularization = simpeg.regularization.WeightedLeastSquares(mesh=mesh)
        # Choose optimization
        optimization = ProjectedGNCG(
            maxIter=max_iterations,
            maxIterLS=5,
            maxIterCG=20,
            tolCG=1e-4,
        )
        # Build inverse problem
        inverse_problem = simpeg.inverse_problem.BaseInvProblem(
            data_misfit, regularization, optimization
        )
        # Define directives
        starting_beta = simpeg.directives.BetaEstimate_ByEig(
            beta0_ratio=1e-1, random_seed=42
        )
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


class TestGravityEquivalentSources(BaseFittingEquivalentSources):
    """
    Test fitting gravity equivalent sources with synthetic data.
    """

    @pytest.mark.parametrize("engine", ("geoana", "choclo"))
    @pytest.mark.parametrize(
        "top_bottom_as_array",
        (False, True),
        ids=("top-bottom-float", "top-bottom-array"),
    )
    def test_predictions_on_data_points(
        self,
        tree_mesh,
        gravity_survey,
        top_bottom_as_array,
        engine,
    ):
        """
        Test eq sources predictions on the same data points.

        The equivalent sources should be able to reproduce the same data with
        which they were trained.
        """
        # Get mesh top and bottom
        mesh_top, mesh_bottom = self.get_mesh_top_bottom(
            tree_mesh, array=top_bottom_as_array
        )
        # Build simulation
        mapping = get_mapping(tree_mesh)
        simulation = gravity.SimulationEquivalentSourceLayer(
            tree_mesh,
            mesh_top,
            mesh_bottom,
            survey=gravity_survey,
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


class TestMagneticEquivalentSources(BaseFittingEquivalentSources):
    """
    Test fitting magnetic equivalent sources with synthetic data.
    """

    @pytest.mark.parametrize("engine", ("geoana", "choclo"))
    @pytest.mark.parametrize(
        "top_bottom_as_array",
        (False, True),
        ids=("top-bottom-float", "top-bottom-array"),
    )
    def test_predictions_on_data_points(
        self,
        tree_mesh,
        magnetic_survey,
        top_bottom_as_array,
        engine,
    ):
        """
        Test eq sources predictions on the same data points.

        The equivalent sources should be able to reproduce the same data with
        which they were trained.
        """
        # Get mesh top and bottom
        mesh_top, mesh_bottom = self.get_mesh_top_bottom(
            tree_mesh, array=top_bottom_as_array
        )
        # Build simulation
        mapping = get_mapping(tree_mesh)
        simulation = magnetics.SimulationEquivalentSourceLayer(
            tree_mesh,
            mesh_top,
            mesh_bottom,
            survey=magnetic_survey,
            chiMap=mapping,
            engine=engine,
        )
        # Generate synthetic data
        model = get_block_model(tree_mesh, 1e-3)
        synthetic_data = self.build_synthetic_data(simulation, model)
        # Build inversion
        inversion = self.build_inversion(
            tree_mesh, simulation, synthetic_data, max_iterations=40
        )
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
