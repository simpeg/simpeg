"""
Test BasePFSimulation class
"""

import pytest
import numpy as np
from discretize import CylindricalMesh, TensorMesh, TreeMesh

import simpeg
from simpeg.potential_fields.base import BasePFSimulation
from simpeg.survey import BaseSurvey
from simpeg.potential_fields import gravity, magnetics


@pytest.fixture
def mock_simulation_class():
    """
    Mock simulation class as child of BasePFSimulation
    """

    class MockSimulation(BasePFSimulation):
        @property
        def G(self):
            """Define a dummy G property to avoid warnings on tests."""
            pass

    return MockSimulation


@pytest.fixture
def tensor_mesh():
    """
    Return sample TensorMesh
    """
    h = (3, 3, 3)
    return TensorMesh(h)


@pytest.fixture
def tree_mesh():
    """
    Return sample TensorMesh
    """
    h = (4, 4, 4)
    mesh = TreeMesh(h)
    mesh.refine_points(points=(0, 0, 0), level=2)
    return mesh


@pytest.fixture
def mock_survey_class():
    """
    Mock survey class as child of BaseSurvey
    """

    class MockSurvey(BaseSurvey):
        pass

    return MockSurvey


class TestEngine:
    """
    Test the engine property and some of its relations with other attributes
    """

    def test_invalid_engine(self, tensor_mesh, mock_simulation_class):
        """
        Test if error is raised after invalid engine
        """
        engine = "invalid engine"
        msg = rf"'engine' must be in \('geoana', 'choclo'\). Got '{engine}'"
        with pytest.raises(ValueError, match=msg):
            mock_simulation_class(tensor_mesh, engine=engine)

    def test_invalid_engine_without_choclo(
        self, tensor_mesh, mock_simulation_class, monkeypatch
    ):
        """
        Test error after choosing "choclo" as engine but not being installed
        """
        monkeypatch.setattr(simpeg.potential_fields.base, "choclo", None)
        engine = "choclo"
        msg = "The choclo package couldn't be found."
        with pytest.raises(ImportError, match=msg):
            mock_simulation_class(tensor_mesh, engine=engine)

    def test_sensitivity_path_as_dir(self, tensor_mesh, mock_simulation_class, tmpdir):
        """
        Test error if the sensitivity_path is a dir

        Error should be raised if using ``engine=="choclo"`` and setting
        ``store_sensitivities="disk"``.
        """
        sensitivity_path = str(tmpdir.mkdir("sensitivities"))
        msg = f"The passed sensitivity_path '{sensitivity_path}' is a directory."
        with pytest.raises(ValueError, match=msg):
            mock_simulation_class(
                tensor_mesh,
                engine="choclo",
                store_sensitivities="disk",
                sensitivity_path=sensitivity_path,
            )


class TestGetActiveNodes:
    """
    Tests _get_active_nodes private method
    """

    def test_invalid_mesh(self, tensor_mesh, mock_simulation_class):
        """
        Test error on invalid mesh class
        """
        # Initialize base simulation with valid mesh (so we don't trigger
        # errors in the constructor)
        simulation = mock_simulation_class(tensor_mesh)
        # Assign an invalid mesh to the simulation
        simulation.mesh = CylindricalMesh(tensor_mesh.h)
        msg = "Invalid mesh of type CylindricalMesh."
        with pytest.raises(TypeError, match=msg):
            simulation._get_active_nodes()

    def test_no_inactive_cells_tensor(self, tensor_mesh, mock_simulation_class):
        """
        Test _get_active_nodes when all cells are active on a tensor mesh
        """
        simulation = mock_simulation_class(tensor_mesh)
        active_nodes, active_cell_nodes = simulation._get_active_nodes()
        np.testing.assert_equal(active_nodes, tensor_mesh.nodes)
        np.testing.assert_equal(active_cell_nodes, tensor_mesh.cell_nodes)

    def test_no_inactive_cells_tree(self, tree_mesh, mock_simulation_class):
        """
        Test _get_active_nodes when all cells are active on a tree mesh
        """
        simulation = mock_simulation_class(tree_mesh)
        active_nodes, active_cell_nodes = simulation._get_active_nodes()
        np.testing.assert_equal(active_nodes, tree_mesh.total_nodes)
        np.testing.assert_equal(active_cell_nodes, tree_mesh.cell_nodes)

    def test_inactive_cells_tensor(self, tensor_mesh, mock_simulation_class):
        """
        Test _get_active_nodes with some inactive cells on a tensor mesh
        """
        # Define active cells: only the first cell is active
        active_cells = np.zeros(tensor_mesh.n_cells, dtype=bool)
        active_cells[0] = True
        # Initialize simulation
        simulation = mock_simulation_class(tensor_mesh, active_cells=active_cells)
        # Build expected active_nodes and active_cell_nodes
        expected_active_nodes = tensor_mesh.nodes[tensor_mesh[0].nodes]
        expected_active_cell_nodes = np.atleast_2d(np.arange(8, dtype=int))
        # Test method
        active_nodes, active_cell_nodes = simulation._get_active_nodes()
        np.testing.assert_equal(active_nodes, expected_active_nodes)
        np.testing.assert_equal(active_cell_nodes, expected_active_cell_nodes)

    def test_inactive_cells_tree(self, tree_mesh, mock_simulation_class):
        """
        Test _get_active_nodes with some inactive cells on a tensor mesh
        """
        # Define active cells: only the first cell is active
        active_cells = np.zeros(tree_mesh.n_cells, dtype=bool)
        active_cells[0] = True

        # Initialize simulation
        simulation = mock_simulation_class(tree_mesh, active_cells=active_cells)

        # Build expected active_nodes (in the right order for a single cell)
        expected_active_nodes = [
            [0, 0, 0],
            [0.25, 0, 0],
            [0, 0.25, 0],
            [0.25, 0.25, 0],
            [0, 0, 0.25],
            [0.25, 0, 0.25],
            [0, 0.25, 0.25],
            [0.25, 0.25, 0.25],
        ]

        # Run method
        active_nodes, active_cell_nodes = simulation._get_active_nodes()

        # Check shape of active nodes and check if all of them are there
        assert active_nodes.shape == (8, 3)
        for node in expected_active_nodes:
            assert node in active_nodes

        # Check shape of active_cell_nodes and check if they are in the right
        # order
        assert active_cell_nodes.shape == (1, 8)
        for node, node_index in zip(expected_active_nodes, active_cell_nodes[0]):
            np.testing.assert_equal(node, active_nodes[node_index])


class TestGetComponentsAndReceivers:
    """
    Test _get_components_and_receivers private method
    """

    @pytest.fixture
    def receiver_locations(self):
        receiver_locations = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        return receiver_locations

    @pytest.fixture
    def gravity_survey(self, receiver_locations):
        receiver_locations = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        components = ["gxy", "guv"]
        receivers = gravity.receivers.Point(
            receiver_locations,
            components=components,
        )
        # Define the SourceField and the Survey
        source_field = gravity.sources.SourceField(receiver_list=[receivers])
        return gravity.Survey(source_field)

    @pytest.fixture
    def magnetic_survey(self, receiver_locations):
        receiver_locations = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        components = ["tmi", "bx"]
        receivers = magnetics.receivers.Point(
            receiver_locations,
            components=components,
        )
        # Define the SourceField and the Survey
        source_field = magnetics.sources.UniformBackgroundField(
            receiver_list=[receivers],
            amplitude=55_000,
            inclination=45.0,
            declination=12.0,
        )
        return magnetics.Survey(source_field)

    def test_missing_source_field(
        self, tensor_mesh, mock_survey_class, mock_simulation_class
    ):
        """
        Test error after missing survey in simulation
        """
        survey = mock_survey_class(source_list=None)
        simulation = mock_simulation_class(tensor_mesh, survey=survey)
        msg = "The survey '(.*)' has no 'source_field' attribute."
        with pytest.raises(AttributeError, match=msg):
            # need to iterate over the generator to actually test its code
            [item for item in simulation._get_components_and_receivers()]

    def test_components_and_receivers_gravity(
        self, tensor_mesh, gravity_survey, mock_simulation_class, receiver_locations
    ):
        """
        Test method on a gravity survey
        """
        simulation = mock_simulation_class(tensor_mesh, survey=gravity_survey)
        components_and_receivers = tuple(
            items for items in simulation._get_components_and_receivers()
        )
        # Check we have a single element in the iterator
        assert len(components_and_receivers) == 1
        # Check if components and receiver locations are correct
        components, receivers = components_and_receivers[0]
        assert components == ["gxy", "guv"]
        np.testing.assert_equal(receivers, receiver_locations)

    def test_components_and_receivers_magnetics(
        self, tensor_mesh, magnetic_survey, mock_simulation_class, receiver_locations
    ):
        """
        Test method on a magnetic survey
        """
        simulation = mock_simulation_class(tensor_mesh, survey=magnetic_survey)
        components_and_receivers = tuple(
            items for items in simulation._get_components_and_receivers()
        )
        # Check we have a single element in the iterator
        assert len(components_and_receivers) == 1
        # Check if components and receiver locations are correct
        components, receivers = components_and_receivers[0]
        assert components == ["tmi", "bx"]
        np.testing.assert_equal(receivers, receiver_locations)


class TestInvalidMeshChoclo:
    @pytest.fixture(params=("tensormesh", "treemesh"))
    def mesh(self, request):
        """Sample 2D mesh."""
        hx, hy = [(0.1, 8)], [(0.1, 8)]
        h = (hx, hy)
        if request.param == "tensormesh":
            mesh = TensorMesh(h, "CC")
        else:
            mesh = TreeMesh(h, origin="CC")
            mesh.finalize()
        return mesh

    def test_invalid_mesh_with_choclo(self, mesh, mock_simulation_class):
        """
        Test if simulation raises error when passing an invalid mesh and using choclo
        """
        msg = (
            "Invalid mesh with 2 dimensions. "
            "Only 3D meshes are supported when using 'choclo' as engine."
        )
        with pytest.raises(ValueError, match=msg):
            mock_simulation_class(mesh, engine="choclo")


class TestDeprecationIndActive:
    """
    Test if using the deprecated ind_active argument/property raise warnings/errors
    """

    def test_deprecated_argument(self, tensor_mesh, mock_simulation_class):
        """Test if passing ind_active argument raises warning."""
        ind_active = np.ones(tensor_mesh.n_cells, dtype=bool)
        version_regex = "v[0-9]+.[0-9]+.[0-9]+"
        msg = (
            "'ind_active' has been deprecated and will be removed in "
            f" SimPEG {version_regex}, please use 'active_cells' instead."
        )
        with pytest.warns(FutureWarning, match=msg):
            sim = mock_simulation_class(tensor_mesh, ind_active=ind_active)
        np.testing.assert_allclose(sim.active_cells, ind_active)

    def test_error_both_args(self, tensor_mesh, mock_simulation_class):
        """Test if passing both ind_active and active_cells raises error."""
        ind_active = np.ones(tensor_mesh.n_cells, dtype=bool)
        version_regex = "v[0-9]+.[0-9]+.[0-9]+"
        msg = (
            f"Cannot pass both 'active_cells' and 'ind_active'."
            "'ind_active' has been deprecated and will be removed in "
            f" SimPEG {version_regex}, please use 'active_cells' instead."
        )
        with pytest.raises(TypeError, match=msg):
            mock_simulation_class(
                tensor_mesh, active_cells=ind_active, ind_active=ind_active
            )

    def test_deprecated_property(self, tensor_mesh, mock_simulation_class):
        """Test if passing both ind_active and active_cells raises error."""
        ind_active = np.ones(tensor_mesh.n_cells, dtype=bool)
        simulation = mock_simulation_class(tensor_mesh, active_cells=ind_active)
        with pytest.warns(FutureWarning):
            simulation.ind_active
