import re

import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, aslinearoperator
import discretize
import simpeg
from simpeg import maps
from simpeg.potential_fields import gravity
from simpeg.utils import model_builder
from geoana.gravity import Prism
import numpy as np


class TestsGravitySimulation:
    """
    Test gravity simulation.
    """

    @pytest.fixture
    def blocks(self):
        """Synthetic blocks to build the sample model."""
        block1 = np.array([[-1.6, 1.6], [-1.6, 1.6], [-1.6, 1.6]])
        block2 = np.array([[-0.8, 0.8], [-0.8, 0.8], [-0.8, 0.8]])
        rho1 = 1.0
        rho2 = 2.0
        return (block1, block2), (rho1, rho2)

    @pytest.fixture(params=("tensormesh", "treemesh"))
    def mesh(self, blocks, request):
        """Sample mesh."""
        cs = 0.2
        (block1, _), _ = blocks
        if request.param == "tensormesh":
            hxind, hyind, hzind = tuple([(cs, 42)] for _ in range(3))
            mesh = discretize.TensorMesh([hxind, hyind, hzind], "CCC")
        else:
            h = cs * np.ones(64)
            mesh = discretize.TreeMesh([h, h, h], origin="CCC")
            x0, x1 = block1[:, 0], block1[:, 1]
            mesh.refine_box(x0, x1, levels=9)
        return mesh

    @pytest.fixture
    def simple_mesh(self):
        """Simpler sample mesh, just to use it as a placeholder in some tests."""
        return discretize.TensorMesh([5, 5, 5], "CCC")

    @pytest.fixture
    def density_and_active_cells(self, mesh, blocks):
        """Sample density and active_cells arrays for the sample mesh."""
        # create a model of two blocks, 1 inside the other
        (block1, block2), (rho1, rho2) = blocks
        block1_inds = self.get_block_inds(mesh.cell_centers, block1)
        block2_inds = self.get_block_inds(mesh.cell_centers, block2)
        # Define densities for each block
        model = np.zeros(mesh.n_cells)
        model[block1_inds] = rho1
        model[block2_inds] = rho2
        # Define active cells and reduce model
        active_cells = model != 0.0
        model_reduced = model[active_cells]
        return model_reduced, active_cells

    def get_block_inds(self, grid, block):
        return np.where(
            (grid[:, 0] > block[0, 0])
            & (grid[:, 0] < block[0, 1])
            & (grid[:, 1] > block[1, 0])
            & (grid[:, 1] < block[1, 1])
            & (grid[:, 2] > block[2, 0])
            & (grid[:, 2] < block[2, 1])
        )

    @pytest.fixture
    def receivers_locations(self):
        nx = 5
        ny = 5
        # Create plane of observations
        xr = np.linspace(-20, 20, nx)
        yr = np.linspace(-20, 20, ny)
        x, y = np.meshgrid(xr, yr)
        z = np.ones_like(x) * 3.0
        receivers_locations = np.vstack([a.ravel() for a in (x, y, z)]).T
        return receivers_locations

    def get_analytic_solution(self, blocks, survey):
        """Compute analytical response from dense prism."""
        (block1, block2), (rho1, rho2) = blocks
        # Build prisms (convert densities from g/cc to kg/m3)
        prisms = [
            Prism(block1[:, 0], block1[:, 1], rho1 * 1000),
            Prism(block2[:, 0], block2[:, 1], -rho1 * 1000),
            Prism(block2[:, 0], block2[:, 1], rho2 * 1000),
        ]
        # Forward model the prisms
        components = survey.source_field.receiver_list[0].components
        receivers_locations = survey.source_field.receiver_list[0].locations
        if "gx" in components or "gy" in components or "gz" in components:
            fields = sum(
                prism.gravitational_field(receivers_locations) for prism in prisms
            )
            fields *= 1e5  # convert to mGal from m/s^2
        else:
            fields = sum(
                prism.gravitational_gradient(receivers_locations) for prism in prisms
            )
            fields *= 1e9  # convert to Eotvos from 1/s^2
        return fields

    @pytest.mark.parametrize(
        "engine, parallelism",
        [("geoana", None), ("geoana", 1), ("choclo", False), ("choclo", True)],
        ids=["geoana_serial", "geoana_parallel", "choclo_serial", "choclo_parallel"],
    )
    @pytest.mark.parametrize("store_sensitivities", ("ram", "disk", "forward_only"))
    def test_accelerations_vs_analytic(
        self,
        engine,
        parallelism,
        store_sensitivities,
        tmp_path,
        blocks,
        mesh,
        density_and_active_cells,
        receivers_locations,
    ):
        """
        Test gravity acceleration components against analytic solutions of prisms.
        """
        components = ["gx", "gy", "gz"]
        # Unpack fixtures
        density, active_cells = density_and_active_cells
        # Create survey
        receivers = gravity.Point(receivers_locations, components=components)
        sources = gravity.SourceField([receivers])
        survey = gravity.Survey(sources)
        # Create reduced identity map for Linear Problem
        idenMap = maps.IdentityMap(nP=int(sum(active_cells)))
        # Create simulation
        if engine == "choclo":
            sensitivity_path = tmp_path / "sensitivity_choclo"
            kwargs = dict(numba_parallel=parallelism)
        else:
            sensitivity_path = tmp_path
            kwargs = dict(n_processes=parallelism)
        sim = gravity.Simulation3DIntegral(
            mesh,
            survey=survey,
            rhoMap=idenMap,
            active_cells=active_cells,
            store_sensitivities=store_sensitivities,
            engine=engine,
            sensitivity_path=str(sensitivity_path),
            sensitivity_dtype=np.float64,
            **kwargs,
        )
        data = sim.dpred(density)
        g_x, g_y, g_z = data[0::3], data[1::3], data[2::3]
        solution = self.get_analytic_solution(blocks, survey)
        # Check results
        rtol, atol = 1e-9, 1e-6
        np.testing.assert_allclose(g_x, solution[:, 0], rtol=rtol, atol=atol)
        np.testing.assert_allclose(g_y, solution[:, 1], rtol=rtol, atol=atol)
        np.testing.assert_allclose(g_z, solution[:, 2], rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "engine, parallelism",
        [("geoana", None), ("geoana", 1), ("choclo", False), ("choclo", True)],
        ids=["geoana_serial", "geoana_parallel", "choclo_serial", "choclo_parallel"],
    )
    @pytest.mark.parametrize("store_sensitivities", ("ram", "disk", "forward_only"))
    def test_tensor_vs_analytic(
        self,
        engine,
        parallelism,
        store_sensitivities,
        tmp_path,
        blocks,
        mesh,
        density_and_active_cells,
        receivers_locations,
    ):
        """
        Test tensor components against analytic solutions of prisms.
        """
        components = ["gxx", "gxy", "gxz", "gyy", "gyz", "gzz"]
        # Unpack fixtures
        density, active_cells = density_and_active_cells
        # Create survey
        receivers = gravity.Point(receivers_locations, components=components)
        sources = gravity.SourceField([receivers])
        survey = gravity.Survey(sources)
        # Create reduced identity map for Linear Problem
        idenMap = maps.IdentityMap(nP=int(sum(active_cells)))
        # Create simulation
        if engine == "choclo":
            sensitivity_path = tmp_path / "sensitivity_choclo"
            kwargs = dict(numba_parallel=parallelism)
        else:
            sensitivity_path = tmp_path
            kwargs = dict(n_processes=parallelism)
        sim = gravity.Simulation3DIntegral(
            mesh,
            survey=survey,
            rhoMap=idenMap,
            active_cells=active_cells,
            store_sensitivities=store_sensitivities,
            engine=engine,
            sensitivity_path=str(sensitivity_path),
            sensitivity_dtype=np.float64,
            **kwargs,
        )
        data = sim.dpred(density)
        g_xx, g_xy, g_xz = data[0::6], data[1::6], data[2::6]
        g_yy, g_yz, g_zz = data[3::6], data[4::6], data[5::6]
        solution = self.get_analytic_solution(blocks, survey)
        # Check results
        rtol, atol = 2e-6, 1e-6
        np.testing.assert_allclose(g_xx, solution[..., 0, 0], rtol=rtol, atol=atol)
        np.testing.assert_allclose(g_xy, solution[..., 0, 1], rtol=rtol, atol=atol)
        np.testing.assert_allclose(g_xz, solution[..., 0, 2], rtol=rtol, atol=atol)
        np.testing.assert_allclose(g_yy, solution[..., 1, 1], rtol=rtol, atol=atol)
        np.testing.assert_allclose(g_yz, solution[..., 1, 2], rtol=rtol, atol=atol)
        np.testing.assert_allclose(g_zz, solution[..., 2, 2], rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "engine, parallelism",
        [("geoana", 1), ("geoana", None), ("choclo", False), ("choclo", True)],
        ids=["geoana_serial", "geoana_parallel", "choclo_serial", "choclo_parallel"],
    )
    @pytest.mark.parametrize("store_sensitivities", ("ram", "disk", "forward_only"))
    def test_guv_vs_analytic(
        self,
        engine,
        parallelism,
        store_sensitivities,
        tmp_path,
        blocks,
        mesh,
        density_and_active_cells,
        receivers_locations,
    ):
        """
        Test guv tensor component against analytic solutions of prisms.
        """
        components = ["guv"]
        # Unpack fixtures
        density, active_cells = density_and_active_cells
        # Create survey
        receivers = gravity.Point(receivers_locations, components=components)
        sources = gravity.SourceField([receivers])
        survey = gravity.Survey(sources)
        # Create reduced identity map for Linear Problem
        idenMap = maps.IdentityMap(nP=int(sum(active_cells)))
        # Create simulation
        if engine == "choclo":
            sensitivity_path = tmp_path / "sensitivity_choclo"
            kwargs = dict(numba_parallel=parallelism)
        else:
            sensitivity_path = tmp_path
            kwargs = dict(n_processes=parallelism)
        sim = gravity.Simulation3DIntegral(
            mesh,
            survey=survey,
            rhoMap=idenMap,
            active_cells=active_cells,
            store_sensitivities=store_sensitivities,
            engine=engine,
            sensitivity_path=str(sensitivity_path),
            sensitivity_dtype=np.float64,
            **kwargs,
        )
        g_uv = sim.dpred(density)
        solution = self.get_analytic_solution(blocks, survey)
        g_xx_solution = solution[..., 0, 0]
        g_yy_solution = solution[..., 1, 1]
        g_uv_solution = 0.5 * (g_yy_solution - g_xx_solution)
        # Check results
        rtol, atol = 2e-6, 1e-6
        np.testing.assert_allclose(g_uv, g_uv_solution, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("engine", ("choclo", "geoana"))
    @pytest.mark.parametrize("store_sensitivities", ("ram", "disk", "forward_only"))
    def test_sensitivity_dtype(
        self,
        engine,
        store_sensitivities,
        simple_mesh,
        receivers_locations,
        tmp_path,
    ):
        """Test sensitivity_dtype."""
        # Create survey
        receivers = gravity.Point(receivers_locations, components="gz")
        sources = gravity.SourceField([receivers])
        survey = gravity.Survey(sources)
        # Create reduced identity map for Linear Problem
        active_cells = np.ones(simple_mesh.n_cells, dtype=bool)
        idenMap = maps.IdentityMap(nP=simple_mesh.n_cells)
        # Create simulation
        sensitivity_path = tmp_path
        if engine == "choclo":
            sensitivity_path /= "dummy"
        simulation = gravity.Simulation3DIntegral(
            simple_mesh,
            survey=survey,
            rhoMap=idenMap,
            active_cells=active_cells,
            engine=engine,
            store_sensitivities=store_sensitivities,
            sensitivity_path=str(sensitivity_path),
        )
        # sensitivity_dtype should be float64 when running forward only,
        # but float32 in other cases
        if store_sensitivities == "forward_only":
            assert simulation.sensitivity_dtype is np.float64
        else:
            assert simulation.sensitivity_dtype is np.float32

    @pytest.mark.parametrize("invalid_dtype", (float, np.float16))
    def test_invalid_sensitivity_dtype_assignment(self, simple_mesh, invalid_dtype):
        """
        Test invalid sensitivity_dtype assignment
        """
        simulation = gravity.Simulation3DIntegral(
            simple_mesh,
        )
        # Check if error is raised
        msg = "sensitivity_dtype must be either np.float32 or np.float64."
        with pytest.raises(TypeError, match=msg):
            simulation.sensitivity_dtype = invalid_dtype

    def test_invalid_engine(self, simple_mesh):
        """Test if error is raised after invalid engine."""
        engine = "invalid engine"
        msg = rf"'engine' must be in \('geoana', 'choclo'\). Got '{engine}'"
        with pytest.raises(ValueError, match=msg):
            gravity.Simulation3DIntegral(simple_mesh, engine=engine)

    def test_choclo_and_n_proceesses(self, simple_mesh):
        """Check if warning is raised after passing n_processes with choclo engine."""
        msg = "The 'n_processes' will be ignored when selecting 'choclo'"
        with pytest.warns(UserWarning, match=msg):
            simulation = gravity.Simulation3DIntegral(
                simple_mesh, engine="choclo", n_processes=2
            )
        # Check if n_processes was overwritten and set to None
        assert simulation.n_processes is None

    def test_choclo_and_sensitivity_path_as_dir(self, simple_mesh, tmp_path):
        """
        Check if error is raised when sensitivity_path is a dir with choclo engine.
        """
        # Create a sensitivity_path directory
        sensitivity_path = tmp_path / "sensitivity_dummy"
        sensitivity_path.mkdir()
        # Check if error is raised
        msg = re.escape(
            f"The passed sensitivity_path '{str(sensitivity_path)}' is a directory"
        )
        with pytest.raises(ValueError, match=msg):
            gravity.Simulation3DIntegral(
                simple_mesh,
                store_sensitivities="disk",
                sensitivity_path=str(sensitivity_path),
                engine="choclo",
            )

    def test_sensitivities_on_disk(self, simple_mesh, receivers_locations, tmp_path):
        """
        Test if sensitivity matrix is correctly being stored in disk when asked
        """
        # Build survey
        receivers = gravity.Point(receivers_locations, components="gz")
        sources = gravity.SourceField([receivers])
        survey = gravity.Survey(sources)
        # Build simulation
        sensitivities_path = tmp_path / "sensitivities"
        simulation = gravity.Simulation3DIntegral(
            mesh=simple_mesh,
            survey=survey,
            store_sensitivities="disk",
            sensitivity_path=str(sensitivities_path),
            engine="choclo",
        )
        simulation.G
        # Check if sensitivity matrix was stored in disk and is a memmap
        assert sensitivities_path.is_file()
        assert type(simulation.G) is np.memmap

    def test_sensitivities_on_ram(self, simple_mesh, receivers_locations, tmp_path):
        """
        Test if sensitivity matrix is correctly being allocated in memory when asked
        """
        # Build survey
        receivers = gravity.Point(receivers_locations, components="gz")
        sources = gravity.SourceField([receivers])
        survey = gravity.Survey(sources)
        # Build simulation
        simulation = gravity.Simulation3DIntegral(
            mesh=simple_mesh,
            survey=survey,
            store_sensitivities="ram",
            engine="choclo",
        )
        simulation.G
        # Check if sensitivity matrix is a Numpy array (stored in memory)
        assert type(simulation.G) is np.ndarray

    def test_choclo_missing(self, simple_mesh, monkeypatch):
        """
        Check if error is raised when choclo is missing and chosen as engine.
        """
        # Monkeypatch choclo in simpeg.potential_fields.base
        monkeypatch.setattr(simpeg.potential_fields.base, "choclo", None)
        # Check if error is raised
        msg = "The choclo package couldn't be found."
        with pytest.raises(ImportError, match=msg):
            gravity.Simulation3DIntegral(simple_mesh, engine="choclo")


class BaseFixtures:
    """
    Base test class with some fixtures.
    """

    @pytest.fixture
    def survey(self):
        # Observation points
        x = np.linspace(-20.0, 20.0, 4)
        x, y = np.meshgrid(x, x)
        z = 5.0 * np.ones_like(x)
        coordinates = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
        receivers = gravity.receivers.Point(coordinates, components="gz")
        source_field = gravity.sources.SourceField(receiver_list=[receivers])
        survey = gravity.survey.Survey(source_field)
        return survey

    @pytest.fixture
    def mesh(self):
        # Mesh
        dh = 5.0
        hx = [(dh, 4)]
        mesh = discretize.TensorMesh([hx, hx, hx], "CCN")
        return mesh

    @pytest.fixture
    def densities(self, mesh):
        # Define densities
        densities = 1e-10 * np.ones(mesh.n_cells)
        ind_sphere = model_builder.get_indices_sphere(
            np.r_[0.0, 0.0, -20.0], 10.0, mesh.cell_centers
        )
        densities[ind_sphere] = 0.2
        return densities


class TestJacobianGravity(BaseFixtures):
    """
    Test methods related to Jacobian matrix in gravity simulation.
    """

    atol_ratio = 1e-7

    @pytest.fixture(params=["identity_map", "exp_map"])
    def mapping(self, mesh, request):
        mapping = (
            maps.IdentityMap(nP=mesh.n_cells)
            if request.param == "identity_map"
            else maps.ExpMap(nP=mesh.n_cells)
        )
        return mapping

    @pytest.mark.parametrize("engine", ["choclo", "geoana"])
    def test_getJ_as_array(self, survey, mesh, densities, mapping, engine):
        """
        Test the getJ method when J is an array in memory.
        """
        simulation = gravity.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            rhoMap=mapping,
            store_sensitivities="ram",
            engine=engine,
        )
        model = mapping * densities
        jac = simulation.getJ(model)
        assert isinstance(jac, np.ndarray)
        # With an identity mapping, the jacobian should be the same as G.
        # With an exp mapping, the jacobian should be G @ the mapping derivative.
        identity_map = type(mapping) is maps.IdentityMap
        expected_jac = (
            simulation.G if identity_map else simulation.G @ mapping.deriv(model)
        )
        np.testing.assert_allclose(jac, expected_jac)

    def test_getJ_as_linear_operator(self, survey, mesh, densities, mapping):
        """
        Test the getJ method when J is a linear operator.
        """
        simulation = gravity.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            rhoMap=mapping,
            store_sensitivities="forward_only",
            engine="choclo",
        )
        model = mapping * densities
        jac = simulation.getJ(model)
        assert isinstance(jac, LinearOperator)
        result = jac @ model
        expected_result = simulation.G @ (mapping.deriv(model).diagonal() * model)
        np.testing.assert_allclose(result, expected_result)

    def test_getJ_as_linear_operator_not_implemented(
        self, survey, mesh, densities, mapping
    ):
        """
        Test getJ raises NotImplementedError when forward only with geoana.
        """
        simulation = gravity.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            rhoMap=mapping,
            store_sensitivities="forward_only",
            engine="geoana",
        )
        model = mapping * densities
        msg = re.escape(
            "Accessing matrix G with "
            'store_sensitivities="forward_only" and engine="geoana" '
            "hasn't been implemented yet."
        )
        with pytest.raises(NotImplementedError, match=msg):
            simulation.getJ(model)

    @pytest.mark.parametrize(
        ("engine", "store_sensitivities"),
        [
            ("choclo", "ram"),
            ("choclo", "forward_only"),
            ("geoana", "ram"),
            pytest.param(
                "geoana",
                "forward_only",
                marks=pytest.mark.xfail(reason="not implemented"),
            ),
        ],
    )
    def test_Jvec(self, survey, mesh, densities, mapping, engine, store_sensitivities):
        """
        Test the Jvec method.
        """
        simulation = gravity.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            rhoMap=mapping,
            store_sensitivities=store_sensitivities,
            engine=engine,
        )
        model = mapping * densities

        vector = np.random.default_rng(seed=42).uniform(size=densities.size)
        dpred = simulation.Jvec(model, vector)

        identity_map = type(mapping) is maps.IdentityMap
        expected_jac = (
            simulation.G
            if identity_map
            else simulation.G @ aslinearoperator(mapping.deriv(model))
        )
        expected_dpred = expected_jac @ vector

        atol = np.max(np.abs(expected_dpred)) * self.atol_ratio
        np.testing.assert_allclose(dpred, expected_dpred, atol=atol)

    @pytest.mark.parametrize(
        ("engine", "store_sensitivities"),
        [
            ("choclo", "ram"),
            ("choclo", "forward_only"),
            ("geoana", "ram"),
            pytest.param(
                "geoana",
                "forward_only",
                marks=pytest.mark.xfail(reason="not implemented"),
            ),
        ],
    )
    def test_Jtvec(self, survey, mesh, densities, mapping, engine, store_sensitivities):
        """
        Test the Jtvec method.
        """
        simulation = gravity.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            rhoMap=mapping,
            store_sensitivities=store_sensitivities,
            engine=engine,
        )
        model = mapping * densities

        vector = np.random.default_rng(seed=42).uniform(size=survey.nD)
        result = simulation.Jtvec(model, vector)

        identity_map = type(mapping) is maps.IdentityMap
        expected_jac = (
            simulation.G
            if identity_map
            else simulation.G @ aslinearoperator(mapping.deriv(model))
        )
        expected = expected_jac.T @ vector

        atol = np.max(np.abs(result)) * self.atol_ratio
        np.testing.assert_allclose(result, expected, atol=atol)

    @pytest.mark.parametrize(
        "engine",
        [
            "choclo",
            pytest.param("geoana", marks=pytest.mark.xfail(reason="not implemented")),
        ],
    )
    @pytest.mark.parametrize("method", ["Jvec", "Jtvec"])
    def test_array_vs_linear_operator(
        self, survey, mesh, densities, mapping, engine, method
    ):
        """
        Test methods when using "ram" and "forward_only".

        They should give the same results.
        """
        simulation_lo, simulation_ram = (
            gravity.simulation.Simulation3DIntegral(
                survey=survey,
                mesh=mesh,
                rhoMap=mapping,
                store_sensitivities=store,
                engine=engine,
            )
            for store in ("forward_only", "ram")
        )
        match method:
            case "Jvec":
                vector_size = densities.size
            case "Jtvec":
                vector_size = survey.nD
            case _:
                raise ValueError(f"Invalid method '{method}'")
        vector = np.random.default_rng(seed=42).uniform(size=vector_size)
        model = mapping * densities
        result_lo = getattr(simulation_lo, method)(model, vector)
        result_ram = getattr(simulation_ram, method)(model, vector)
        atol = np.max(np.abs(result_ram)) * self.atol_ratio
        np.testing.assert_allclose(result_lo, result_ram, atol=atol)

    @pytest.mark.parametrize("engine", ["choclo", "geoana"])
    @pytest.mark.parametrize("weights", [True, False])
    def test_getJtJdiag(self, survey, mesh, densities, mapping, engine, weights):
        """
        Test the ``getJtJdiag`` method with G as an array in memory.
        """
        simulation = gravity.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            rhoMap=mapping,
            store_sensitivities="ram",
            engine=engine,
        )
        model = mapping * densities
        kwargs = {}
        if weights:
            w_matrix = diags(np.random.default_rng(seed=42).uniform(size=survey.nD))
            kwargs = {"W": w_matrix}
        jtj_diag = simulation.getJtJdiag(model, **kwargs)

        identity_map = type(mapping) is maps.IdentityMap
        expected_jac = (
            simulation.G if identity_map else simulation.G @ mapping.deriv(model)
        )
        if weights:
            expected = np.diag(expected_jac.T @ w_matrix.T @ w_matrix @ expected_jac)
        else:
            expected = np.diag(expected_jac.T @ expected_jac)

        atol = np.max(np.abs(jtj_diag)) * self.atol_ratio
        np.testing.assert_allclose(jtj_diag, expected, atol=atol)

    @pytest.mark.parametrize(
        "engine",
        [
            "choclo",
            pytest.param("geoana", marks=pytest.mark.xfail(reason="not implemented")),
        ],
    )
    @pytest.mark.parametrize("weights", [True, False])
    def test_getJtJdiag_forward_only(
        self, survey, mesh, densities, mapping, engine, weights
    ):
        """
        Test the ``getJtJdiag`` method without building G.
        """
        simulation, simulation_ram = (
            gravity.simulation.Simulation3DIntegral(
                survey=survey,
                mesh=mesh,
                rhoMap=mapping,
                store_sensitivities=store,
                engine=engine,
            )
            for store in ("forward_only", "ram")
        )
        model = mapping * densities
        kwargs = {}
        if weights:
            weights = np.random.default_rng(seed=42).uniform(size=survey.nD)
            kwargs = {"W": diags(np.sqrt(weights))}
        jtj_diag = simulation.getJtJdiag(model, **kwargs)
        jtj_diag_ram = simulation_ram.getJtJdiag(model, **kwargs)

        atol = np.max(np.abs(jtj_diag)) * self.atol_ratio
        np.testing.assert_allclose(jtj_diag, jtj_diag_ram, atol=atol)

    @pytest.mark.parametrize("engine", ("choclo", "geoana"))
    def test_getJtJdiag_caching(self, survey, mesh, densities, mapping, engine):
        """
        Test the caching behaviour of the ``getJtJdiag`` method.
        """
        simulation = gravity.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            rhoMap=mapping,
            store_sensitivities="ram",
            engine=engine,
        )
        # Get diagonal of J.T @ J without any weight
        model = mapping * densities
        jtj_diagonal_1 = simulation.getJtJdiag(model)
        assert hasattr(simulation, "_gtg_diagonal")
        assert hasattr(simulation, "_weights_sha256")
        gtg_diagonal_1 = simulation._gtg_diagonal
        weights_sha256_1 = simulation._weights_sha256

        # Compute it again and make sure we get the same result
        np.testing.assert_allclose(jtj_diagonal_1, simulation.getJtJdiag(model))

        # Get a new diagonal with weights
        weights_matrix = diags(
            np.random.default_rng(seed=42).uniform(size=simulation.survey.nD)
        )
        jtj_diagonal_2 = simulation.getJtJdiag(model, W=weights_matrix)
        assert hasattr(simulation, "_gtg_diagonal")
        assert hasattr(simulation, "_weights_sha256")
        gtg_diagonal_2 = simulation._gtg_diagonal
        weights_sha256_2 = simulation._weights_sha256

        # The two results should be different
        assert not np.array_equal(jtj_diagonal_1, jtj_diagonal_2)
        assert not np.array_equal(gtg_diagonal_1, gtg_diagonal_2)
        assert weights_sha256_1.digest() != weights_sha256_2.digest()


class TestGLinearOperator(BaseFixtures):
    """
    Test G as a linear operator.
    """

    @pytest.fixture
    def mapping(self, mesh):
        return maps.IdentityMap(nP=mesh.n_cells)

    def test_not_implemented(self, survey, mesh, mapping):
        """
        Test NotImplementedError when using geoana as engine.
        """
        simulation = gravity.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            rhoMap=mapping,
            store_sensitivities="forward_only",
            engine="geoana",
        )
        msg = re.escape(
            "Accessing matrix G with "
            'store_sensitivities="forward_only" and engine="geoana" '
            "hasn't been implemented yet."
        )
        with pytest.raises(NotImplementedError, match=msg):
            simulation.G

    @pytest.mark.parametrize("parallel", [True, False])
    def test_G_dot_m(self, survey, mesh, mapping, densities, parallel):
        """Test G @ m."""
        simulation, simulation_ram = (
            gravity.simulation.Simulation3DIntegral(
                survey=survey,
                mesh=mesh,
                rhoMap=mapping,
                store_sensitivities=store,
                engine="choclo",
                numba_parallel=parallel,
            )
            for store in ("forward_only", "ram")
        )
        assert isinstance(simulation.G, LinearOperator)
        assert isinstance(simulation_ram.G, np.ndarray)
        np.testing.assert_allclose(
            simulation.G @ densities, simulation_ram.G @ densities
        )

    @pytest.mark.parametrize("parallel", [True, False])
    def test_G_t_dot_v(self, survey, mesh, mapping, parallel):
        """Test G.T @ v."""
        simulation, simulation_ram = (
            gravity.simulation.Simulation3DIntegral(
                survey=survey,
                mesh=mesh,
                rhoMap=mapping,
                store_sensitivities=store,
                engine="choclo",
                numba_parallel=parallel,
            )
            for store in ("forward_only", "ram")
        )
        assert isinstance(simulation.G, LinearOperator)
        assert isinstance(simulation_ram.G, np.ndarray)
        vector = np.random.default_rng(seed=42).uniform(size=survey.nD)
        np.testing.assert_allclose(simulation.G.T @ vector, simulation_ram.G.T @ vector)


class TestDeprecationWarning(BaseFixtures):
    """
    Test warnings after deprecated properties or methods of the simulation class.
    """

    def test_gtg_diagonal(self, survey, mesh):
        """Test deprecation warning on gtg_diagonal property."""
        mapping = maps.IdentityMap(nP=mesh.n_cells)
        simulation = gravity.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            rhoMap=mapping,
            store_sensitivities="ram",
            engine="choclo",
        )
        msg = re.escape(
            "The `gtg_diagonal` property has been deprecated. "
            "It will be removed in SimPEG v0.25.0.",
        )
        with pytest.warns(FutureWarning, match=msg):
            simulation.gtg_diagonal


class TestConversionFactor:
    """Test _get_conversion_factor function."""

    @pytest.mark.parametrize(
        "component",
        ("gx", "gy", "gz", "gxx", "gyy", "gzz", "gxy", "gxz", "gyz", "guv"),
    )
    def test_conversion_factor(self, component):
        """
        Test _get_conversion_factor function with valid components
        """
        conversion_factor = gravity.simulation._get_conversion_factor(component)
        if len(component) == 2:
            assert conversion_factor == 1e5 * 1e3  # SI to mGal and g/cc to kg/m3
        else:
            assert conversion_factor == 1e9 * 1e3  # SI to Eotvos and g/cc to kg/m3

    def test_invalid_conversion_factor(self):
        """
        Test invalid conversion factor _get_conversion_factor function
        """
        component = "invalid-component"
        with pytest.raises(ValueError, match=f"Invalid component '{component}'"):
            gravity.simulation._get_conversion_factor(component)
