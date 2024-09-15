import discretize
import numpy as np
import pytest
from geoana.em.static import MagneticPrism
from scipy.constants import mu_0

import simpeg
from simpeg import maps, utils
from simpeg.potential_fields import magnetics as mag


def get_block_inds(grid: np.ndarray, block: np.ndarray) -> np.ndarray:
    """
    Get the indices for a block

    Parameters
    ----------
    grid : np.ndarray
        (n, 3) array of xyz locations
    block : np.ndarray
        (3, 2) array of (xmin, xmax), (ymin, ymax), (zmin, zmax) dimensions of
        the block.

    Returns
    -------
    np.ndarray
        boolean array of indices corresponding to the block
    """

    return np.where(
        (grid[:, 0] > block[0, 0])
        & (grid[:, 0] < block[0, 1])
        & (grid[:, 1] > block[1, 0])
        & (grid[:, 1] < block[1, 1])
        & (grid[:, 2] > block[2, 0])
        & (grid[:, 2] < block[2, 1])
    )


def create_block_model(
    mesh: discretize.TensorMesh,
    blocks: tuple[np.ndarray, ...],
    block_params: tuple[float, ...] | tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a magnetic model from a sequence of blocks

    Parameters
    ----------
    mesh : discretize.TensorMesh
        TensorMesh object to put the model on
    blocks : Tuple[np.ndarray, ...]
        Tuple of block definitions (each element is (3, 2) array of
        (xmin, xmax), (ymin, ymax), (zmin, zmax) dimensions of the block)
    block_params : Tuple[float, ...]
        Tuple of parameters to assign for each block. Must be the same length
        as ``blocks``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of the magnetic model and active_cells (a boolean array)

    Raises
    ------
    ValueError
        if ``blocks`` and ``block_params`` have incompatible dimensions
    """
    if len(blocks) != len(block_params):
        raise ValueError(
            "'blocks' and 'block_params' must have the same number of elements"
        )
    model = np.zeros((mesh.n_cells, np.atleast_1d(block_params[0]).shape[0]))
    for block, params in zip(blocks, block_params):
        block_ind = get_block_inds(mesh.cell_centers, block)
        model[block_ind] = params
    active_cells = np.any(np.abs(model) > 0, axis=1)
    return model.squeeze(), active_cells


def create_mag_survey(
    components: list[str],
    receiver_locations: np.ndarray,
    inducing_field_params: tuple[float, float, float],
) -> mag.Survey:
    """
    create a magnetic Survey

    Parameters
    ----------
    components : List[str]
        List of components to model
    receiver_locations : np.ndarray
        (n, 3) array of xyz receiver locations
    inducing_field_params : Tuple[float, float, float]
        amplitude, inclination, and declination of the inducing field

    Returns
    -------
    mag.Survey
        a magnetic Survey instance
    """

    receivers = mag.Point(receiver_locations, components=components)
    strenght, inclination, declination = inducing_field_params
    source_field = mag.UniformBackgroundField(
        receiver_list=[receivers],
        amplitude=strenght,
        inclination=inclination,
        declination=declination,
    )
    return mag.Survey(source_field)


class TestsMagSimulation:
    """
    Test mag simulation against the analytic solutions single prisms
    """

    @pytest.fixture
    def mag_mesh(self) -> discretize.TensorMesh:
        """
        a small tensor mesh for testing magnetic simulations

        Returns
        -------
        discretize.TensorMesh
            the tensor mesh for testing
        """
        # Define a mesh
        cs = 0.2
        hxind = [(cs, 41)]
        hyind = [(cs, 41)]
        hzind = [(cs, 41)]
        mesh = discretize.TensorMesh([hxind, hyind, hzind], "CCC")
        return mesh

    @pytest.fixture
    def two_blocks(self) -> tuple[np.ndarray, np.ndarray]:
        """
        The parameters defining two blocks

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (3, 2) arrays of (xmin, xmax), (ymin, ymax), (zmin, zmax)
            dimensions of each block.
        """
        block1 = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
        block2 = np.array([[-0.7, 0.7], [-0.7, 0.7], [-0.7, 0.7]])
        return block1, block2

    @pytest.fixture
    def receiver_locations(self) -> np.ndarray:
        """
        a grid of receivers for testing

        Returns
        -------
        np.ndarray
            (n, 3) array of receiver locations
        """
        # Create plane of observations
        nx, ny = 5, 5
        xr = np.linspace(-20, 20, nx)
        yr = np.linspace(-20, 20, ny)
        X, Y = np.meshgrid(xr, yr)
        Z = np.ones_like(X) * 3.0
        return np.c_[X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]

    @pytest.fixture
    def inducing_field(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """
        inducing field

        Return inducing field as amplitude and angles and as vector components.

        Returns
        -------
        tuple[tuple[float, float, float], tuple[float, float, float]]
            (amplitude, inclination, declination), (b_x, b_y, b_z)
        """
        h0_amplitude, h0_inclination, h0_declination = (50000.0, 60.0, 250.0)
        b0 = mag.analytics.IDTtoxyz(-h0_inclination, h0_declination, h0_amplitude)
        return (h0_amplitude, h0_inclination, h0_declination), b0

    @pytest.mark.parametrize(
        "engine,parallel_kwargs",
        [
            ("geoana", {"n_processes": None}),
            ("geoana", {"n_processes": 1}),
            ("choclo", {"numba_parallel": False}),
            ("choclo", {"numba_parallel": True}),
        ],
        ids=["geoana_serial", "geoana_parallel", "choclo_serial", "choclo_parallel"],
    )
    @pytest.mark.parametrize("store_sensitivities", ("ram", "disk", "forward_only"))
    def test_magnetic_field_and_tmi_w_susceptibility(
        self,
        engine,
        parallel_kwargs,
        store_sensitivities,
        tmp_path,
        mag_mesh,
        two_blocks,
        receiver_locations,
        inducing_field,
    ):
        """
        Test forwarding the magnetic field and tmi (with susceptibility as model)
        """
        inducing_field_params, b0 = inducing_field

        chi1 = 0.01
        chi2 = 0.02
        model, active_cells = create_block_model(mag_mesh, two_blocks, (chi1, chi2))
        model_reduced = model[active_cells]
        # Create reduced identity map for Linear Problem
        identity_map = maps.IdentityMap(nP=int(sum(active_cells)))

        survey = create_mag_survey(
            components=["bx", "by", "bz", "tmi"],
            receiver_locations=receiver_locations,
            inducing_field_params=inducing_field_params,
        )

        sim = mag.Simulation3DIntegral(
            mag_mesh,
            survey=survey,
            chiMap=identity_map,
            ind_active=active_cells,
            sensitivity_path=str(tmp_path / f"{engine}"),
            store_sensitivities=store_sensitivities,
            engine=engine,
            **parallel_kwargs,
        )

        data = sim.dpred(model_reduced)
        d_x = data[0::4]
        d_y = data[1::4]
        d_z = data[2::4]
        d_t = data[3::4]

        # Compute analytical response from magnetic prism
        block1, block2 = two_blocks
        prism_1 = MagneticPrism(block1[:, 0], block1[:, 1], chi1 * b0 / mu_0)
        prism_2 = MagneticPrism(block2[:, 0], block2[:, 1], -chi1 * b0 / mu_0)
        prism_3 = MagneticPrism(block2[:, 0], block2[:, 1], chi2 * b0 / mu_0)

        d = (
            prism_1.magnetic_flux_density(receiver_locations)
            + prism_2.magnetic_flux_density(receiver_locations)
            + prism_3.magnetic_flux_density(receiver_locations)
        )

        # TMI projection
        tmi = sim.tmi_projection
        d_t2 = d_x * tmi[0] + d_y * tmi[1] + d_z * tmi[2]

        # Check results
        rtol, atol = 1e-7, 1e-6
        np.testing.assert_allclose(
            d_t, d_t2, rtol=rtol, atol=atol
        )  # double check internal projection
        np.testing.assert_allclose(d_x, d[:, 0], rtol=rtol, atol=atol)
        np.testing.assert_allclose(d_y, d[:, 1], rtol=rtol, atol=atol)
        np.testing.assert_allclose(d_z, d[:, 2], rtol=rtol, atol=atol)
        np.testing.assert_allclose(d_t, d @ tmi, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "engine, parallel_kwargs",
        [
            ("geoana", {"n_processes": None}),
            ("geoana", {"n_processes": 1}),
            ("choclo", {"numba_parallel": False}),
            ("choclo", {"numba_parallel": True}),
        ],
        ids=["geoana_serial", "geoana_parallel", "choclo_serial", "choclo_parallel"],
    )
    @pytest.mark.parametrize("store_sensitivities", ("ram", "disk", "forward_only"))
    def test_magnetic_gradiometry_w_susceptibility(
        self,
        engine,
        parallel_kwargs,
        store_sensitivities,
        tmp_path,
        mag_mesh,
        two_blocks,
        receiver_locations,
        inducing_field,
    ):
        """
        Test magnetic gradiometry components (with susceptibility as model)
        """
        inducing_field_params, b0 = inducing_field
        chi1 = 0.01
        chi2 = 0.02
        model, active_cells = create_block_model(mag_mesh, two_blocks, (chi1, chi2))
        model_reduced = model[active_cells]
        # Create reduced identity map for Linear Problem
        identity_map = maps.IdentityMap(nP=int(sum(active_cells)))

        survey = create_mag_survey(
            components=["bxx", "bxy", "bxz", "byy", "byz", "bzz"],
            receiver_locations=receiver_locations,
            inducing_field_params=inducing_field_params,
        )
        sim = mag.Simulation3DIntegral(
            mag_mesh,
            survey=survey,
            chiMap=identity_map,
            ind_active=active_cells,
            sensitivity_path=str(tmp_path / f"{engine}"),
            store_sensitivities=store_sensitivities,
            engine=engine,
            **parallel_kwargs,
        )
        if engine == "choclo":
            # gradient simulation not implemented for choclo yet
            with pytest.raises(NotImplementedError):
                data = sim.dpred(model_reduced)
        else:
            data = sim.dpred(model_reduced)
            d_xx = data[0::6]
            d_xy = data[1::6]
            d_xz = data[2::6]
            d_yy = data[3::6]
            d_yz = data[4::6]
            d_zz = data[5::6]

            # Compute analytical response from magnetic prism
            block1, block2 = two_blocks
            prism_1 = MagneticPrism(block1[:, 0], block1[:, 1], chi1 * b0 / mu_0)
            prism_2 = MagneticPrism(block2[:, 0], block2[:, 1], -chi1 * b0 / mu_0)
            prism_3 = MagneticPrism(block2[:, 0], block2[:, 1], chi2 * b0 / mu_0)

            d = (
                prism_1.magnetic_field_gradient(receiver_locations)
                + prism_2.magnetic_field_gradient(receiver_locations)
                + prism_3.magnetic_field_gradient(receiver_locations)
            ) * mu_0

            # Check results
            rtol, atol = 5e-7, 1e-6
            np.testing.assert_allclose(d_xx, d[..., 0, 0], rtol=rtol, atol=atol)
            np.testing.assert_allclose(d_xy, d[..., 0, 1], rtol=rtol, atol=atol)
            np.testing.assert_allclose(d_xz, d[..., 0, 2], rtol=rtol, atol=atol)
            np.testing.assert_allclose(d_yy, d[..., 1, 1], rtol=rtol, atol=atol)
            np.testing.assert_allclose(d_yz, d[..., 1, 2], rtol=rtol, atol=atol)
            np.testing.assert_allclose(d_zz, d[..., 2, 2], rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "engine, parallel_kwargs",
        [
            ("geoana", {"n_processes": None}),
            ("geoana", {"n_processes": 1}),
            ("choclo", {"numba_parallel": False}),
            ("choclo", {"numba_parallel": True}),
        ],
        ids=["geoana_serial", "geoana_parallel", "choclo_serial", "choclo_parallel"],
    )
    @pytest.mark.parametrize("store_sensitivities", ("ram", "disk", "forward_only"))
    def test_magnetic_vector_and_tmi_w_magnetization(
        self,
        engine,
        parallel_kwargs,
        store_sensitivities,
        tmp_path,
        mag_mesh,
        two_blocks,
        receiver_locations,
        inducing_field,
    ):
        """
        Test magnetic vector and TMI (using magnetization vectors as model)
        """
        inducing_field_params, b0 = inducing_field
        M1 = (utils.mat_utils.dip_azimuth2cartesian(45, -40) * 0.05).squeeze()
        M2 = (utils.mat_utils.dip_azimuth2cartesian(120, 32) * 0.1).squeeze()

        model, active_cells = create_block_model(mag_mesh, two_blocks, (M1, M2))
        model_reduced = model[active_cells].reshape(-1, order="F")
        # Create reduced identity map for Linear Problem
        identity_map = maps.IdentityMap(nP=int(sum(active_cells)) * 3)

        survey = create_mag_survey(
            components=["bx", "by", "bz", "tmi"],
            receiver_locations=receiver_locations,
            inducing_field_params=inducing_field_params,
        )

        sim = mag.Simulation3DIntegral(
            mag_mesh,
            survey=survey,
            chiMap=identity_map,
            ind_active=active_cells,
            sensitivity_path=str(tmp_path / f"{engine}"),
            store_sensitivities=store_sensitivities,
            model_type="vector",
            engine=engine,
            **parallel_kwargs,
        )

        data = sim.dpred(model_reduced).reshape(-1, 4)

        # Compute analytical response from magnetic prism
        block1, block2 = two_blocks
        prism_1 = MagneticPrism(
            block1[:, 0], block1[:, 1], M1 * np.linalg.norm(b0) / mu_0
        )
        prism_2 = MagneticPrism(
            block2[:, 0], block2[:, 1], -M1 * np.linalg.norm(b0) / mu_0
        )
        prism_3 = MagneticPrism(
            block2[:, 0], block2[:, 1], M2 * np.linalg.norm(b0) / mu_0
        )

        d = (
            prism_1.magnetic_flux_density(receiver_locations)
            + prism_2.magnetic_flux_density(receiver_locations)
            + prism_3.magnetic_flux_density(receiver_locations)
        )
        tmi = sim.tmi_projection

        # Check results
        rtol, atol = 5e-7, 1e-6
        np.testing.assert_allclose(data[:, 0], d[:, 0], rtol=rtol, atol=atol)
        np.testing.assert_allclose(data[:, 1], d[:, 1], rtol=rtol, atol=atol)
        np.testing.assert_allclose(data[:, 2], d[:, 2], rtol=rtol, atol=atol)
        np.testing.assert_allclose(data[:, 3], d @ tmi, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "engine, parallel_kwargs",
        [
            ("geoana", {"n_processes": None}),
            ("geoana", {"n_processes": 1}),
            ("choclo", {"numba_parallel": False}),
            ("choclo", {"numba_parallel": True}),
        ],
        ids=["geoana_serial", "geoana_parallel", "choclo_serial", "choclo_parallel"],
    )
    @pytest.mark.parametrize("store_sensitivities", ("ram", "disk", "forward_only"))
    def test_magnetic_field_amplitude_w_magnetization(
        self,
        engine,
        parallel_kwargs,
        store_sensitivities,
        tmp_path,
        mag_mesh,
        two_blocks,
        receiver_locations,
        inducing_field,
    ):
        """
        Test magnetic field amplitude (using magnetization vectors as model)
        """
        inducing_field_params, b0 = inducing_field
        M1 = (utils.mat_utils.dip_azimuth2cartesian(45, -40) * 0.05).squeeze()
        M2 = (utils.mat_utils.dip_azimuth2cartesian(120, 32) * 0.1).squeeze()

        model, active_cells = create_block_model(mag_mesh, two_blocks, (M1, M2))
        model_reduced = model[active_cells].reshape(-1, order="F")
        # Create reduced identity map for Linear Problem
        identity_map = maps.IdentityMap(nP=int(sum(active_cells)) * 3)

        survey = create_mag_survey(
            components=["bx", "by", "bz"],
            receiver_locations=receiver_locations,
            inducing_field_params=inducing_field_params,
        )

        sim = mag.Simulation3DIntegral(
            mag_mesh,
            survey=survey,
            chiMap=identity_map,
            ind_active=active_cells,
            sensitivity_path=str(tmp_path / f"{engine}"),
            store_sensitivities=store_sensitivities,
            model_type="vector",
            is_amplitude_data=True,
            engine=engine,
            **parallel_kwargs,
        )

        data = sim.dpred(model_reduced)

        # Compute analytical response from magnetic prism
        block1, block2 = two_blocks
        prism_1 = MagneticPrism(
            block1[:, 0], block1[:, 1], M1 * np.linalg.norm(b0) / mu_0
        )
        prism_2 = MagneticPrism(
            block2[:, 0], block2[:, 1], -M1 * np.linalg.norm(b0) / mu_0
        )
        prism_3 = MagneticPrism(
            block2[:, 0], block2[:, 1], M2 * np.linalg.norm(b0) / mu_0
        )

        d = (
            prism_1.magnetic_flux_density(receiver_locations)
            + prism_2.magnetic_flux_density(receiver_locations)
            + prism_3.magnetic_flux_density(receiver_locations)
        )
        d_amp = np.linalg.norm(d, axis=1)

        # Check results
        rtol, atol = 5e-7, 1e-6
        np.testing.assert_allclose(data, d_amp, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("engine", ("choclo", "geoana"))
    @pytest.mark.parametrize("store_sensitivities", ("ram", "disk", "forward_only"))
    def test_sensitivity_dtype(
        self,
        engine,
        store_sensitivities,
        mag_mesh,
        receiver_locations,
        tmp_path,
    ):
        """Test sensitivity_dtype."""
        # Create survey
        receivers = mag.Point(receiver_locations, components="tmi")
        sources = mag.UniformBackgroundField(
            [receivers], amplitude=50_000, inclination=45, declination=10
        )
        survey = mag.Survey(sources)
        # Create reduced identity map for Linear Problem
        active_cells = np.ones(mag_mesh.n_cells, dtype=bool)
        idenMap = maps.IdentityMap(nP=mag_mesh.n_cells)
        # Create simulation
        sensitivity_path = tmp_path
        if engine == "choclo":
            sensitivity_path /= "dummy"
        simulation = mag.Simulation3DIntegral(
            mag_mesh,
            survey=survey,
            chiMap=idenMap,
            ind_active=active_cells,
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
    def test_invalid_sensitivity_dtype_assignment(self, mag_mesh, invalid_dtype):
        """
        Test invalid sensitivity_dtype assignment
        """
        simulation = mag.Simulation3DIntegral(mag_mesh)
        # Check if error is raised
        msg = "sensitivity_dtype must be either np.float32 or np.float64."
        with pytest.raises(TypeError, match=msg):
            simulation.sensitivity_dtype = invalid_dtype

    def test_invalid_engine(self, mag_mesh):
        """Test if error is raised after invalid engine."""
        engine = "invalid engine"
        msg = rf"'engine' must be in \('geoana', 'choclo'\). Got '{engine}'"
        with pytest.raises(ValueError, match=msg):
            mag.Simulation3DIntegral(mag_mesh, engine=engine)

    def test_choclo_and_n_proceesses(self, mag_mesh):
        """Check if warning is raised after passing n_processes with choclo engine."""
        msg = "The 'n_processes' will be ignored when selecting 'choclo'"
        with pytest.warns(UserWarning, match=msg):
            simulation = mag.Simulation3DIntegral(
                mag_mesh, engine="choclo", n_processes=2
            )
        # Check if n_processes was overwritten and set to None
        assert simulation.n_processes is None

    def test_choclo_and_sensitivity_path_as_dir(self, mag_mesh, tmp_path):
        """
        Check if error is raised when sensitivity_path is a dir with choclo engine.
        """
        # Create a sensitivity_path directory
        sensitivity_path = tmp_path / "sensitivity_dummy"
        sensitivity_path.mkdir()
        # Check if error is raised
        msg = f"The passed sensitivity_path '{str(sensitivity_path)}' is a directory"
        with pytest.raises(ValueError, match=msg):
            mag.Simulation3DIntegral(
                mag_mesh,
                store_sensitivities="disk",
                sensitivity_path=str(sensitivity_path),
                engine="choclo",
            )

    def test_sensitivities_on_disk(self, mag_mesh, receiver_locations, tmp_path):
        """
        Test if sensitivity matrix is correctly being stored in disk when asked
        """
        # Build survey
        survey = create_mag_survey(
            components=["tmi"],
            receiver_locations=receiver_locations,
            inducing_field_params=(50000.0, 20.0, 45.0),
        )
        # Build simulation
        sensitivities_path = tmp_path / "sensitivities"
        simulation = mag.Simulation3DIntegral(
            mesh=mag_mesh,
            survey=survey,
            store_sensitivities="disk",
            sensitivity_path=str(sensitivities_path),
            engine="choclo",
        )
        simulation.G
        # Check if sensitivity matrix was stored in disk and is a memmap
        assert sensitivities_path.is_file()
        assert type(simulation.G) is np.memmap

    def test_sensitivities_on_ram(self, mag_mesh, receiver_locations, tmp_path):
        """
        Test if sensitivity matrix is correctly being allocated in memory when asked
        """
        # Build survey
        survey = create_mag_survey(
            components=["tmi"],
            receiver_locations=receiver_locations,
            inducing_field_params=(50000.0, 20.0, 45.0),
        )
        # Build simulation
        simulation = mag.Simulation3DIntegral(
            mesh=mag_mesh,
            survey=survey,
            store_sensitivities="ram",
            engine="choclo",
        )
        simulation.G
        # Check if sensitivity matrix is a Numpy array (stored in memory)
        assert type(simulation.G) is np.ndarray

    def test_choclo_missing(self, mag_mesh, monkeypatch):
        """
        Check if error is raised when choclo is missing and chosen as engine.
        """
        # Monkeypatch choclo in simpeg.potential_fields.base
        monkeypatch.setattr(simpeg.potential_fields.base, "choclo", None)
        # Check if error is raised
        msg = "The choclo package couldn't be found."
        with pytest.raises(ImportError, match=msg):
            mag.Simulation3DIntegral(mag_mesh, engine="choclo")


def test_ana_mag_tmi_grad_forward():
    """
    Test TMI gradiometry using susceptibilities as model
    """
    nx = 61
    ny = 61

    h0_amplitude, h0_inclination, h0_declination = (50000.0, 60.0, 250.0)
    b0 = mag.analytics.IDTtoxyz(-h0_inclination, h0_declination, h0_amplitude)
    chi1 = 0.01
    chi2 = 0.02

    # Define a mesh
    cs = 0.2
    hxind = [(cs, 41)]
    hyind = [(cs, 41)]
    hzind = [(cs, 41)]
    mesh = discretize.TensorMesh([hxind, hyind, hzind], "CCC")

    # create a model of two blocks, 1 inside the other
    block1 = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
    block2 = np.array([[-0.7, 0.7], [-0.7, 0.7], [-0.7, 0.7]])

    def get_block_inds(grid, block):
        return np.where(
            (grid[:, 0] > block[0, 0])
            & (grid[:, 0] < block[0, 1])
            & (grid[:, 1] > block[1, 0])
            & (grid[:, 1] < block[1, 1])
            & (grid[:, 2] > block[2, 0])
            & (grid[:, 2] < block[2, 1])
        )

    block1_inds = get_block_inds(mesh.cell_centers, block1)
    block2_inds = get_block_inds(mesh.cell_centers, block2)

    model = np.zeros(mesh.n_cells)
    model[block1_inds] = chi1
    model[block2_inds] = chi2

    active_cells = model != 0.0
    model_reduced = model[active_cells]

    # Create reduced identity map for Linear Problem
    idenMap = maps.IdentityMap(nP=int(sum(active_cells)))

    # Create plane of observations
    xr = np.linspace(-20, 20, nx)
    dxr = xr[1] - xr[0]
    yr = np.linspace(-20, 20, ny)
    dyr = yr[1] - yr[0]
    X, Y = np.meshgrid(xr, yr)
    Z = np.ones_like(X) * 3.0
    locXyz = np.c_[X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]
    components = ["tmi", "tmi_x", "tmi_y", "tmi_z"]

    rxLoc = mag.Point(locXyz, components=components)
    srcField = mag.UniformBackgroundField(
        receiver_list=[rxLoc],
        amplitude=h0_amplitude,
        inclination=h0_inclination,
        declination=h0_declination,
    )
    survey = mag.Survey(srcField)

    # Create reduced identity map for Linear Problem
    idenMap = maps.IdentityMap(nP=int(sum(active_cells)))

    sim = mag.Simulation3DIntegral(
        mesh,
        survey=survey,
        chiMap=idenMap,
        ind_active=active_cells,
        store_sensitivities="forward_only",
        n_processes=None,
    )

    data = sim.dpred(model_reduced)
    tmi = data[0::4]
    d_x = data[1::4]
    d_y = data[2::4]
    d_z = data[3::4]

    # Compute analytical response from magnetic prism
    prism_1 = MagneticPrism(block1[:, 0], block1[:, 1], chi1 * b0 / mu_0)
    prism_2 = MagneticPrism(block2[:, 0], block2[:, 1], -chi1 * b0 / mu_0)
    prism_3 = MagneticPrism(block2[:, 0], block2[:, 1], chi2 * b0 / mu_0)

    d = (
        prism_1.magnetic_field_gradient(locXyz)
        + prism_2.magnetic_field_gradient(locXyz)
        + prism_3.magnetic_field_gradient(locXyz)
    ) * mu_0
    tmi_x = (
        d[:, 0, 0] * b0[0] + d[:, 0, 1] * b0[1] + d[:, 0, 2] * b0[2]
    ) / h0_amplitude
    tmi_y = (
        d[:, 1, 0] * b0[0] + d[:, 1, 1] * b0[1] + d[:, 1, 2] * b0[2]
    ) / h0_amplitude
    tmi_z = (
        d[:, 2, 0] * b0[0] + d[:, 2, 1] * b0[1] + d[:, 2, 2] * b0[2]
    ) / h0_amplitude
    np.testing.assert_allclose(d_x, tmi_x, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(d_y, tmi_y, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(d_z, tmi_z, rtol=1e-10, atol=1e-12)

    # finite difference test y-grad
    np.testing.assert_allclose(
        np.diff(tmi.reshape(nx, ny, order="F")[:, ::2], axis=1) / (2 * dyr),
        tmi_y.reshape(nx, ny, order="F")[:, 1::2],
        atol=1.0,
        rtol=1e-1,
    )
    # finite difference test x-grad
    np.testing.assert_allclose(
        np.diff(tmi.reshape(nx, ny, order="F")[::2, :], axis=0) / (2 * dxr),
        tmi_x.reshape(nx, ny, order="F")[1::2, :],
        atol=1.0,
        rtol=1e-1,
    )


class TestInvalidMeshChoclo:
    @pytest.fixture(params=("tensormesh", "treemesh"))
    def mesh(self, request):
        """Sample 2D mesh."""
        hx, hy = [(0.1, 8)], [(0.1, 8)]
        h = (hx, hy)
        if request.param == "tensormesh":
            mesh = discretize.TensorMesh(h, "CC")
        else:
            mesh = discretize.TreeMesh(h, origin="CC")
            mesh.finalize()
        return mesh

    def test_invalid_mesh_with_choclo(self, mesh):
        """
        Test if simulation raises error when passing an invalid mesh and using choclo
        """
        # Build survey
        receivers_locations = np.array([[0, 0, 0]])
        receivers = mag.Point(receivers_locations)
        sources = mag.UniformBackgroundField(
            receiver_list=[receivers],
            amplitude=50_000,
            inclination=45.0,
            declination=12.0,
        )
        survey = mag.Survey(sources)
        # Check if error is raised
        msg = (
            "Invalid mesh with 2 dimensions. "
            "Only 3D meshes are supported when using 'choclo' as engine."
        )
        with pytest.raises(ValueError, match=msg):
            mag.Simulation3DIntegral(mesh, survey, engine="choclo")


def test_removed_modeltype():
    """Test if accesing removed modelType property raises error."""
    h = [[(2, 2)], [(2, 2)], [(2, 2)]]
    mesh = discretize.TensorMesh(h)
    receiver_location = np.array([[0, 0, 100]])
    receiver = mag.Point(receiver_location, components="tmi")
    background_field = mag.UniformBackgroundField(
        receiver_list=[receiver], amplitude=50_000, inclination=90, declination=0
    )
    survey = mag.Survey(background_field)
    mapping = maps.IdentityMap(mesh, nP=mesh.n_cells)
    sim = mag.Simulation3DIntegral(mesh, survey=survey, chiMap=mapping)
    message = "modelType has been removed, please use model_type."
    with pytest.raises(NotImplementedError, match=message):
        sim.modelType
