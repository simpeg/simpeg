import numpy as np
import unittest
from scipy.constants import mu_0
from discretize.tests import check_derivative

from SimPEG.electromagnetics import natural_source as nsem
from SimPEG import maps
from discretize import TensorMesh, TreeMesh, CylindricalMesh
from pymatsolver import Pardiso


def check_deriv(sim, test_mod, **kwargs):
    x0 = test_mod.copy()

    def func(x):
        d = sim.dpred(x)

        def J_func(v):
            return sim.Jvec(x0, v)

        return d, J_func

    passed = check_derivative(func, x0, plotIt=False, **kwargs)
    return passed


def check_adjoint(sim, test_mod):
    u = np.random.rand(len(test_mod))
    v = np.random.rand(sim.survey.nD)

    f = sim.fields(test_mod)
    Ju = sim.Jvec(test_mod, u, f=f)
    JTv = sim.Jtvec(test_mod, v, f=f)

    np.testing.assert_allclose(v.dot(Ju), u.dot(JTv))


def create_simulation_1d(sim_type, deriv_type):

    cs = 100
    ncz = 200
    npad = 20
    pf = 1.2

    hz = [(cs, npad, -pf), (cs, ncz), (cs, npad)]

    mesh = TensorMesh([hz,])
    mesh.x0 = np.r_[-mesh.h[0][:-npad].sum()]

    frequencies = np.logspace(-2, 1, 30)

    rx_list = [
        nsem.receivers.PointNaturalSource([[0]], orientation="xy", component="real"),
        nsem.receivers.PointNaturalSource([[0]], orientation="xy", component="imag"),
        nsem.receivers.PointNaturalSource(
            [[0]], orientation="xy", component="apparent_resistivity"
        ),
        nsem.receivers.PointNaturalSource([[0]], orientation="xy", component="phase"),
        nsem.receivers.PointNaturalSource([[0]], orientation="yx", component="real"),
        nsem.receivers.PointNaturalSource([[0]], orientation="yx", component="imag"),
        nsem.receivers.PointNaturalSource(
            [[0]], orientation="yx", component="apparent_resistivity"
        ),
        nsem.receivers.PointNaturalSource([[0]], orientation="yx", component="phase"),
    ]
    src_list = [nsem.sources.Planewave1D(rx_list, frequency=f) for f in frequencies]
    survey = nsem.Survey1D(src_list)

    sigma_back = 1e-1
    sigma_right = 1e-3
    sigma_air = 1e-8

    sigma_1d = np.ones(mesh.n_cells) * sigma_back
    sigma_1d[-npad:] = sigma_air
    sigma_1d[50:-npad] = sigma_right

    if deriv_type == "sigma":
        sim_kwargs = {"sigmaMap": maps.ExpMap()}
        test_mod = np.log(sigma_1d)
    else:
        sim_kwargs = {"muMap": maps.ExpMap(), "sigma": sigma_1d}
        test_mod = np.log(mu_0) * np.ones(mesh.n_cells)
    if sim_type.lower() == "e":
        sim = nsem.simulation.Simulation1DElectricField(
            mesh, survey=survey, **sim_kwargs, solver=Pardiso,
        )
    else:
        sim = nsem.simulation.Simulation1DMagneticField(
            mesh, survey=survey, **sim_kwargs, solver=Pardiso,
        )
    return sim, test_mod


def create_simulation_2d(sim_type, deriv_type, mesh_type, fixed_boundary=False):
    cs = 200
    ncx = 108
    ncz = 108
    npad = 10
    pf = 1.2

    hx = [(cs, npad, -pf), (cs, ncx), (cs, npad, pf)]
    hz = [(cs, npad, -pf), (cs, ncz), (cs, npad)]

    if mesh_type == "TreeMesh":
        mesh = TensorMesh([hx, hz])
        mesh.origin = np.r_[-mesh.hx.sum() / 2, -mesh.hy[:-npad].sum()]

        mesh = TreeMesh([hx, hz], mesh.origin)
        mesh.refine_ball(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [10000, 15000, 20000], [-1, -2, -3]
        )
    else:
        mesh = TensorMesh([hx, hz])
        mesh.origin = np.r_[-mesh.hx.sum() / 2, -mesh.hy[:-npad].sum()]

    sigma_back = 1e-1
    sigma_right = 1e-3
    sigma_air = 1e-8

    cells = mesh.cell_centers
    sigma = np.ones(mesh.n_cells) * sigma_back
    sigma[cells[:, 0] >= 0] = sigma_right
    sigma[cells[:, -1] >= 0] = sigma_air

    if deriv_type == "sigma":
        sim_kwargs = {"sigmaMap": maps.ExpMap()}
        test_mod = np.log(sigma)
    else:
        sim_kwargs = {"muMap": maps.ExpMap(), "sigma": sigma}
        test_mod = np.log(mu_0) * np.ones(mesh.n_cells)

    frequencies = np.logspace(-1, 1, 2)

    rx_locs = np.c_[np.linspace(-8000, 8000, 51), np.zeros(51)]

    if sim_type.lower() == "e":
        if fixed_boundary:
            # get field from 1D simulation
            survey_1d = nsem.Survey1D(
                [nsem.sources.Planewave1D([], frequency=f) for f in frequencies]
            )
            mesh_1d = TensorMesh([mesh.h[1]], [mesh.origin[1]])
            sim_1d = nsem.simulation.Simulation1DElectricField(
                mesh_1d, survey=survey_1d, sigmaMap=maps.IdentityMap()
            )

            b_left, b_right, _, __ = mesh.cell_boundary_indices
            f_left = sim_1d.fields(sigma[b_left])
            f_right = sim_1d.fields(sigma[b_right])

            b_e = mesh.boundary_edges
            top = np.where(b_e[:, 1] == mesh.nodes_y[-1])
            bot = np.where(b_e[:, 1] == mesh.nodes_y[0])
            left = np.where(b_e[:, 0] == mesh.nodes_x[0])
            right = np.where(b_e[:, 0] == mesh.nodes_x[-1])
            h_bc = {}
            for src in survey_1d.source_list:
                h_bc_freq = np.zeros(mesh.boundary_edges.shape[0], dtype=np.complex)
                h_bc_freq[top] = 1.0
                h_bc_freq[right] = f_right[src, "h"][:, 0]
                h_bc_freq[left] = f_left[src, "h"][:, 0]
                h_bc[src.frequency] = h_bc_freq
            sim_kwargs["h_bc"] = h_bc

        rx_list = [
            nsem.receivers.PointNaturalSource(
                rx_locs, orientation="xy", component="real"
            ),
            nsem.receivers.PointNaturalSource(
                rx_locs, orientation="xy", component="imag"
            ),
            nsem.receivers.PointNaturalSource(
                rx_locs, orientation="xy", component="apparent_resistivity"
            ),
            nsem.receivers.PointNaturalSource(
                rx_locs, orientation="xy", component="phase"
            ),
        ]
        src_list = [nsem.sources.Planewave1D(rx_list, frequency=f) for f in frequencies]
        survey = nsem.Survey(src_list)

        sim = nsem.simulation.Simulation2DElectricField(
            mesh, survey=survey, **sim_kwargs, solver=Pardiso,
        )
    else:
        if fixed_boundary:
            # get field from 1D simulation
            survey_1d = nsem.Survey1D(
                [nsem.sources.Planewave1D([], frequency=f) for f in frequencies]
            )
            mesh_1d = TensorMesh([mesh.h[1]], [mesh.origin[1]])
            sim_1d = nsem.simulation.Simulation1DMagneticField(
                mesh_1d, survey=survey_1d, sigmaMap=maps.IdentityMap()
            )

            b_left, b_right, _, __ = mesh.cell_boundary_indices
            f_left = sim_1d.fields(sigma[b_left])
            f_right = sim_1d.fields(sigma[b_right])

            b_e = mesh.boundary_edges
            top = np.where(b_e[:, 1] == mesh.nodes_y[-1])
            left = np.where(b_e[:, 0] == mesh.nodes_x[0])
            right = np.where(b_e[:, 0] == mesh.nodes_x[-1])
            e_bc = {}
            for src in survey_1d.source_list:
                e_bc_freq = np.zeros(mesh.boundary_edges.shape[0], dtype=np.complex)
                e_bc_freq[top] = 1.0
                e_bc_freq[right] = f_right[src, "e"][:, 0]
                e_bc_freq[left] = f_left[src, "e"][:, 0]
                e_bc[src.frequency] = e_bc_freq
            sim_kwargs["e_bc"] = e_bc

        rx_list = [
            nsem.receivers.PointNaturalSource(
                rx_locs, orientation="yx", component="real"
            ),
            nsem.receivers.PointNaturalSource(
                rx_locs, orientation="yx", component="imag"
            ),
            nsem.receivers.PointNaturalSource(
                rx_locs, orientation="yx", component="apparent_resistivity"
            ),
            nsem.receivers.PointNaturalSource(
                rx_locs, orientation="yx", component="phase"
            ),
        ]
        src_list = [nsem.sources.Planewave1D(rx_list, frequency=f) for f in frequencies]
        survey = nsem.Survey(src_list)

        sim = nsem.simulation.Simulation2DMagneticField(
            mesh, survey=survey, **sim_kwargs, solver=Pardiso,
        )
    return sim, test_mod


class Sim_1D(unittest.TestCase):
    def test_errors(self):
        mesh = TensorMesh([5, 5])
        survey = nsem.Survey1D([nsem.sources.Planewave1D([], frequency=10)])

        with self.assertRaises(ValueError):
            nsem.simulation.Simulation1DElectricField(mesh, survey=survey)
        with self.assertRaises(ValueError):
            nsem.simulation.Simulation1DMagneticField(mesh, survey=survey)

    def test_e_sigma_deriv(self):
        sim, test_mod = create_simulation_1d("e", "sigma")
        assert check_deriv(sim, test_mod, num=3)

    def test_h_sigma_deriv(self):
        sim, test_mod = create_simulation_1d("h", "sigma")
        assert check_deriv(sim, test_mod, num=3)

    def test_e_mu_deriv(self):
        sim, test_mod = create_simulation_1d("e", "mu")
        assert check_deriv(sim, test_mod, num=3)

    def test_h_mu_deriv(self):
        sim, test_mod = create_simulation_1d("h", "mu")
        assert check_deriv(sim, test_mod, num=3)

    def test_e_sigma_adjoint(self):
        sim, test_mod = create_simulation_1d("e", "sigma")
        check_adjoint(sim, test_mod)

    def test_h_sigma_adjoint(self):
        sim, test_mod = create_simulation_1d("h", "sigma")
        check_adjoint(sim, test_mod)

    def test_e_mu_adjoint(self):
        sim, test_mod = create_simulation_1d("e", "mu")
        check_adjoint(sim, test_mod)

    def test_h_mu_adjoint(self):
        sim, test_mod = create_simulation_1d("h", "mu")
        check_adjoint(sim, test_mod)


class Sim_2D(unittest.TestCase):
    def test_errors(self):
        rx_locs = np.c_[np.linspace(-8000, 8000, 3), np.zeros(3)]
        mesh_1d = TensorMesh([5])
        mesh_2d = TensorMesh([5, 5])
        r_xy = nsem.receivers.PointNaturalSource(
            rx_locs, orientation="xy", component="apparent_resistivity"
        )
        r_yx = nsem.receivers.PointNaturalSource(
            rx_locs, orientation="yx", component="apparent_resistivity"
        )
        survey_xy = nsem.Survey1D([nsem.sources.Planewave1D([r_xy], frequency=10)])
        survey_yx = nsem.Survey1D([nsem.sources.Planewave1D([r_yx], frequency=10)])

        # Check mesh dim error
        with self.assertRaises(ValueError):
            nsem.simulation.Simulation2DElectricField(mesh_1d, survey=survey_xy)
        with self.assertRaises(ValueError):
            nsem.simulation.Simulation2DMagneticField(mesh_1d, survey=survey_yx)

        # Check receiver orientation error
        with self.assertRaises(TypeError):
            nsem.simulation.Simulation2DElectricField(mesh_2d, survey=survey_yx)
        with self.assertRaises(TypeError):
            nsem.simulation.Simulation2DMagneticField(mesh_2d, survey=survey_xy)

        # Check mesh type error without a given h_bc
        bad_mesh = CylindricalMesh([10, 10])
        with self.assertRaises(NotImplementedError):
            nsem.simulation.Simulation2DElectricField(bad_mesh, survey=survey_xy)
        with self.assertRaises(NotImplementedError):
            nsem.simulation.Simulation2DMagneticField(bad_mesh, survey=survey_yx)

        # Check fixed boundary condition Type Error
        with self.assertRaises(TypeError):
            nsem.simulation.Simulation2DElectricField(
                mesh_2d, survey=survey_xy, h_bc=100
            )
        with self.assertRaises(TypeError):
            nsem.simulation.Simulation2DMagneticField(
                mesh_2d, survey=survey_yx, e_bc=100
            )
        with self.assertRaises(TypeError):
            nsem.simulation.Simulation2DElectricField(
                mesh_2d, survey=survey_xy, h_bc=np.random.rand(20)
            )
        with self.assertRaises(TypeError):
            nsem.simulation.Simulation2DMagneticField(
                mesh_2d, survey=survey_yx, e_bc=np.random.rand(20)
            )

        # Check fixed boundary condition Key Error
        with self.assertRaises(KeyError):
            nsem.simulation.Simulation2DElectricField(
                mesh_2d, survey=survey_xy, h_bc={"a": "hello"}
            )
        with self.assertRaises(KeyError):
            nsem.simulation.Simulation2DMagneticField(
                mesh_2d, survey=survey_yx, e_bc={"a": "hello"}
            )

        # Check fixed boundary condition length error
        bc = {}
        for freq in survey_xy.frequencies:
            bc[freq] = np.random.rand(mesh_2d.boundary_edges.shape[0] + 3)
        with self.assertRaises(ValueError):
            nsem.simulation.Simulation2DElectricField(
                mesh_2d, survey=survey_xy, h_bc=bc
            )
        with self.assertRaises(ValueError):
            nsem.simulation.Simulation2DMagneticField(
                mesh_2d, survey=survey_yx, e_bc=bc
            )

    def test_e_sigma_deriv(self):
        sim, test_mod = create_simulation_2d("e", "sigma", "TensorMesh")
        assert check_deriv(sim, test_mod, num=3)

    def test_h_sigma_deriv(self):
        sim, test_mod = create_simulation_2d("h", "sigma", "TensorMesh")
        assert check_deriv(sim, test_mod, num=3)

    def test_e_mu_deriv(self):
        sim, test_mod = create_simulation_2d("e", "mu", "TensorMesh")
        assert check_deriv(sim, test_mod, num=3)

    def test_h_mu_deriv(self):
        sim, test_mod = create_simulation_2d("h", "mu", "TensorMesh")
        assert check_deriv(sim, test_mod, num=3)

    def test_e_sigma_adjoint(self):
        sim, test_mod = create_simulation_2d("e", "sigma", "TensorMesh")
        check_adjoint(sim, test_mod)

    def test_h_sigma_adjoint(self):
        sim, test_mod = create_simulation_2d("h", "sigma", "TensorMesh")
        check_adjoint(sim, test_mod)

    def test_e_mu_adjoint(self):
        sim, test_mod = create_simulation_2d("e", "mu", "TensorMesh")
        check_adjoint(sim, test_mod)

    def test_h_mu_adjoint(self):
        sim, test_mod = create_simulation_2d("h", "mu", "TensorMesh")
        check_adjoint(sim, test_mod)

    def test_e_sigma_adjoint_tree(self):
        sim, test_mod = create_simulation_2d("e", "sigma", "TreeMesh")
        check_adjoint(sim, test_mod)

    def test_h_sigma_adjoint_tree(self):
        sim, test_mod = create_simulation_2d("h", "sigma", "TreeMesh")
        check_adjoint(sim, test_mod)

    def test_e_sigma_deriv_fixed(self):
        sim, test_mod = create_simulation_2d(
            "e", "sigma", "TensorMesh", fixed_boundary=True
        )
        assert check_deriv(sim, test_mod, num=3)

    def test_h_sigma_deriv_fixed(self):
        sim, test_mod = create_simulation_2d(
            "h", "sigma", "TensorMesh", fixed_boundary=True
        )
        assert check_deriv(sim, test_mod, num=3)

    def test_e_sigma_adjoint_fixed(self):
        sim, test_mod = create_simulation_2d(
            "e", "sigma", "TensorMesh", fixed_boundary=True
        )
        check_adjoint(sim, test_mod)

    def test_h_sigma_adjoint_fixed(self):
        sim, test_mod = create_simulation_2d(
            "h", "sigma", "TensorMesh", fixed_boundary=True
        )
        check_adjoint(sim, test_mod)
