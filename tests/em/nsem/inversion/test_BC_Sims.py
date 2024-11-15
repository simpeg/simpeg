import numpy as np
import unittest
from scipy.constants import mu_0
from discretize.tests import check_derivative

from simpeg.electromagnetics import natural_source as nsem
from simpeg import maps
from discretize import TensorMesh, TreeMesh, CylindricalMesh


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
    rng = np.random.default_rng(seed=42)
    u = rng.uniform(size=len(test_mod))
    v = rng.uniform(size=sim.survey.nD)

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

    mesh = TensorMesh(
        [
            hz,
        ]
    )
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
    src_list = [nsem.sources.Planewave(rx_list, frequency=f) for f in frequencies]
    survey = nsem.Survey(src_list)

    conductivity_back = 1e-1
    conductivity_right = 1e-3
    conductivity_air = 1e-8

    conductivity_1d = np.ones(mesh.n_cells) * conductivity_back
    conductivity_1d[-npad:] = conductivity_air
    conductivity_1d[50:-npad] = conductivity_right

    if deriv_type == "conductivity":
        sim_kwargs = {"conductivity_map": maps.ExpMap()}
        test_mod = np.log(conductivity_1d)
    else:
        sim_kwargs = {
            "permeability_map": maps.ExpMap(),
            "conductivity": conductivity_1d,
        }
        test_mod = np.log(mu_0) * np.ones(mesh.n_cells)
    if sim_type.lower() == "e":
        sim = nsem.simulation.Simulation1DElectricField(
            mesh,
            survey=survey,
            **sim_kwargs,
        )
    else:
        sim = nsem.simulation.Simulation1DMagneticField(
            mesh,
            survey=survey,
            **sim_kwargs,
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
        mesh.origin = np.r_[-mesh.h[0].sum() / 2, -mesh.h[1][:-npad].sum()]

        mesh = TreeMesh([hx, hz], mesh.origin)
        mesh.refine_ball(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [10000, 15000, 20000], [-1, -2, -3]
        )
    else:
        mesh = TensorMesh([hx, hz])
        mesh.origin = np.r_[-mesh.h[0].sum() / 2, -mesh.h[1][:-npad].sum()]

    conductivity_back = 1e-1
    conductivity_right = 1e-3
    conductivity_air = 1e-8

    cells = mesh.cell_centers
    conductivity = np.ones(mesh.n_cells) * conductivity_back
    conductivity[cells[:, 0] >= 0] = conductivity_right
    conductivity[cells[:, -1] >= 0] = conductivity_air

    if deriv_type == "conductivity":
        sim_kwargs = {"conductivity_map": maps.ExpMap()}
        test_mod = np.log(conductivity)
    else:
        sim_kwargs = {"permeability_map": maps.ExpMap(), "conductivity": conductivity}
        test_mod = np.log(mu_0) * np.ones(mesh.n_cells)

    frequencies = np.logspace(-1, 1, 2)

    rx_locs = np.c_[np.linspace(-8000, 8000, 51), np.zeros(51)]

    if sim_type.lower() == "e":
        if fixed_boundary:
            # get field from 1D simulation
            survey_1d = nsem.Survey(
                [nsem.sources.Planewave([], frequency=f) for f in frequencies]
            )
            mesh_1d = TensorMesh([mesh.h[1]], [mesh.origin[1]])
            sim_1d = nsem.simulation.Simulation1DElectricField(
                mesh_1d, survey=survey_1d, conductivity_map=maps.IdentityMap()
            )

            b_left, b_right, _, __ = mesh.cell_boundary_indices
            f_left = sim_1d.fields(conductivity[b_left])
            f_right = sim_1d.fields(conductivity[b_right])

            b_e = mesh.boundary_edges
            top = np.where(b_e[:, 1] == mesh.nodes_y[-1])
            left = np.where(b_e[:, 0] == mesh.nodes_x[0])
            right = np.where(b_e[:, 0] == mesh.nodes_x[-1])
            h_bc = {}
            for src in survey_1d.source_list:
                h_bc_freq = np.zeros(mesh.boundary_edges.shape[0], dtype=complex)
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
        src_list = [nsem.sources.Planewave(rx_list, frequency=f) for f in frequencies]
        survey = nsem.Survey(src_list)

        sim = nsem.simulation.Simulation2DElectricField(
            mesh,
            survey=survey,
            **sim_kwargs,
        )
    else:
        if fixed_boundary:
            # get field from 1D simulation
            survey_1d = nsem.Survey(
                [nsem.sources.Planewave([], frequency=f) for f in frequencies]
            )
            mesh_1d = TensorMesh([mesh.h[1]], [mesh.origin[1]])
            sim_1d = nsem.simulation.Simulation1DMagneticField(
                mesh_1d, survey=survey_1d, conductivity_map=maps.IdentityMap()
            )

            b_left, b_right, _, __ = mesh.cell_boundary_indices
            f_left = sim_1d.fields(conductivity[b_left])
            f_right = sim_1d.fields(conductivity[b_right])

            b_e = mesh.boundary_edges
            top = np.where(b_e[:, 1] == mesh.nodes_y[-1])
            left = np.where(b_e[:, 0] == mesh.nodes_x[0])
            right = np.where(b_e[:, 0] == mesh.nodes_x[-1])
            e_bc = {}
            for src in survey_1d.source_list:
                e_bc_freq = np.zeros(mesh.boundary_edges.shape[0], dtype=complex)
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
        src_list = [nsem.sources.Planewave(rx_list, frequency=f) for f in frequencies]
        survey = nsem.Survey(src_list)

        sim = nsem.simulation.Simulation2DMagneticField(
            mesh,
            survey=survey,
            **sim_kwargs,
        )
    return sim, test_mod


class Sim_1D(unittest.TestCase):
    def test_errors(self):
        mesh = TensorMesh([5, 5])
        survey = nsem.Survey([nsem.sources.Planewave([], frequency=10)])

        with self.assertRaises(ValueError):
            nsem.simulation.Simulation1DElectricField(mesh, survey=survey)
        with self.assertRaises(ValueError):
            nsem.simulation.Simulation1DMagneticField(mesh, survey=survey)

    def test_e_conductivity_deriv(self):
        sim, test_mod = create_simulation_1d("e", "conductivity")
        assert check_deriv(sim, test_mod, num=3, random_seed=235)

    def test_h_conductivity_deriv(self):
        sim, test_mod = create_simulation_1d("h", "conductivity")
        assert check_deriv(sim, test_mod, num=3, random_seed=5212)

    def test_e_mu_deriv(self):
        sim, test_mod = create_simulation_1d("e", "permeability")
        assert check_deriv(sim, test_mod, num=3, random_seed=63246)

    def test_h_mu_deriv(self):
        sim, test_mod = create_simulation_1d("h", "permeability")
        assert check_deriv(sim, test_mod, num=3, random_seed=124)

    def test_e_conductivity_adjoint(self):
        sim, test_mod = create_simulation_1d("e", "conductivity")
        check_adjoint(sim, test_mod)

    def test_h_conductivity_adjoint(self):
        sim, test_mod = create_simulation_1d("h", "conductivity")
        check_adjoint(sim, test_mod)

    def test_e_mu_adjoint(self):
        sim, test_mod = create_simulation_1d("e", "permeability")
        check_adjoint(sim, test_mod)

    def test_h_mu_adjoint(self):
        sim, test_mod = create_simulation_1d("h", "permeability")
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
        survey_xy = nsem.Survey([nsem.sources.Planewave([r_xy], frequency=10)])
        survey_yx = nsem.Survey([nsem.sources.Planewave([r_yx], frequency=10)])

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

        random_array = np.random.default_rng(seed=42).uniform(size=20)
        with self.assertRaises(TypeError):
            nsem.simulation.Simulation2DElectricField(
                mesh_2d, survey=survey_xy, h_bc=random_array
            )
        with self.assertRaises(TypeError):
            nsem.simulation.Simulation2DMagneticField(
                mesh_2d, survey=survey_yx, e_bc=random_array
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
        rng = np.random.default_rng(seed=42)
        for freq in survey_xy.frequencies:
            bc[freq] = rng.uniform(size=mesh_2d.boundary_edges.shape[0] + 3)
        with self.assertRaises(ValueError):
            nsem.simulation.Simulation2DElectricField(
                mesh_2d, survey=survey_xy, h_bc=bc
            )
        with self.assertRaises(ValueError):
            nsem.simulation.Simulation2DMagneticField(
                mesh_2d, survey=survey_yx, e_bc=bc
            )

    def test_e_conductivity_deriv(self):
        sim, test_mod = create_simulation_2d("e", "conductivity", "TensorMesh")
        assert check_deriv(sim, test_mod, num=3, random_seed=125)

    def test_h_conductivity_deriv(self):
        sim, test_mod = create_simulation_2d("h", "conductivity", "TensorMesh")
        assert check_deriv(sim, test_mod, num=3, random_seed=7425)

    def test_e_mu_deriv(self):
        sim, test_mod = create_simulation_2d("e", "permeability", "TensorMesh")
        assert check_deriv(sim, test_mod, num=3, random_seed=236423)

    def test_h_mu_deriv(self):
        sim, test_mod = create_simulation_2d("h", "permeability", "TensorMesh")
        assert check_deriv(sim, test_mod, num=3, random_seed=34632)

    def test_e_conductivity_adjoint(self):
        sim, test_mod = create_simulation_2d("e", "conductivity", "TensorMesh")
        check_adjoint(sim, test_mod)

    def test_h_conductivity_adjoint(self):
        sim, test_mod = create_simulation_2d("h", "conductivity", "TensorMesh")
        check_adjoint(sim, test_mod)

    def test_e_mu_adjoint(self):
        sim, test_mod = create_simulation_2d("e", "permeability", "TensorMesh")
        check_adjoint(sim, test_mod)

    def test_h_mu_adjoint(self):
        sim, test_mod = create_simulation_2d("h", "permeability", "TensorMesh")
        check_adjoint(sim, test_mod)

    def test_e_conductivity_adjoint_tree(self):
        sim, test_mod = create_simulation_2d("e", "conductivity", "TreeMesh")
        check_adjoint(sim, test_mod)

    def test_h_conductivity_adjoint_tree(self):
        sim, test_mod = create_simulation_2d("h", "conductivity", "TreeMesh")
        check_adjoint(sim, test_mod)

    def test_e_conductivity_deriv_fixed(self):
        sim, test_mod = create_simulation_2d(
            "e", "conductivity", "TensorMesh", fixed_boundary=True
        )
        assert check_deriv(sim, test_mod, num=3, random_seed=2634)

    def test_h_conductivity_deriv_fixed(self):
        sim, test_mod = create_simulation_2d(
            "h", "conductivity", "TensorMesh", fixed_boundary=True
        )
        assert check_deriv(sim, test_mod, num=3, random_seed=3651326)

    def test_e_conductivity_adjoint_fixed(self):
        sim, test_mod = create_simulation_2d(
            "e", "conductivity", "TensorMesh", fixed_boundary=True
        )
        check_adjoint(sim, test_mod)

    def test_h_conductivity_adjoint_fixed(self):
        sim, test_mod = create_simulation_2d(
            "h", "conductivity", "TensorMesh", fixed_boundary=True
        )
        check_adjoint(sim, test_mod)
