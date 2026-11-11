"""
Unit tests for the helpers behind the robin boundary conditions of the 1D and 2D
NSEM simulations.
"""

import functools
import numpy as np
import pytest
import discretize
import scipy.sparse as sp
from scipy.constants import mu_0
from scipy.integrate import quad
from discretize.utils import Zero

from simpeg import maps, tests
from simpeg.electromagnetics import natural_source as nsem
from simpeg.electromagnetics.natural_source.simulation import (
    _log_integral,
    _upward_continuation_ops,
)
from simpeg.electromagnetics.utils import omega
from simpeg.utils import get_default_solver

FREQ = 1.0
SIM_1D = [nsem.Simulation1DElectricField, nsem.Simulation1DMagneticField]
SIM_2D = [nsem.Simulation2DElectricField, nsem.Simulation2DMagneticField]
# (simulation class, property, its value from sigma)
PROPERTIES = [
    (nsem.Simulation1DElectricField, "sigma", lambda s: s),
    (nsem.Simulation1DElectricField, "mui", lambda s: np.full_like(s, 1 / mu_0)),
    (nsem.Simulation1DMagneticField, "rho", lambda s: 1 / s),
    (nsem.Simulation1DMagneticField, "mu", lambda s: np.full_like(s, mu_0)),
    (nsem.Simulation2DElectricField, "sigma", lambda s: s),
    (nsem.Simulation2DElectricField, "mui", lambda s: np.full_like(s, 1 / mu_0)),
    (nsem.Simulation2DMagneticField, "rho", lambda s: 1 / s),
    (nsem.Simulation2DMagneticField, "mu", lambda s: np.full_like(s, mu_0)),
]


def tensor_mesh(dim):
    h = [(100.0, 4, -1.3), (100.0, 8), (100.0, 4, 1.3)]
    return discretize.TensorMesh([h] * dim, "C" * dim)


def tree_mesh():
    mesh = discretize.TreeMesh(
        [np.full(16, 100.0), np.full(16, 100.0)],
        origin=[-800.0, -1200.0],
        diagonal_balance=True,
    )
    mesh.refine_box([-400, -600], [400, 0], -1, finalize=False)
    mesh.refine(mesh.max_level - 2)
    return mesh


def make_sim(sim_class, mesh=None, **kwargs):
    """A simulation with one frequency and no receivers."""
    if mesh is None:
        mesh = tensor_mesh(1 if sim_class in SIM_1D else 2)
    survey = nsem.Survey([nsem.sources.Planewave([], FREQ)])
    if not any(key.endswith("Map") for key in kwargs) and "sigma" not in kwargs:
        kwargs["sigma"] = conductivity(mesh)
    return sim_class(mesh, survey=survey, solver=get_default_solver(), **kwargs)


def conductivity(mesh):
    z = mesh.cell_centers if mesh.dim == 1 else mesh.cell_centers[:, 1]
    rng = np.random.default_rng(0)
    return np.where(z < 0, 1e-2, 1e-8) * np.exp(0.1 * rng.normal(size=mesh.n_cells))


class TestLogIntegral:
    @pytest.mark.parametrize("x", [-3.0, -1.0, 0.0, 0.5, 2.0, 4.0])
    def test_matches_quadrature(self, x):
        a, b = np.r_[-1.0, 0.5], np.r_[2.0, 3.0]
        out = _log_integral(np.r_[x], a, b)
        for i in range(2):
            expected = quad(
                lambda t: np.log(abs(x - t)), a[i], b[i], points=[x], limit=200
            )[0]
            np.testing.assert_allclose(out[i, 0], expected, rtol=1e-10, atol=1e-12)

    def test_shape(self):
        # one row per interval, one column per point
        out = _log_integral(np.zeros(5), np.r_[1.0, 2.0, 3.0], np.r_[2.0, 3.0, 4.0])
        assert out.shape == (3, 5)


class TestUpwardContinuation:
    def ops(self, n=8001, half_width=2e4):
        nodes = np.linspace(-half_width, half_width, n)
        x = 0.5 * (nodes[1:] + nodes[:-1])
        D, d_left, d_right = _upward_continuation_ops(
            x, np.diff(nodes), nodes[0], nodes[-1], FREQ
        )
        return nodes, x, D, d_left, d_right

    def test_constant_field_has_no_anomaly(self):
        _, _, D, d_left, d_right = self.ops(n=101)
        rows = D.sum(axis=1) + d_left + d_right
        np.testing.assert_allclose(rows, 0, atol=1e-12 * np.abs(D).max())

    @pytest.mark.parametrize("depth", [100.0, 300.0])
    def test_harmonic_field(self, depth):
        """For an electric field decaying upward from a line source at
        ``depth`` below the top, the operator gives the anomalous magnetic
        field ``dE/dz / (i omega mu_0)``."""
        nodes, x, D, d_left, d_right = self.ops()

        def e_field(x):
            return depth / (x**2 + depth**2)

        h = D @ e_field(x) + d_left * e_field(nodes[0]) + d_right * e_field(nodes[-1])
        expected = (x**2 - depth**2) / (x**2 + depth**2) ** 2
        expected = expected / (1j * omega(FREQ) * mu_0)
        core = np.abs(x) < 5 * depth
        np.testing.assert_allclose(
            h[core], expected[core], atol=5e-3 * np.abs(expected).max()
        )


class TestRobinOps:
    @pytest.mark.parametrize("sim_class", SIM_1D)
    def test_1d(self, sim_class):
        sim = make_sim(sim_class)
        P, K = sim._robin_ops
        # the bottom node, and the bottom cell with unit length
        assert P.shape == (1, sim.mesh.n_nodes) and P.nnz == 1 and P[0, 0] == 1.0
        assert K.shape == (1, sim.mesh.n_cells) and K.nnz == 1 and K[0, 0] == 1.0

    @pytest.mark.parametrize("sim_class", SIM_2D)
    @pytest.mark.parametrize("side_boundary", ["robin", "neumann"])
    @pytest.mark.parametrize("mesh_type", ["tensor", "tree"])
    def test_2d(self, sim_class, side_boundary, mesh_type):
        mesh = tensor_mesh(2) if mesh_type == "tensor" else tree_mesh()
        sim = make_sim(sim_class, mesh, side_boundary=side_boundary)
        P, K = sim._robin_ops
        edges = mesh.edges[P.indices]
        is_x = P.indices < mesh.n_edges_x
        on_bottom = is_x & (edges[:, 1] == mesh.nodes_y[0])
        on_side = ~is_x & np.isin(edges[:, 0], mesh.nodes_x[[0, -1]])
        # every bottom edge, and the side edges only with robin sides
        assert on_bottom.sum() == np.sum(mesh.boundary_edges[:, 1] == mesh.nodes_y[0])
        n_side = np.sum(np.isin(mesh.boundary_edges[:, 0], mesh.nodes_x[[0, -1]]))
        assert on_side.sum() == (n_side if side_boundary == "robin" else 0)
        assert np.all(on_bottom | on_side)
        # K holds each edge's length at the one cell that has the edge on its boundary
        assert np.all(np.diff(K.indptr) == 1)
        np.testing.assert_allclose(K.data, mesh.edge_lengths[P.indices])
        cells = K.indices
        lower = mesh.cell_centers[cells] - mesh.h_gridded[cells] / 2
        upper = mesh.cell_centers[cells] + mesh.h_gridded[cells] / 2
        # the edge lies on the cell's outer face, along the boundary
        face = np.where(edges[:, 0] == mesh.nodes_x[0], lower[:, 0], upper[:, 0])
        np.testing.assert_array_equal(lower[on_bottom, 1], edges[on_bottom, 1])
        np.testing.assert_array_equal(face[on_side], edges[on_side, 0])
        along = np.where(on_bottom, 0, 1)
        rows = np.arange(len(edges))
        assert np.all(lower[rows, along] < edges[rows, along])
        assert np.all(edges[rows, along] < upper[rows, along])

    @pytest.mark.parametrize("sim_class", SIM_2D)
    def test_side_boundary_setter_resets_ops(self, sim_class):
        sim = make_sim(sim_class)
        n_robin = sim._robin_ops[0].shape[0]
        sim.side_boundary = "neumann"
        assert sim._robin_ops[0].shape[0] < n_robin


class TestRobinCoefficient:
    @pytest.mark.parametrize("sim_class", SIM_1D + SIM_2D)
    def test_plane_wave(self, sim_class):
        """The admittance (E simulations) or impedance (H simulations) of a
        plane wave going out of the mesh, i k / mu or i k rho."""
        sim = make_sim(sim_class)
        sigma = conductivity(sim.mesh)
        k = np.sqrt(-1j * omega(FREQ) * mu_0 * sigma)
        if sim_class in (
            nsem.Simulation1DElectricField,
            nsem.Simulation2DElectricField,
        ):
            expected = 1j * k / mu_0
        else:
            expected = 1j * k / sigma
        np.testing.assert_allclose(sim._robin_coefficient(FREQ), expected, rtol=1e-12)

    @pytest.mark.parametrize("sim_class", SIM_1D + SIM_2D)
    def test_scalar_property(self, sim_class):
        """A scalar property is broadcast to every cell."""
        sim = make_sim(sim_class, sigma=1e-2)
        coef = sim._robin_coefficient(FREQ)
        assert coef.shape == (sim.mesh.n_cells,)
        np.testing.assert_allclose(coef, coef[0])

    @pytest.mark.parametrize("sim_class", SIM_1D + SIM_2D)
    def test_matrix(self, sim_class):
        sim = make_sim(sim_class)
        P, K = sim._robin_ops
        R = sim._robin_matrix(FREQ)
        expected = P.T @ sp.diags(K @ sim._robin_coefficient(FREQ)) @ P
        assert abs(R - expected).max() == 0
        assert abs(R - R.T).max() == 0
        # only on the robin nodes or edges
        assert set(R.nonzero()[0]) == set(P.indices)


class TestRobinDeriv:
    @pytest.mark.parametrize("sim_class, prop, value", PROPERTIES)
    def test_deriv_and_adjoint(self, sim_class, prop, value):
        mesh = tensor_mesh(1 if sim_class in SIM_1D else 2)
        sigma = conductivity(mesh)
        kwargs = {f"{prop}Map": maps.ExpMap(mesh)}
        if prop in ("mui", "mu"):
            kwargs["sigma"] = sigma
        sim = make_sim(sim_class, mesh, **kwargs)
        rng = np.random.default_rng(1)
        m0 = np.log(value(sigma)) + 0.1 * rng.normal(size=mesh.n_cells)
        n = sim._robin_ops[0].shape[1]
        u = rng.normal(size=n) + 1j * rng.normal(size=n)

        def fun(m):
            sim.model = m
            return (
                sim._robin_matrix(FREQ) @ u,
                lambda v: sim._robin_deriv(FREQ, prop, u, v, adjoint=False),
            )

        assert tests.check_derivative(fun, m0, num=3, plotIt=False, random_seed=2)

        sim.model = m0
        v = rng.normal(size=mesh.n_cells)
        w = rng.normal(size=n) + 1j * rng.normal(size=n)
        np.testing.assert_allclose(
            w @ sim._robin_deriv(FREQ, prop, u, v, adjoint=False),
            v @ sim._robin_deriv(FREQ, prop, u, w, adjoint=True),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("sim_class, prop, value", PROPERTIES)
    def test_multiple_columns(self, sim_class, prop, value):
        """Several perturbations as the columns of v, as in getADeriv."""
        mesh = tensor_mesh(1 if sim_class in SIM_1D else 2)
        sigma = conductivity(mesh)
        kwargs = {f"{prop}Map": maps.ExpMap(mesh)}
        if prop in ("mui", "mu"):
            kwargs["sigma"] = sigma
        sim = make_sim(sim_class, mesh, **kwargs)
        rng = np.random.default_rng(3)
        sim.model = np.log(value(sigma))
        n = sim._robin_ops[0].shape[1]
        u = rng.normal(size=n) + 1j * rng.normal(size=n)
        robin = functools.partial(sim._robin_deriv, FREQ, prop)
        system = functools.partial(getattr(sim, f"getADeriv_{prop}"), FREQ)
        for adjoint, n_rows in ((False, mesh.n_cells), (True, n)):
            V = rng.normal(size=(n_rows, 3))
            for deriv in (robin, system):
                columns = [deriv(u, V[:, i], adjoint=adjoint) for i in range(3)]
                np.testing.assert_allclose(
                    deriv(u, V, adjoint=adjoint), np.column_stack(columns)
                )

    @pytest.mark.parametrize("sim_class", SIM_1D + SIM_2D)
    def test_zero_without_map(self, sim_class):
        sim = make_sim(sim_class)
        prop = sim._robin_properties[0]
        assert isinstance(sim._robin_deriv(FREQ, prop, None, None, False), Zero)

    @pytest.mark.parametrize("sim_class", SIM_2D)
    def test_zero_with_custom_boundary_data(self, sim_class):
        mesh = tensor_mesh(2)
        bc = {FREQ: np.ones(mesh.boundary_edges.shape[0], dtype=complex)}
        key = "h_bc" if sim_class is nsem.Simulation2DElectricField else "e_bc"
        prop = sim_class._robin_properties[0]
        sim = make_sim(sim_class, mesh, **{f"{prop}Map": maps.ExpMap(mesh), key: bc})
        assert not sim._uses_robin
        assert isinstance(sim._robin_deriv(FREQ, prop, None, None, False), Zero)


class TestBoundaryOptions:
    @pytest.mark.parametrize(
        "sim_class, option",
        [(cls, "side_boundary") for cls in SIM_2D]
        + [(nsem.Simulation2DMagneticField, "top_boundary")],
    )
    def test_invalid(self, sim_class, option):
        with pytest.raises(ValueError):
            make_sim(sim_class, **{option: "dirichlet"})

    @pytest.mark.parametrize(
        "sim_class, option, value",
        [
            (nsem.Simulation2DElectricField, "side_boundary", "neumann"),
            (nsem.Simulation2DMagneticField, "side_boundary", "neumann"),
            (nsem.Simulation2DMagneticField, "top_boundary", "uniform"),
        ],
    )
    def test_setter_clears_factorization(self, sim_class, option, value):
        sim = make_sim(sim_class)
        sim.__dict__["Ainv"] = "a stale factorization"
        setattr(sim, option, value)
        assert "Ainv" not in sim.__dict__


class TestTopRobin:
    @pytest.mark.parametrize("mesh_type", ["tensor", "tree"])
    def test_top_edges(self, mesh_type):
        mesh = tensor_mesh(2) if mesh_type == "tensor" else tree_mesh()
        sim = make_sim(nsem.Simulation2DMagneticField, mesh)
        top, P_top = sim._top_edges
        np.testing.assert_array_equal(mesh.boundary_edges[top, 1], mesh.nodes_y[-1])
        assert np.all(np.diff(mesh.boundary_edges[top, 0]) > 0)
        np.testing.assert_array_equal(
            mesh.edges[P_top.indices], mesh.boundary_edges[top]
        )
        assert len(top) == np.sum(mesh.boundary_edges[:, 1] == mesh.nodes_y[-1])

    @pytest.mark.parametrize("mesh_type", ["tensor", "tree"])
    def test_ops(self, mesh_type):
        mesh = tensor_mesh(2) if mesh_type == "tensor" else tree_mesh()
        sim = make_sim(nsem.Simulation2DMagneticField, mesh)
        B, b_0, u_left, u_right = sim._top_robin_ops(FREQ)
        n_top = len(sim._top_edges[0])
        assert B.shape == (n_top, n_top)
        np.testing.assert_array_equal(B, B.T)
        np.testing.assert_allclose(b_0, B.sum(axis=1))
        # cached per frequency
        assert sim._top_robin_ops(FREQ)[0] is B

        # a laterally uniform field (H_x = 1 and the same electric field e at
        # both corners) has e along the whole top: the boundary integral of the
        # electric field, B h - b_0 - u_left e - u_right e, is the edge lengths
        # (with the orientation of M_bc) times e
        top, P_top = sim._top_edges
        weights = (P_top @ sim._M_bc[:, top]).diagonal()
        e = 0.3 - 0.7j
        integral = B @ np.ones(n_top) - b_0 - (u_left + u_right) * e
        np.testing.assert_allclose(integral, weights * e, rtol=1e-10)
        np.testing.assert_allclose(np.abs(weights), mesh.edge_lengths[P_top.indices])

    @pytest.mark.parametrize("top_boundary", ["robin", "uniform"])
    def test_side_columns(self, top_boundary):
        sim = make_sim(nsem.Simulation2DMagneticField, top_boundary=top_boundary)
        src = sim._sim_left.survey.source_list[0]
        for (e, h_top), f in zip(sim._side_columns(FREQ), sim.boundary_fields()):
            if top_boundary == "uniform":
                assert h_top == 1.0
            else:
                # the column's incident field, -h of the 1D simulation at its top
                assert h_top == -f[src, "h"][-1, 0]
            np.testing.assert_allclose(e, f[src, "e"][:, 0] / h_top)
