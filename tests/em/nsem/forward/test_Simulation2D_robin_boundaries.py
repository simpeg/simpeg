"""
Tests for the robin boundary conditions of the 2D NSEM simulations.
"""

import numpy as np
import pytest
import discretize
from scipy.constants import mu_0

from simpeg import maps, tests
from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import get_default_solver

FREQUENCIES = [0.1, 1.0]
LOCATIONS = np.c_[np.linspace(-1000.0, 1000.0, 5), np.zeros(5)]
SIMULATIONS = {
    "xy": nsem.Simulation2DElectricField,
    "yx": nsem.Simulation2DMagneticField,
}


def make_survey(orientation, locations=LOCATIONS, components=("app_res", "phase")):
    receivers = [
        nsem.receivers.Impedance(locations, orientation=orientation, component=c)
        for c in components
    ]
    if orientation == "yx":
        receivers += [
            nsem.receivers.Tipper(locations, orientation="zx", component=c)
            for c in ("real", "imag")
        ]
    return nsem.Survey([nsem.sources.Planewave(receivers, f) for f in FREQUENCIES])


def padded_mesh(side, air=10e3, dx=100.0, depth=100e3):
    """100 m core cells, padded at 1.3x out to the given distances."""

    def n_pad(dist):
        w, n = dx, 0
        while dist > 0:
            w *= 1.3
            dist -= w
            n += 1
        return n

    n_bottom = n_pad(depth)
    hx = [(dx, n_pad(side), -1.3), (dx, 30), (dx, n_pad(side), 1.3)]
    hz = [(dx, n_bottom, -1.3), (dx, 30), (dx, n_pad(air), 1.3)]
    mesh = discretize.TensorMesh([hx, hz], "C0")
    mesh.origin = np.r_[mesh.origin[0], -mesh.h[1][: n_bottom + 30].sum()]
    return mesh


def contact(cell_centers):
    x, z = cell_centers.T
    sigma = np.full(len(x), 1e-8)
    sigma[(z < 0) & (x < 0)] = 1e-1
    sigma[(z < 0) & (x >= 0)] = 1e-3
    sigma[z < -3000] = 1e-2
    return sigma


def app_res(mesh, orientation, **kwargs):
    sim = SIMULATIONS[orientation](
        mesh,
        survey=make_survey(orientation, components=("app_res",)),
        sigma=contact(mesh.cell_centers),
        solver=get_default_solver(),
        **kwargs,
    )
    # apparent resistivities only (the yx survey also has tipper data)
    return sim.dpred().reshape(len(FREQUENCIES), -1)[:, : len(LOCATIONS)]


def max_error(d, reference):
    return np.max(np.abs(d / reference - 1))


@pytest.mark.parametrize("orientation", ["xy", "yx"])
def test_boundary_options(orientation):
    sim_class = SIMULATIONS[orientation]
    mesh = padded_mesh(side=10e3)
    sigma = contact(mesh.cell_centers)
    sim = sim_class(
        mesh, survey=make_survey(orientation), sigma=sigma, solver=get_default_solver()
    )
    assert sim.side_boundary == "robin"
    if orientation == "yx":
        assert sim.top_boundary == "robin"
    with pytest.raises(ValueError):
        sim_class(mesh, survey=make_survey(orientation), side_boundary="dirichlet")

    # changing an option updates the system matrix and drops old factorizations
    A_robin = sim.getA(0.1)
    sim.fields()
    assert "Ainv" in sim.__dict__
    sim.side_boundary = "neumann"
    assert "Ainv" not in sim.__dict__
    assert abs(sim.getA(0.1) - A_robin).max() > 0

    # custom boundary data turns every robin term off
    bc = {f: np.ones(mesh.boundary_edges.shape[0], dtype=complex) for f in FREQUENCIES}
    key = "h_bc" if orientation == "xy" else "e_bc"
    sim = sim_class(mesh, survey=make_survey(orientation), sigma=sigma, **{key: bc})
    C = mesh.edge_curl
    w = 2 * np.pi * 0.1
    if orientation == "xy":
        expected = C.T @ sim.MccMui @ C + 1j * w * sim.MeSigma
    else:
        expected = C.T @ sim.MccRho @ C + 1j * w * sim.MeMu
    assert abs(sim.getA(0.1) - expected).max() <= 1e-12 * abs(expected).max()


@pytest.mark.parametrize("orientation", ["xy", "yx"])
def test_close_sides(orientation):
    """With a contact reaching sides only 10 km away, robin sides absorb what
    the 1D columns can't represent, and are much closer to distant sides."""
    reference = app_res(padded_mesh(side=2000e3), orientation)
    mesh = padded_mesh(side=10e3)
    err = {
        side: max_error(app_res(mesh, orientation, side_boundary=side), reference)
        for side in ("robin", "neumann")
    }
    assert err["robin"] < 0.25 * err["neumann"]
    assert err["robin"] < 0.02


def test_thin_air():
    """With distant sides, the robin top on a thin air layer reproduces a very
    thick one. The uniform top is several percent off with 5 km of air."""
    reference = app_res(
        padded_mesh(side=2000e3, air=2000e3), "yx", top_boundary="uniform"
    )
    thin = padded_mesh(side=2000e3, air=5e3)
    assert max_error(app_res(thin, "yx"), reference) < 1e-3
    assert max_error(app_res(thin, "yx", top_boundary="uniform"), reference) > 1e-2


@pytest.mark.parametrize(
    "orientation, prop, top_boundary",
    [
        ("xy", "sigma", None),
        ("xy", "mui", None),
        ("yx", "rho", "robin"),
        ("yx", "mu", "robin"),
        ("yx", "rho", "uniform"),
    ],
)
def test_derivatives(orientation, prop, top_boundary):
    """The robin terms depend on the model: the bottom and side terms through the
    system matrix, and the side columns (and the robin top) through the right
    hand side. The tree mesh has hanging nodes and coarse cells on the boundary."""
    mesh = discretize.TreeMesh(
        [np.full(32, 100.0), np.full(32, 100.0)],
        origin=[-1600.0, -2600.0],
        diagonal_balance=True,
    )
    mesh.refine_box([-800, -1200], [800, 0], -1, finalize=False)
    mesh.refine(mesh.max_level - 2)
    locations = np.c_[np.linspace(-600, 600, 5), np.zeros(5)]
    survey = make_survey(orientation, locations, components=("real", "imag"))
    sigma = contact(mesh.cell_centers)
    kwargs = {f"{prop}Map": maps.ExpMap(mesh)}
    if top_boundary is not None:
        kwargs["top_boundary"] = top_boundary
    if prop == "sigma":
        m0 = np.log(sigma)
    elif prop == "rho":
        m0 = -np.log(sigma)
    else:
        kwargs["sigma"] = sigma
        m0 = np.full(mesh.n_cells, np.log(mu_0) * (1 if prop == "mu" else -1))
    rng = np.random.default_rng(42)
    m0 = m0 + 0.05 * rng.normal(size=mesh.n_cells)
    sim = SIMULATIONS[orientation](
        mesh, survey=survey, solver=get_default_solver(), **kwargs
    )
    src = survey.source_list[0]
    freq = src.frequency
    u = rng.normal(size=mesh.n_edges) + 1j * rng.normal(size=mesh.n_edges)

    def system(m):
        sim.model = m
        return sim.getA(freq) @ u, lambda v: sim.getADeriv(freq, u, v)

    def rhs(m):
        sim.model = m
        return sim.getRHS(freq), lambda v: sim.getRHSDeriv(freq, src, v)

    for fun in (system, rhs):
        assert tests.check_derivative(fun, m0, num=3, plotIt=False, random_seed=42)

    sim.model = m0
    A = sim.getA(freq)
    assert abs(A - A.T).max() <= 1e-12 * abs(A).max()
    v = rng.normal(size=m0.size)
    w = rng.normal(size=mesh.n_edges) + 1j * rng.normal(size=mesh.n_edges)
    np.testing.assert_allclose(
        w @ sim.getADeriv(freq, u, v),
        v @ sim.getADeriv(freq, u, w, adjoint=True),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        w @ np.ravel(sim.getRHSDeriv(freq, src, v)),
        v @ np.ravel(sim.getRHSDeriv(freq, src, w, adjoint=True)),
        rtol=1e-7,
    )
    f = sim.fields(m0)
    d = rng.normal(size=survey.nD)
    np.testing.assert_allclose(
        d @ sim.Jvec(m0, v, f=f), v @ sim.Jtvec(m0, d, f=f), rtol=1e-4
    )
