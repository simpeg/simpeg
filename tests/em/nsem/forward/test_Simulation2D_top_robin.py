"""
Tests for the robin (upward continuation) top boundary of Simulation2DMagneticField.
"""

import numpy as np
import pytest
import discretize
from scipy.constants import mu_0

from simpeg import maps, tests
from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import get_default_solver

FREQUENCIES = [0.1, 1.0, 10.0]
LOCATIONS = np.c_[np.linspace(-1000.0, 1000.0, 5), np.zeros(5)]


def make_survey(locations=LOCATIONS, components=("app_res", "phase")):
    receivers = [
        nsem.receivers.Impedance(locations, orientation="yx", component=c)
        for c in components
    ]
    return nsem.Survey([nsem.sources.Planewave(receivers, f) for f in FREQUENCIES])


def padded_mesh(side, air, dx=100.0, depth=100e3):
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


def layered(cell_centers):
    z = cell_centers[:, 1]
    sigma = np.where(z < 0, 1e-2, 1e-8)
    sigma[z < -3000] = 1e-1
    return sigma


def dpred(mesh, model, top_boundary):
    sim = nsem.Simulation2DMagneticField(
        mesh,
        survey=make_survey(),
        sigma=model(mesh.cell_centers),
        top_boundary=top_boundary,
        solver=get_default_solver(),
    )
    return sim.dpred()


def test_default_top_boundary():
    mesh = discretize.TensorMesh([np.full(8, 100.0)] * 2, "CC")
    sim = nsem.Simulation2DMagneticField(mesh, survey=make_survey())
    assert sim.top_boundary == "robin"


def test_bad_top_boundary():
    mesh = discretize.TensorMesh([np.full(8, 100.0)] * 2, "CC")
    with pytest.raises(ValueError):
        nsem.Simulation2DMagneticField(
            mesh, survey=make_survey(), top_boundary="dirichlet"
        )


def test_layered_matches_uniform_top():
    """For a layered earth both tops describe the same plane wave."""
    mesh = padded_mesh(side=25e3, air=10e3)
    np.testing.assert_allclose(
        dpred(mesh, layered, "robin"), dpred(mesh, layered, "uniform"), rtol=1e-4
    )


def test_thin_air():
    """With distant sides, a thin air layer reproduces a very thick one.

    The uniform top needs the top far above the surface: with 5 km of air its
    apparent resistivities are several percent off.
    """
    reference = dpred(padded_mesh(side=2000e3, air=2000e3), contact, "uniform")
    thin = padded_mesh(side=2000e3, air=5e3)
    d_robin = dpred(thin, contact, "robin").reshape(len(FREQUENCIES), 2, -1)
    d_uniform = dpred(thin, contact, "uniform").reshape(len(FREQUENCIES), 2, -1)
    reference = reference.reshape(len(FREQUENCIES), 2, -1)
    err_robin = np.max(np.abs(d_robin[:, 0] / reference[:, 0] - 1))
    err_uniform = np.max(np.abs(d_uniform[:, 0] / reference[:, 0] - 1))
    assert err_robin < 1e-3
    assert err_uniform > 1e-2


@pytest.mark.parametrize("prop", ["rho", "sigma", "mu"])
@pytest.mark.parametrize("mesh_type", ["tensor", "tree"])
def test_derivatives(mesh_type, prop):
    if mesh_type == "tensor":
        hx = [(100.0, 6, -1.4), (100.0, 20), (100.0, 6, 1.4)]
        hz = [(100.0, 6, -1.4), (100.0, 16), (100.0, 4, 1.4)]
        mesh = discretize.TensorMesh([hx, hz], "C0")
        mesh.origin = np.r_[mesh.origin[0], -mesh.h[1][:22].sum()]
    else:
        mesh = discretize.TreeMesh(
            [np.full(32, 100.0), np.full(32, 100.0)],
            origin=[-1600.0, -2600.0],
            diagonal_balance=True,
        )
        mesh.refine_box([-800, -1200], [800, 0], -1, finalize=False)
        mesh.refine(mesh.max_level - 2)
    locations = np.c_[np.linspace(-600, 600, 5), np.zeros(5)]
    receivers = [
        nsem.receivers.Impedance(locations, orientation="yx", component=c)
        for c in ("real", "imag")
    ] + [
        nsem.receivers.Tipper(locations, orientation="zx", component=c)
        for c in ("real", "imag")
    ]
    survey = nsem.Survey([nsem.sources.Planewave(receivers, f) for f in (0.1, 10.0)])
    sigma = contact(mesh.cell_centers)
    kwargs = {f"{prop}Map": maps.ExpMap(mesh)}
    if prop == "rho":
        m0 = -np.log(sigma)
    elif prop == "sigma":
        m0 = np.log(sigma)
    else:
        kwargs["sigma"] = sigma
        m0 = np.full(mesh.n_cells, np.log(mu_0))
    sim = nsem.Simulation2DMagneticField(
        mesh,
        survey=survey,
        top_boundary="robin",
        solver=get_default_solver(),
        **kwargs,
    )
    rng = np.random.default_rng(3)
    m0 = m0 + 0.05 * rng.normal(size=mesh.n_cells)

    # the right hand side carries the side columns and the top
    src = survey.source_list[0]

    def rhs(m):
        sim.model = m
        return sim.getRHS(src.frequency), lambda v: sim.getRHSDeriv(
            src.frequency, src, v
        )

    assert tests.check_derivative(rhs, m0, num=3, plotIt=False, random_seed=42)
    sim.model = m0
    v = rng.normal(size=m0.size)
    w = rng.normal(size=mesh.n_edges) + 1j * rng.normal(size=mesh.n_edges)
    np.testing.assert_allclose(
        w @ sim.getRHSDeriv(src.frequency, src, v),
        v @ sim.getRHSDeriv(src.frequency, src, w, adjoint=True),
        rtol=1e-8,
    )

    # the system matrix must stay symmetric for the adjoint solves
    A = sim.getA(src.frequency)
    assert abs(A - A.T).max() <= 1e-12 * abs(A).max()

    f = sim.fields(m0)
    d = rng.normal(size=survey.nD)
    np.testing.assert_allclose(
        d @ sim.Jvec(m0, v, f=f), v @ sim.Jtvec(m0, d, f=f), rtol=1e-4
    )
