"""
Tests for the robin (outgoing wave) sides of the 2D NSEM simulations.
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


def make_survey(orientation):
    receivers = [
        nsem.receivers.Impedance(LOCATIONS, orientation=orientation, component=c)
        for c in ("app_res", "phase")
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


def layered(cell_centers):
    z = cell_centers[:, 1]
    sigma = np.where(z < 0, 1e-2, 1e-8)
    sigma[z < -3000] = 1e-1
    return sigma


def app_res(mesh, model, orientation, **kwargs):
    sim = SIMULATIONS[orientation](
        mesh,
        survey=make_survey(orientation),
        sigma=model(mesh.cell_centers),
        solver=get_default_solver(),
        **kwargs,
    )
    return sim.dpred().reshape(len(FREQUENCIES), 2, -1)[:, 0]


@pytest.mark.parametrize("orientation", ["xy", "yx"])
def test_default_and_bad_side_boundary(orientation):
    mesh = discretize.TensorMesh([np.full(8, 100.0)] * 2, "CC")
    sim_class = SIMULATIONS[orientation]
    assert sim_class(mesh, survey=make_survey(orientation)).side_boundary == "robin"
    with pytest.raises(ValueError):
        sim_class(mesh, survey=make_survey(orientation), side_boundary="dirichlet")


@pytest.mark.parametrize("orientation", ["xy", "yx"])
def test_layered_matches_neumann_sides(orientation):
    """With laterally uniform sides there is nothing to absorb."""
    mesh = padded_mesh(side=10e3)
    np.testing.assert_allclose(
        app_res(mesh, layered, orientation, side_boundary="robin"),
        app_res(mesh, layered, orientation, side_boundary="neumann"),
        rtol=1e-5,
    )


@pytest.mark.parametrize("orientation", ["xy", "yx"])
def test_close_sides(orientation):
    """With a contact reaching sides only 10 km away, robin sides absorb what
    the 1D columns can't represent, and are much closer to distant sides."""
    reference = app_res(padded_mesh(side=2000e3), contact, orientation)
    mesh = padded_mesh(side=10e3)
    err = {
        side: np.max(
            np.abs(
                app_res(mesh, contact, orientation, side_boundary=side) / reference - 1
            )
        )
        for side in ("robin", "neumann")
    }
    assert err["robin"] < 0.25 * err["neumann"]
    assert err["robin"] < 0.02


@pytest.mark.parametrize("prop", ["sigma", "mui", "rho", "mu"])
@pytest.mark.parametrize("mesh_type", ["tensor", "tree"])
def test_system_matrix_derivative(mesh_type, prop):
    """The side (and bottom) robin terms depend on the model."""
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
    orientation = "xy" if prop in ("sigma", "mui") else "yx"
    sigma = contact(mesh.cell_centers)
    kwargs = {f"{prop}Map": maps.ExpMap(mesh)}
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
        mesh, survey=make_survey(orientation), solver=get_default_solver(), **kwargs
    )
    u = rng.normal(size=mesh.n_edges) + 1j * rng.normal(size=mesh.n_edges)

    def fun(m):
        sim.model = m
        return sim.getA(0.1) @ u, lambda v: sim.getADeriv(0.1, u, v)

    assert tests.check_derivative(fun, m0, num=3, plotIt=False, random_seed=42)

    sim.model = m0
    A = sim.getA(0.1)
    assert abs(A - A.T).max() <= 1e-12 * abs(A).max()
    v = rng.normal(size=m0.size)
    w = rng.normal(size=mesh.n_edges) + 1j * rng.normal(size=mesh.n_edges)
    np.testing.assert_allclose(
        w @ sim.getADeriv(0.1, u, v),
        v @ sim.getADeriv(0.1, u, w, adjoint=True),
        rtol=1e-10,
    )
