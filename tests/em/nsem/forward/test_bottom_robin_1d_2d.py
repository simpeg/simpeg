"""
Tests for the Robin (absorbing) bottom boundary of the 1D and 2D NSEM simulations.
"""

import numpy as np
import pytest
import discretize

from simpeg import maps
from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import get_default_solver

SIGMA = 1e-2
FREQUENCIES = [0.1, 1.0, 10.0]


def layered(z, depth=3000.0):
    sigma = np.full(z.shape, 1e-8)
    sigma[z < 0] = 1e-1
    sigma[z < -depth] = SIGMA
    return sigma


def mesh_1d(bottom_pads):
    hz = [(50.0, bottom_pads, -1.3), (50.0, 80), (50.0, 15, 1.3)]
    mesh = discretize.TensorMesh([hz], "0")
    mesh.origin = [-mesh.h[0][: bottom_pads + 80].sum()]
    return mesh


def dpred_1d(mesh, orientation):
    sim_class = {
        "xy": nsem.simulation.Simulation1DElectricField,
        "yx": nsem.simulation.Simulation1DMagneticField,
    }[orientation]
    survey = nsem.survey.Survey(
        [
            nsem.sources.Planewave(
                [
                    nsem.receivers.Impedance(
                        [[0.0]], orientation=orientation, component="app_res"
                    )
                ],
                f,
            )
            for f in FREQUENCIES
        ]
    )
    sim = sim_class(
        mesh,
        survey=survey,
        sigmaMap=maps.IdentityMap(mesh),
        solver=get_default_solver(),
    )
    return sim.dpred(layered(mesh.cell_centers))


@pytest.mark.parametrize("orientation", ["xy", "yx"])
def test_1d_insensitive_to_bottom_depth(orientation):
    """The bottom is shallower than a skin depth at the lowest frequency.

    With a reflecting bottom this changed the 0.1 Hz apparent resistivity by
    more than 10%. The Robin bottom absorbs the downgoing field instead.
    """
    shallow = dpred_1d(mesh_1d(bottom_pads=12), orientation)  # ~9 km deep
    deep = dpred_1d(mesh_1d(bottom_pads=30), orientation)  # ~570 km deep
    np.testing.assert_allclose(shallow, deep, rtol=1e-3)


@pytest.mark.parametrize(
    "orientation, boundaries",
    [
        ("xy", {"side_boundary": "robin"}),
        ("xy", {"side_boundary": "neumann"}),
        ("yx", {"top_boundary": "robin", "side_boundary": "robin"}),
        ("yx", {"top_boundary": "robin", "side_boundary": "neumann"}),
        ("yx", {"top_boundary": "uniform", "side_boundary": "robin"}),
        ("yx", {"top_boundary": "uniform", "side_boundary": "neumann"}),
    ],
)
def test_2d_matches_1d_on_shallow_mesh(orientation, boundaries):
    """For a layered earth, the 2D simulations reproduce their 1D boundary columns,
    whatever the top and side boundary conditions."""
    h = [(50.0, 12, -1.3), (50.0, 80), (50.0, 12, 1.3)]
    mesh = discretize.TensorMesh([h, h], "CC")
    mesh.origin = mesh.origin + np.r_[0, -mesh.origin[1] - mesh.h[1][:52].sum()]
    sim_class = {
        "xy": nsem.simulation.Simulation2DElectricField,
        "yx": nsem.simulation.Simulation2DMagneticField,
    }[orientation]
    survey = nsem.survey.Survey(
        [
            nsem.sources.Planewave(
                [
                    nsem.receivers.Impedance(
                        np.array([[0.0, 0.0]]),
                        orientation=orientation,
                        component="app_res",
                    )
                ],
                f,
            )
            for f in FREQUENCIES
        ]
    )
    sim = sim_class(
        mesh,
        survey=survey,
        sigmaMap=maps.IdentityMap(mesh),
        solver=get_default_solver(),
        **boundaries,
    )
    d_2d = sim.dpred(layered(mesh.cell_centers[:, 1]))

    mesh_1d_col = discretize.TensorMesh([mesh.h[1]], origin=[mesh.nodes_y[0]])
    d_1d = dpred_1d(mesh_1d_col, orientation)
    np.testing.assert_allclose(d_2d, d_1d, rtol=1e-5)


@pytest.mark.parametrize(
    "sim_class, prop",
    [
        (nsem.simulation.Simulation1DElectricField, "sigma"),
        (nsem.simulation.Simulation1DElectricField, "mu"),
        (nsem.simulation.Simulation1DMagneticField, "rho"),
        (nsem.simulation.Simulation1DMagneticField, "mu"),
    ],
)
def test_1d_derivatives(sim_class, prop):
    """The bottom robin term depends on the bottom cell's properties."""
    from scipy.constants import mu_0

    from simpeg import tests

    mesh = mesh_1d(bottom_pads=12)
    orientation = (
        "yx" if sim_class is nsem.simulation.Simulation1DElectricField else "xy"
    )
    receivers = [
        nsem.receivers.Impedance([[0.0]], orientation=orientation, component=c)
        for c in ("real", "imag")
    ]
    survey = nsem.survey.Survey(
        [nsem.sources.Planewave(receivers, f) for f in FREQUENCIES]
    )
    sigma = layered(mesh.cell_centers)
    kwargs = {f"{prop}Map": maps.ExpMap(mesh)}
    if prop == "sigma":
        m0 = np.log(sigma)
    elif prop == "rho":
        m0 = -np.log(sigma)
    else:
        kwargs["sigma"] = sigma
        m0 = np.full(mesh.n_cells, np.log(mu_0))
    rng = np.random.default_rng(42)
    m0 = m0 + 0.05 * rng.normal(size=mesh.n_cells)
    sim = sim_class(mesh, survey=survey, solver=get_default_solver(), **kwargs)

    def fun(m):
        return sim.dpred(m), lambda v: sim.Jvec(m, v)

    assert tests.check_derivative(fun, m0, num=3, plotIt=False, random_seed=42)
    f = sim.fields(m0)
    v = rng.normal(size=m0.size)
    w = rng.normal(size=survey.nD)
    # the magnetic field formulation is poorly conditioned with a resistive air
    # layer (its curl-curl term carries the air resistivity), so its adjoint is
    # only good to about 1e-8
    rtol = 1e-8 if sim_class is nsem.simulation.Simulation1DElectricField else 1e-6
    np.testing.assert_allclose(
        w @ sim.Jvec(m0, v, f=f), v @ sim.Jtvec(m0, w, f=f), rtol=rtol
    )
