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


@pytest.mark.parametrize("orientation", ["xy", "yx"])
def test_2d_matches_1d_on_shallow_mesh(orientation):
    """For a layered earth, the 2D simulations reproduce their 1D boundary columns."""
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
    )
    d_2d = sim.dpred(layered(mesh.cell_centers[:, 1]))

    mesh_1d_col = discretize.TensorMesh([mesh.h[1]], origin=[mesh.nodes_y[0]])
    d_1d = dpred_1d(mesh_1d_col, orientation)
    np.testing.assert_allclose(d_2d, d_1d, rtol=1e-5)
