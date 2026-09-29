"""
Tests for the 2D NSEM simulations on a TreeMesh.
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


@pytest.mark.parametrize("orientation", ["xy", "yx"])
def test_2d_tree_mesh_matches_tensor_mesh(orientation):
    """The 1D side columns of a 2D TreeMesh are rebuilt from the mesh bottom.

    With an origin below zero, the side column cell widths used to be rebuilt
    from absolute positions, giving wrong boundary data on a TreeMesh.
    """
    h = [np.full(16, 100.0), np.full(64, 100.0)]
    origin = [-800.0, -4000.0]
    tensor = discretize.TensorMesh(h, origin=origin)
    tree = discretize.TreeMesh(h, origin=origin, diagonal_balance=True)
    tree.refine(-1)
    sim_class = {
        "xy": nsem.simulation.Simulation2DElectricField,
        "yx": nsem.simulation.Simulation2DMagneticField,
    }[orientation]
    locations = np.array([[-350.0, 0.0], [50.0, 0.0], [450.0, 0.0]])
    d = []
    for mesh in (tensor, tree):
        survey = nsem.survey.Survey(
            [
                nsem.sources.Planewave(
                    [
                        nsem.receivers.Impedance(
                            locations, orientation=orientation, component="app_res"
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
        d.append(sim.dpred(layered(mesh.cell_centers[:, 1])))
    np.testing.assert_allclose(d[1], d[0], rtol=1e-5)
